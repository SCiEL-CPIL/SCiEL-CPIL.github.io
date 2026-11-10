"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad
import time

# EVOLVE-BLOCK-START

# --- Core physics functions ---

def _omega_m(a, Om0, w0, wa):
    """Compute matter density parameter Omega_m(a) for CPL dark energy."""
    Omd0 = 1.0 - Om0
    if abs(Omd0) < 1e-15:
        E2 = Om0 * a**(-3)
    else:
        E2 = Om0 * a**(-3) + Omd0 * a**(-3.0 * (1.0 + w0 + wa))
    return Om0 * a**(-3) / E2


def _gamma_formula(a, Om0, w0, wa, coeffs):
    """Compute gamma from the six-coefficient formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma


def _f_formula(a, Om0, w0, wa, coeffs):
    """Compute f = Omega_m(a)^gamma from the formula."""
    gamma = _gamma_formula(a, Om0, w0, wa, coeffs)
    Om_a = _omega_m(a, Om0, w0, wa)
    return Om_a ** gamma


def _reference_f(a, Om0, w0, wa):
    """
    Compute reference f(a) = d ln D / d ln a using the second-order ODE.
    Uses a high-accuracy numerical integration of the Riccati equation for f.
    """
    Om_a = _omega_m(a, Om0, w0, wa)
    Om_de_a = 1.0 - Om_a
    w_a = w0 + wa * (1.0 - a)
    Hprime_over_H = 1.5 * (Om_a + (1.0 + w_a) * Om_de_a)
    
    def ode_f(x, y):
        a_x = np.exp(x)
        Om_x = _omega_m(a_x, Om0, w0, wa)
        Om_de_x = 1.0 - Om_x
        w_x = w0 + wa * (1.0 - a_x)
        Hprime_H = 1.5 * (Om_x + (1.0 + w_x) * Om_de_x)
        f_val = y[0]
        dfdx = -(f_val**2 + f_val * (3.0 + Hprime_H) - 1.5 * Om_x)
        return [dfdx]
    
    x_start = np.log(0.01)
    x_end = np.log(a)
    if x_end <= x_start:
        return Om_a ** 0.55
    
    Om_init = _omega_m(0.01, Om0, w0, wa)
    f_init = Om_init ** 0.55
    
    try:
        sol = solve_ivp(ode_f, [x_start, x_end], [f_init], rtol=1e-8, atol=1e-10, max_step=0.1)
        if sol.success:
            return sol.y[0, -1]
        else:
            return Om_a ** 0.55
    except Exception:
        return Om_a ** 0.55


def _reference_D_ratio(a, Om0, w0, wa):
    """
    Compute normalized growth factor D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    """
    if a >= 1.0:
        return 1.0
    
    def integrand(x):
        a_x = np.exp(x)
        return _reference_f(a_x, Om0, w0, wa)
    
    try:
        val, _ = quad(integrand, np.log(a), 0.0, limit=50)
        return np.exp(-val)
    except Exception:
        return 1.0


def _compute_errors_fast(coeffs, ref_data, a_grid):
    """Compute relative errors for f and D over the a-grid using pre-computed references."""
    f_errors = []
    D_errors = []
    
    # ref_data is a list of dicts for each model: {'a': ..., 'f_ref': ..., 'D_ref': ...}
    # We need to match the model index. The caller passes the specific ref_data for the current model.
    
    for i, a in enumerate(a_grid):
        # Formula values
        f_form = _f_formula(a, ref_data['Om0'], ref_data['w0'], ref_data['wa'], coeffs)
        
        # Reference values (pre-computed)
        f_ref = ref_data['f_refs'][i]
        D_ref = ref_data['D_refs'][i]
        
        # For D formula: D(a)/D(1) = exp(-integral_ln(a)^0 f_formula(exp(x)) dx)
        def integrand_formula(x):
            a_x = np.exp(x)
            return _f_formula(a_x, ref_data['Om0'], ref_data['w0'], ref_data['wa'], coeffs)
        
        try:
            val_formula, _ = quad(integrand_formula, np.log(a), 0.0, limit=50)
            D_form = np.exp(-val_formula)
        except Exception:
            D_form = 1.0
        
        # Relative errors
        if abs(f_ref) > 1e-15:
            f_errors.append(abs(f_form - f_ref) / abs(f_ref))
        if abs(D_ref) > 1e-15:
            D_errors.append(abs(D_form - D_ref) / abs(D_ref))
    
    return max(f_errors + D_errors) if (f_errors or D_errors) else 1.0


# --- Optimization setup ---

# Validation cosmologies: (Om0, w0, wa)
_VALIDATION_MODELS = [
    (0.30, -1.0, 0.0),    # flat LCDM
    (0.30, -0.9, 0.2),    # CPL with positive wa
    (0.30, -1.1, -0.3),   # CPL with negative wa
    (0.27, -0.95, 0.1),   # non-flat-ish
    (0.32, -1.0, 0.15),   # another CPL
]

# Scale factors to evaluate
_A_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

# Global deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin before 90s kill


# Global cache for pre-computed reference data
_REF_DATA_CACHE = []

def _objective(coeffs):
    """Max relative error across all validation models using cached references."""
    max_err = 0.0
    for i, (Om0, w0, wa) in enumerate(_VALIDATION_MODELS):
        ref_data = _REF_DATA_CACHE[i]
        err = _compute_errors_fast(coeffs, ref_data, _A_GRID)
        max_err = max(max_err, err)
    return max_err


def _gamma_in_range(coeffs, Om0, w0, wa):
    """Check if gamma stays in [0.1, 1.2] for all a in grid."""
    for a in _A_GRID:
        gamma = _gamma_formula(a, Om0, w0, wa, coeffs)
        if gamma < 0.1 or gamma > 1.2:
            return False
    return True


def _is_valid(coeffs):
    """Check coefficients are in [-10,10] and gamma in [0.1,1.2]."""
    if any(c < -10 or c > 10 for c in coeffs):
        return False
    for Om0, w0, wa in _VALIDATION_MODELS:
        if not _gamma_in_range(coeffs, Om0, w0, wa):
            return False
    return True


def _precompute_references():
    """Pre-compute reference f and D for all validation models and grid points."""
    global _REF_DATA_CACHE
    _REF_DATA_CACHE = []
    
    for Om0, w0, wa in _VALIDATION_MODELS:
        f_refs = np.zeros(len(_A_GRID))
        D_refs = np.zeros(len(_A_GRID))
        
        for i, a in enumerate(_A_GRID):
            f_refs[i] = _reference_f(a, Om0, w0, wa)
            D_refs[i] = _reference_D_ratio(a, Om0, w0, wa)
        
        _REF_DATA_CACHE.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'f_refs': f_refs, 'D_refs': D_refs
        })

def _run_optimization():
    """Run optimization to find best coefficients."""
    # Pre-compute references once
    _precompute_references()
    
    # Start from the seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_obj = _objective(x0)
    
    # Try multiple starting points with limited iterations
    starts = [
        x0,
        np.array([0.5, 0.1, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.05, 0.0, 0.0]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 5.0:
            break
        
        try:
            result = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            
            if result.fun < best_obj:
                if _is_valid(result.x):
                    best_coeffs = result.x.copy()
                    best_obj = result.fun
        except Exception:
            pass
    
    return best_coeffs.tolist()


def construct():
    """Return the six growth-index coefficients."""
    try:
        return _run_optimization()
    except Exception:
        # Fallback to seed if optimization fails
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END
