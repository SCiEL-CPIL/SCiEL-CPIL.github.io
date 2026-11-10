"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import solve_ivp, quad
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # 8s safety margin before 90s kill

# --- Core physics functions ---

def _omega_m(a, Om0, w0, wa):
    """Compute matter density parameter Omega_m(a) for CPL dark energy."""
    Omd0 = 1.0 - Om0
    a = max(a, 1e-10)
    if abs(Omd0) < 1e-15:
        E2 = Om0 * a**(-3)
    else:
        E2 = Om0 * a**(-3) + Omd0 * a**(-3.0 * (1.0 + w0 + wa))
    E2 = max(E2, 1e-30)
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
    if Om_a <= 0:
        return 0.0
    return Om_a ** gamma


def _reference_f(a, Om0, w0, wa):
    """
    Compute reference f(a) = d ln D / d ln a using the second-order ODE.
    Integrates the Riccati equation for f from a=0.01 to a.
    """
    Om_a = _omega_m(a, Om0, w0, wa)
    if a <= 0.01:
        return Om_a ** 0.55
    
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
    
    Om_init = _omega_m(0.01, Om0, w0, wa)
    f_init = Om_init ** 0.55
    
    try:
        sol = solve_ivp(ode_f, [x_start, x_end], [f_init], rtol=1e-9, atol=1e-11, max_step=0.05)
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
        val, _ = quad(integrand, np.log(a), 0.0, limit=100)
        return np.exp(-val)
    except Exception:
        return 1.0


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
_A_GRID = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])

# Pre-computed reference data
_REF_DATA_CACHE = []


def _precompute_references():
    """Pre-compute reference f and D for all validation models and grid points."""
    global _REF_DATA_CACHE
    _REF_DATA_CACHE = []
    
    for Om0, w0, wa in _VALIDATION_MODELS:
        f_refs = np.zeros(len(_A_GRID))
        D_refs = np.zeros(len(_A_GRID))
        
        for i, a in enumerate(_A_GRID):
            if time.time() > _DEADLINE - 10.0:
                f_refs[i] = _omega_m(a, Om0, w0, wa) ** 0.55
                D_refs[i] = 1.0
            else:
                f_refs[i] = _reference_f(a, Om0, w0, wa)
                D_refs[i] = _reference_D_ratio(a, Om0, w0, wa)
        
        _REF_DATA_CACHE.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'f_refs': f_refs, 'D_refs': D_refs
        })


def _compute_max_error(coeffs):
    """Compute maximum relative error across all models and grid points."""
    max_err = 0.0
    
    for i, (Om0, w0, wa) in enumerate(_VALIDATION_MODELS):
        ref_data = _REF_DATA_CACHE[i]
        
        for j, a in enumerate(_A_GRID):
            # Formula f
            f_form = _f_formula(a, Om0, w0, wa, coeffs)
            f_ref = ref_data['f_refs'][j]
            if abs(f_ref) > 1e-15:
                err_f = abs(f_form - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
            
            # Formula D: D(a)/D(1) = exp(-integral_ln(a)^0 f_formula(exp(x)) dx)
            if a < 1.0:
                def integrand_formula(x):
                    a_x = np.exp(x)
                    return _f_formula(a_x, Om0, w0, wa, coeffs)
                
                try:
                    val_formula, _ = quad(integrand_formula, np.log(a), 0.0, limit=50)
                    D_form = np.exp(-val_formula)
                except Exception:
                    D_form = 1.0
                
                D_ref = ref_data['D_refs'][j]
                if abs(D_ref) > 1e-15:
                    err_D = abs(D_form - D_ref) / abs(D_ref)
                    max_err = max(max_err, err_D)
    
    return max_err


def _gamma_in_range(coeffs, Om0, w0, wa):
    """Check if gamma stays in [0.1, 1.2] for all a in a dense grid."""
    a_dense = np.linspace(0.01, 1.0, 100)
    for a in a_dense:
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


def _run_optimization():
    """Run optimization to find best coefficients."""
    # Pre-compute references once
    _precompute_references()
    
    # Start from the seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_obj = _compute_max_error(x0)
    
    # Multiple starting points for diversity
    starts = [
        x0,
        np.array([0.55, 0.1, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.05, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.05]),
        np.array([0.55, 0.05, 0.05, 0.0, 0.0, 0.0]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 5.0:
            break
        
        try:
            result = minimize(
                _compute_max_error,
                start,
                method='Nelder-Mead',
                options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-7}
            )
            
            if result.fun < best_obj and _is_valid(result.x):
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
