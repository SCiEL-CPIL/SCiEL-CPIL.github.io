"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_toeplitz


def _elapsed(t0):
    return time.time() - t0


def _raw_value(h):
    """Compute the objective value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _is_valid(h):
    """Check if h is a valid solution."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True


def _project_simplex(v, target_sum):
    """Project vector v onto the simplex {x >= 0, sum(x) = target_sum}."""
    n = len(v)
    # Clamp to [0, 1] first
    v = np.clip(v, 0, 1)
    # Now project onto sum = target_sum with bounds [0, 1]
    # Use water-filling algorithm
    def water_fill(v, s):
        """Find lambda such that sum(clip(v - lambda, 0, 1)) = s"""
        lo, hi = v.min() - 1, v.max() + 1
        
        def f(lam):
            return np.sum(np.clip(v - lam, 0, 1)) - s
        
        # Bisection
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        lam = (lo + hi) / 2
        return np.clip(v - lam, 0, 1)
    
    return water_fill(v, target_sum)


def _make_poisson_kernel():
    """Create the inverse of the discrete Laplacian (Poisson kernel).
    
    The discrete Poisson equation: -Δu = f where Δ is the 1D Laplacian.
    We solve (I - α*L) u = f for some smoothing parameter α.
    Actually, let's use a simple approach: the solution to -u'' = f with 
    Dirichlet BCs gives us a smooth function.
    
    For the discrete case, we'll use the matrix inverse of the Laplacian.
    """
    # Discrete Laplacian with Dirichlet BCs
    # L = tridiag(-1, 2, -1)
    n = N - 2  # interior points (we fix boundary to 0)
    if n <= 0:
        return np.eye(N)
    
    # Build tridiagonal matrix
    main = 2 * np.ones(n)
    off = -1 * np.ones(n - 1)
    
    # Use solve_toeplitz for efficiency
    # Actually, let's just build and invert for simplicity (N is small)
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2
        if i > 0:
            L[i, i-1] = -1
        if i < n-1:
            L[i, i+1] = -1
    
    return L


def construct():
    t0 = time.time()
    deadline = t0 + 145  # Safety margin before 150s limit
    
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_val = _raw_value(best_h)
    
    # Strategy: Represent h as solution to Poisson equation with sparse sources
    # h = base + K @ s where K is the Poisson kernel (smoothing operator)
    # and s is a sparse source vector (low-dimensional parameters)
    
    # We'll parameterize h using a small number of "source" points
    # h[i] = 0.5 + sum_j s_j * G(i, j) where G is the Green's function
    
    # The Green's function for the discrete Laplacian with Dirichlet BCs
    # G(i, j) = min(i, j) * (N - max(i, j)) / N for 1 <= i,j <= N-1
    # (with 0-indexing and appropriate adjustments)
    
    # Let's use a simpler approach: h = 0.5 + A @ z where A is a fixed 
    # basis matrix (e.g., low-frequency Fourier modes or polynomial basis)
    # and z is a small parameter vector. This ensures smoothness.
    
    # Actually, let's try the Poisson approach properly:
    # We want h to be smooth, so we represent it as the solution to -Δh = s
    # where s is sparse (few non-zero entries). Then h = K @ s where K is 
    # the inverse of the Laplacian.
    
    # Build the Poisson kernel K: K @ s = h (interior), with h[0]=h[N-1]=0
    # Actually, we need h to sum to N/2, so let's be more careful.
    
    # Alternative: use a basis of smooth functions that automatically satisfy
    # the constraints. Let's use cosine basis (discrete cosine transform).
    
    # h[i] = 0.5 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N)
    # This automatically gives smooth functions. The sum constraint:
    # sum(h) = N*0.5 + sum_k a_k * sum_i cos(2*pi*k*i/N)
    # For k not multiple of N, sum_i cos(2*pi*k*i/N) = 0
    # So sum(h) = N*0.5 automatically! Great.
    
    # And we need h[i] in [0, 1]. We'll enforce this via projection.
    
    # Let's use K = 5 or so basis functions (low frequencies)
    K_basis = 5
    i_idx = np.arange(N)
    
    # Build basis matrix: B[i, k] = cos(2*pi*k*i/N) for k=1..K_basis
    B = np.zeros((N, K_basis))
    for k in range(1, K_basis + 1):
        B[:, k-1] = np.cos(2 * np.pi * k * i_idx / N)
    
    def _h_from_params(z):
        """Reconstruct h from parameter vector z."""
        return 0.5 + B @ z
    
    def _objective(z):
        """Objective: raw value of h from parameters z."""
        h = _h_from_params(z)
        h = np.clip(h, 0, 1)  # Project to [0, 1]
        # Fix sum to exactly N/2
        h = _project_simplex(h, N / 2)
        return _raw_value(h)
    
    # Initial parameters
    z0 = np.zeros(K_basis)
    
    # Use SLSQP to minimize the objective
    try:
        res = minimize(
            _objective,
            z0,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if _elapsed(t0) < deadline:
            z_best = res.x
            h_candidate = _h_from_params(z_best)
            h_candidate = np.clip(h_candidate, 0, 1)
            h_candidate = _project_simplex(h_candidate, N / 2)
            if _is_valid(h_candidate):
                val = _raw_value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Try with more basis functions
    if _elapsed(t0) < deadline:
        K_basis2 = 10
        B2 = np.zeros((N, K_basis2))
        for k in range(1, K_basis2 + 1):
            B2[:, k-1] = np.cos(2 * np.pi * k * i_idx / N)
        
        def _h_from_params2(z):
            return 0.5 + B2 @ z
        
        def _objective2(z):
            h = _h_from_params2(z)
            h = np.clip(h, 0, 1)
            h = _project_simplex(h, N / 2)
            return _raw_value(h)
        
        z02 = np.zeros(K_basis2)
        try:
            res2 = minimize(
                _objective2,
                z02,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if _elapsed(t0) < deadline:
                z_best2 = res2.x
                h_candidate = _h_from_params2(z_best2)
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = _project_simplex(h_candidate, N / 2)
                if _is_valid(h_candidate):
                    val = _raw_value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Polish: try local optimization on the full 100-dim vector from best solution
    if _elapsed(t0) < deadline:
        h_start = best_h.copy()
        
        def _obj_full(h):
            h = np.clip(h, 0, 1)
            h = _project_simplex(h, N / 2)
            return _raw_value(h)
        
        def _grad_full(h):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros(N)
            for i in range(N):
                hp = h.copy()
                hp[i] += eps
                hp = np.clip(hp, 0, 1)
                hp = _project_simplex(hp, N / 2)
                hm = h.copy()
                hm[i] -= eps
                hm = np.clip(hm, 0, 1)
                hm = _project_simplex(hm, N / 2)
                g[i] = (_raw_value(hp) - _raw_value(hm)) / (2 * eps)
            return g
        
        try:
            res3 = minimize(
                _obj_full,
                h_start,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if _elapsed(t0) < deadline:
                h_candidate = res3.x
                h_candidate = np.clip(h_candidate, 0, 1)
                h_candidate = _project_simplex(h_candidate, N / 2)
                if _is_valid(h_candidate):
                    val = _raw_value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0, 1)
    best_h = _project_simplex(best_h, N / 2)
    
    return best_h
# EVOLVE-BLOCK-END
