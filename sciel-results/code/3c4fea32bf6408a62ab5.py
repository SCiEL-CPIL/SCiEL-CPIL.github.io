import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
dw = 1.0 / N
warnings.filterwarnings('ignore')

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution on a fine grid
    n_fine = 4 * N
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    
    x = np.linspace(-0.25, 0.25, N + 1)
    
    autoconv = np.zeros(n_fine)
    for idx, t in enumerate(t_fine):
        total = 0.0
        for i in range(N):
            if h[i] == 0:
                continue
            a = x[i]
            b = x[i+1]
            lo = max(a, t - 0.25)
            hi = min(b, t + 0.25)
            if lo >= hi:
                continue
            ts_lo = t - hi
            ts_hi = t - lo
            for j in range(N):
                if h[j] == 0:
                    continue
                c = x[j]
                d = x[j+1]
                olo = max(c, ts_lo)
                ohi = min(d, ts_hi)
                if olo < ohi:
                    total += h[i] * h[j] * (ohi - olo)
        autoconv[idx] = total
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_from_z(z):
    """Compute score from complex vector z where h_i = |z_i|^2."""
    h = np.abs(z)**2
    return _compute_score(h)

def _neg_score_from_z(z):
    """Negative score for minimization."""
    return -_score_from_z(z)

def _generate_sidon_set(n, size):
    """Generate a Sidon subset of {0,...,n-1} of given size using greedy."""
    S = []
    sums = set()
    for i in range(n):
        if len(S) >= size:
            break
        new_sums = set()
        valid = True
        for j in S + [i]:
            for k in S + [i]:
                if j <= k:
                    s = j + k
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
        if valid:
            S.append(i)
            sums |= new_sums
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes with complex parameterization
    for size in range(2, 14):
        if time.time() > deadline - 10:
            break
        
        S = _generate_sidon_set(N, size)
        if len(S) < size:
            continue
        
        # Initialize z with equal magnitudes at Sidon positions
        z0 = np.zeros(N, dtype=complex)
        for idx in S:
            z0[idx] = 1.0 / np.sqrt(size)
        
        # Also add small random phases to break symmetry
        np.random.seed(42 + size)
        z0[S] *= np.exp(1j * np.random.uniform(0, 2*np.pi, size))
        
        h0 = np.abs(z0)**2
        score0 = _compute_score(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Optimize over z (complex -> 2*N real variables)
        try:
            def _obj_flat(z_flat):
                z = z_flat[:N].astype(complex) + 1j * z_flat[N:]
                return _neg_score_from_z(z)
            
            z0_flat = np.concatenate([z0.real, z0.imag])
            
            result = minimize(_obj_flat, z0_flat, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12})
            
            z_trial = result.x[:N].astype(complex) + 1j * result.x[N:]
            h_trial = np.abs(z_trial)**2
            # Keep only Sidon positions
            h_trial[~np.isin(np.arange(N), S)] = 0.0
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Also try the original 2-element support
    h0 = np.zeros(N)
    h0[0] = 2.0/3.0
    h0[1] = 1.0/3.0
    score0 = _compute_score(h0)
    if score0 > best_score:
        best_score = score0
        best_h = h0.copy()
    
    # Final polish using complex parameterization
    if time.time() < deadline - 5:
        try:
            # Convert best_h to z
            z_init = np.sqrt(best_h) + 0j
            # Add small imaginary part to break symmetry
            np.random.seed(123)
            z_init += 1j * 0.01 * np.random.randn(N)
            
            def _obj_flat(z_flat):
                z = z_flat[:N].astype(complex) + 1j * z_flat[N:]
                return _neg_score_from_z(z)
            
            z0_flat = np.concatenate([z_init.real, z_init.imag])
            
            result = minimize(_obj_flat, z0_flat, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14})
            
            z_trial = result.x[:N].astype(complex) + 1j * result.x[N:]
            h_trial = np.abs(z_trial)**2
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
