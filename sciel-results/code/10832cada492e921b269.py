import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes to minimize the union volume.
    Uses the product structure to reduce the problem to optimizing 1D interval unions.
    Uses a differentiable Inclusion-Exclusion principle with soft-min/max/plus approximations.
    """
    N = 8
    w = 1.0 / N
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    U0 = keich.copy()

    # Precompute subsets for Inclusion-Exclusion
    # subsets is a list of lists of indices
    subsets = []
    for mask in range(1, 1 << N):
        indices = [i for i in range(N) if (mask >> i) & 1]
        subsets.append(indices)
        
    # Store signs: (-1)^(|S|+1)
    signs = []
    for s in subsets:
        signs.append(1.0 if len(s) % 2 == 1 else -1.0)
    signs = np.array(signs)

    def softplus(x, lam):
        # stable softplus: log(1 + exp(x)) / lam
        # For large x, exp(x) overflows. Use logaddexp(0, x)
        return np.logaddexp(0, x * lam) / lam

    def softmin_vec(v, lam):
        # -softmax(-v)
        # stable: - (max + log(sum(exp(-v - max)))) / lam
        v = np.asarray(v)
        m = np.max(-v)
        return -(m + np.log(np.sum(np.exp(-v - m)))) / lam

    def softmax_vec(v, lam):
        v = np.asarray(v)
        m = np.max(v)
        return (m + np.log(np.sum(np.exp(v - m)))) / lam

    def union_length_soft(U, c, lam=50.0):
        """
        Computes the soft union length of N intervals [U_i + c*i/N, U_i + c*i/N + w].
        """
        starts = U + c * np.arange(N) / N
        ends = starts + w
        
        L = 0.0
        for idx, s in enumerate(subsets):
            s_arr = np.array(s)
            s_starts = starts[s_arr]
            s_ends = ends[s_arr]
            
            min_end = softmin_vec(s_ends, lam)
            max_start = softmax_vec(s_starts, lam)
            
            inter_len = softplus(min_end - max_start, lam)
            L += signs[idx] * inter_len
            
        return L

    def objective(U):
        """
        Minimizes the integral of L(c)^2 over c in [0, 1].
        """
        # Use a fine grid for integration
        c_vals = np.linspace(0, 1, 100)
        total = 0.0
        # lam can be tuned. 50 is a reasonable start.
        lam = 50.0 
        for c in c_vals:
            l = union_length_soft(U, c, lam)
            total += l * l
        return total / len(c_vals)

    # Optimize U
    bounds = [(-1.0, 1.0)] * N
    # L-BFGS-B can handle the smooth objective now
    res = minimize(objective, U0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    U_opt = res.x

    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = U_opt[i_flat]
    base_y = U_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(positions)):
        # Fallback to seed if optimization failed
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
