# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 600.0
_SAFETY_MARGIN = 30.0


def _interval_union_length(intervals):
    """
    Compute the total length of the union of 1D intervals.
    intervals: list of (start, end) tuples.
    Returns: float length of union.
    """
    if len(intervals) == 0:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    total = 0.0
    for s, e in merged:
        total += (e - s)
    return total


def _compute_volume_1d(base_positions, N=8):
    """
    Compute the 1D union length for given base positions.
    base_positions: array of length N, base position for each i in [0, N-1].
    Tube i at height c occupies [base[i] + c*i/N, base[i] + c*i/N + 1/N].
    We need the volume (length in 1D x height) of the union over c in [0,1].
    
    Actually, for the factorization V = Vx * Vy, we need the 2D area in (x,c) plane
    covered by the union of parallelograms. But since the problem factorizes,
    the 3D volume is (area_x) * (area_y) where area_x is the area in (x,c) plane.
    
    Wait, let me re-read. The union volume in 3D is the measure of points (x,y,c) 
    in the union of all tubes. Each tube k=(i,j) is a product of an x-interval 
    (depending on i and c) and a y-interval (depending on j and c).
    
    So the set is: {(x,y,c) : exists (i,j) such that x in [x_i(c), x_i(c)+1/N] 
    and y in [y_j(c), y_j(c)+1/N]}
    
    This is NOT a product set in general because the condition is "exists (i,j)".
    However, if we choose x_i depending only on i and y_j depending only on j,
    then the set is: {(x,y,c) : (x,c) in A AND (y,c) in B} where
    A = union over i of [x_i(c), x_i(c)+1/N] x {c}
    B = union over j of [y_j(c), y_j(c)+1/N] x {c}
    
    Wait, that's not quite right either. Let me think again.
    
    The union is: U = union_{i,j} (S_i x T_j) where S_i = {(x,c): x in [x_i + c*i/N, x_i + c*i/N + 1/N]}
    and T_j = {(y,c): y in [y_j + c*j/N, y_j + c*j/N + 1/N]}
    
    So U = {(x,y,c) : exists i,j such that (x,c) in S_i and (y,c) in T_j}
       = {(x,y,c) : (exists i: (x,c) in S_i) AND (exists j: (y,c) in T_j)}
       = (union_i S_i) x (union_j T_j) ... but wait, the c is shared!
    
    Actually, U = {(x,y,c) : (x,c) in A AND (y,c) in B} where A = union_i S_i and B = union_j T_j.
    This is NOT A x B because A and B are subsets of R^2 (with c as second coordinate).
    
    The volume is: integral_0^1 [measure of x such that (x,c) in A] * [measure of y such that (y,c) in B] dc
    = integral_0^1 L_A(c) * L_B(c) dc
    
    where L_A(c) is the 1D length of the union of intervals [x_i + c*i/N, x_i + c*i/N + 1/N] at height c.
    
    So if we choose x_i = f(i) and y_j = g(j), the volume is:
    V = integral_0^1 L_A(c) * L_B(c) dc
    
    This does NOT factor as Vx * Vy in general. However, if we can make L_A(c) and L_B(c) 
    constant (independent of c), then V = L_A * L_B * 1 = Vx * Vy.
    
    For the Keich construction, the union length is constant in c. Let's verify this.
    """
    # Compute L(c) for c in [0,1] by sampling
    # But for optimization, we need an efficient computation.
    # The union length L(c) is piecewise linear in c.
    # Each interval is [x_i + c*i/N, x_i + 1/N + c*i/N].
    # The breakpoints occur when endpoints of different intervals cross.
    # For small N, we can compute L(c) exactly.
    
    # For now, use a numerical integration approach with many samples.
    # This is expensive but N=8 is small.
    c_samples = np.linspace(0, 1, 100)
    lengths = np.zeros(len(c_samples))
    for k, c in enumerate(c_samples):
        intervals = [(base_positions[i] + c * i / N, base_positions[i] + c * i / N + 1.0 / N) 
                     for i in range(N)]
        lengths[k] = _interval_union_length(intervals)
    # Trapezoidal rule
    vol_2d = np.trapz(lengths, c_samples)
    return vol_2d


def _objective_1d(base_positions, N=8):
    """Objective function for 1D problem: minimize the 2D area under the union length curve."""
    return _compute_volume_1d(base_positions, N)


def _solve_1d(N=8, time_limit=580.0):
    """
    Solve the 1D interval packing problem using local optimization.
    """
    best_pos = None
    best_val = np.inf
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    n_starts = 20
    
    for start in range(n_starts):
        if time.time() - _START_TIME > time_limit:
            break
        
        # Random initial positions
        x0 = rng.uniform(-1, 1, N)
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            res = minimize(_objective_1d, x0, method='Nelder-Mead', 
                          args=(N,), options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_pos = res.x.copy()
        except Exception:
            pass
    
    # Also try the Keich construction as a starting point
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    try:
        res = minimize(_objective_1d, keich, method='Nelder-Mead', 
                      args=(N,), options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_pos = res.x.copy()
    except Exception:
        pass
    
    if best_pos is None:
        best_pos = keich.copy()
    
    return best_pos, best_val


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses the factorization insight: if base_x depends only on i and base_y depends 
    only on j, the 3D volume is integral_0^1 L_x(c) * L_y(c) dc where L_x and L_y 
    are the 1D union lengths at height c. We solve two independent 1D problems to 
    minimize this product.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    N = 8
    
    # Solve 1D problem for x-coordinates
    base_x, val_x = _solve_1d(N, time_limit=570.0)
    
    # Solve 1D problem for y-coordinates (same problem, so same solution)
    # But we can try different initializations for variety
    base_y, val_y = _solve_1d(N, time_limit=575.0)
    
    # If time is running out, fall back to Keich
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        pos_k = int(np.log2(N))
        j_terms = np.arange(1, pos_k + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        base_x = keich.copy()
        base_y = keich.copy()
    
    # Build the 64x2 position array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_full = base_x[i_flat]
    base_y_full = base_y[j_flat]
    positions = np.stack([base_x_full, base_y_full], axis=1)
    
    # Validate: check that positions are finite
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        pos_k = int(np.log2(N))
        j_terms = np.arange(1, pos_k + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        positions = np.stack([keich[i_flat], keich[j_flat]], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
