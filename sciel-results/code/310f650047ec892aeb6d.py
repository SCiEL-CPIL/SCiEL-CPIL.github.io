# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n
    
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    area = union_len.mean()
    
    return area

def _solve_lp_for_order(order, max_iter=1000):
    """
    Solves for x_i such that the order of L_i(y) is consistent with 'order' 
    at a representative set of heights, minimizing a proxy for the union area.
    
    We formulate this as an LP.
    Variables: x_0, ..., x_{n-1}
    
    The union length at height y is w + sum_{j=0}^{n-2} min(L_{order[j+1]}(y) - L_{order[j]}(y), w).
    min(g, w) is concave. Minimizing a sum of concave functions is generally non-convex.
    HOWEVER, if we assume that the "gaps" L_{j+1} - L_j are always <= w (i.e., the triangles overlap significantly or touch), 
    then min(g, w) = g.
    In that regime, Union Length = w + sum (L_{j+1} - L_j) = w + L_{last} - L_{first}.
    This simplifies the problem significantly. The area becomes roughly the span of the triangles plus a constant.
    
    Let's check if the "overlap" assumption holds for good solutions.
    For n=64, w(y) ranges from 1/64 to 0.
    If the triangles are tightly packed, the gaps are small.
    
    If we assume the "tight packing" regime (gaps < w), the objective is linear in x.
    Objective: Minimize Area ~ E_y [ w(y) + L_{max}(y) - L_{min}(y) ]
    
    L_{max}(y) = x_{order[-1]} + a_{order[-1]} y
    L_{min}(y) = x_{order[0]} + a_{order[0]} y
    
    Area ~ E_y [ (1-y)/n + x_{last} + a_{last} y - (x_{first} + a_{first} y) ]
    Area ~ C + x_{last} - x_{first} + (a_{last} - a_{first}) E[y]
    
    This suggests we just want to minimize x_{last} - x_{first}, which is 0 if we can make them equal.
    But we must maintain the order: x_{order[k]} + a_{order[k]} y <= x_{order[k+1]} + a_{order[k+1]} y for all y in [0,1].
    This implies x_{order[k+1]} - x_{order[k]} >= max_y (a_{order[k]} - a_{order[k+1]}) y.
    Since a_i are increasing with i, if order is sorted by a (i.e., identity), a_{order[k]} < a_{order[k+1]}, so max is 0 (at y=0).
    But if order is NOT sorted by a, say a_i > a_j but i comes before j in order (meaning L_i < L_j), 
    then we need x_j - x_i >= (a_i - a_j) * 1.
    
    So, for a fixed permutation `order` of indices 0..63:
    Constraints:
    For each k in 0..62:
    x_{order[k+1]} - x_{order[k]} >= max_{y in [0,1]} (a_{order[k]} - a_{order[k+1]}) y
    
    Let delta_a = a_{order[k]} - a_{order[k+1]}.
    If delta_a > 0, constraint is x_{order[k+1]} - x_{order[k]} >= delta_a.
    If delta_a <= 0, constraint is x_{order[k+1]} - x_{order[k]} >= 0.
    
    Objective: Minimize x_{order[-1]} - x_{order[0]} (proxy for span/area in tight regime).
    
    This is a simple LP!
    We can solve this for any permutation.
    The optimal permutation is likely the one that minimizes the total "cost" of these constraints, 
    which corresponds to sorting the a_i's in a way that minimizes the sum of positive differences?
    Actually, if we sort a_i in increasing order, all delta_a <= 0, so constraints are just x_{next} >= x_{curr}.
    Min x_last - x_first = 0.
    This implies x_i = 0 for all i?
    If x_i = 0 for all i, then L_i(y) = a_i y.
    Order of L_i(y) is always 0, 1, ..., 63 (since a_i increasing).
    Gaps: L_{i+1} - L_i = (a_{i+1} - a_i) y = (1/64) y.
    w(y) = (1-y)/64.
    Is gap < w?
    (y/64) < (1-y)/64  => y < 1-y => y < 0.5.
    So for y < 0.5, they overlap/touch. For y > 0.5, they are separated.
    The "tight packing" assumption fails for y > 0.5.
    
    So the LP solution for the identity order gives x_i = 0.
    Let's calculate the area for x_i = 0.
    This is the "standard" Kakeya needle position (all starting at origin).
    
    The Keich construction does something different. It permutes the order.
    The bit-reversal order creates a "fractal" interleaving that keeps the gaps small across all scales.
    
    Strategy:
    1. Generate a few candidate permutations (Bit-reversal, Random, Sorted).
    2. For each, solve the LP to get x.
    3. Evaluate the true area.
    4. Use the best x as a seed for a short, focused local refinement (e.g., Nelder-Mead or Gradient Descent) 
       to handle the non-convex parts (where gaps > w).
    
    """
    n = 64
    a = (np.arange(n) + 1) / n
    
    # LP Variables: x_0 ... x_{n-1}
    # Objective: c^T x where c is 0 except c[order[-1]]=1, c[order[0]]=-1
    c = np.zeros(n)
    c[order[-1]] = 1
    c[order[0]] = -1
    
    A_ub = []
    b_ub = []
    
    for k in range(n - 1):
        i_curr = order[k]
        i_next = order[k + 1]
        delta_a = a[i_curr] - a[i_next]
        
        # Constraint: x_next - x_curr >= delta_a
        # -x_next + x_curr <= -delta_a
        row = np.zeros(n)
        row[i_curr] = 1
        row[i_next] = -1
        A_ub.append(row)
        b_ub.append(-delta_a)
        
    if not A_ub:
        return np.zeros(n)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if res.success:
        return res.x
    else:
        return np.zeros(n)

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    n = 64
    
    # 1. Generate Candidate Orders
    orders = []
    
    # Identity Order (0, 1, ..., 63)
    orders.append(np.arange(n))
    
    # Bit-Reversal Order (Keich-inspired)
    k = int(np.log2(n))
    def bit_reverse(i, k):
        r = 0
        for j in range(k):
            if (i >> j) & 1:
                r |= (1 << (k - 1 - j))
        return r
    orders.append(np.array([bit_reverse(i, k) for i in range(n)]))
    
    # Reverse Bit-Reversal
    orders.append(np.array([bit_reverse(n - 1 - i, k) for i in range(n)]))
    
    # Random Permutations
    for _ in range(10):
        orders.append(np.random.permutation(n))
        
    # 2. Solve LP for each order and evaluate
    best_x = np.zeros(n)
    best_area = np.inf
    
    for order in orders:
        x_candidate = _solve_lp_for_order(order)
        if np.all(np.isfinite(x_candidate)):
            area = _calc_area(x_candidate)
            if area < best_area:
                best_area = area
                best_x = x_candidate.copy()
                
    # 3. Refine the best candidate using a fast gradient-based method
    # The LP solution is a local minimum in the "tight packing" regime.
    # We use Nelder-Mead or L-BFGS-B to escape local minima caused by the "gap > w" regions.
    
    # Precompute for fast gradient
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    
    def fast_area_and_grad(x_curr):
        L = x_curr[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        
        # Gradient of min(g, w) w.r.t g is 1 if g < w, 0 if g > w
        # d(union_len)/dx_i involves the positions in the sorted order.
        # This is complex to compute analytically fast.
        # Let's use finite differences or just Nelder-Mead which doesn't need gradients.
        
        contrib = np.minimum(gaps, w[None, :])
        union_len = w + contrib.sum(axis=0)
        area = union_len.mean()
        return area
        
    # Use scipy.optimize.minimize with Nelder-Mead
    # Nelder-Mead is robust for non-smooth functions.
    # We limit the time.
    from scipy.optimize import minimize
    
    def objective(x):
        return _calc_area(x)
        
    x0 = best_x.copy()
    
    # Set a time limit for the optimizer
    # We check time in a callback
    time_out = [False]
    def callback(xk):
        if time.time() > deadline - 30:
            time_out[0] = True
            return True # Stop optimization
        
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8},
                       callback=callback)
        if res.success or res.fun < best_area:
            if np.all(np.isfinite(res.x)):
                area_res = res.fun
                if area_res < best_area:
                    best_area = area_res
                    best_x = res.x.copy()
    except Exception as e:
        pass # If optimizer fails, keep LP solution
        
    # Final safety check
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n)
        
    return best_x

# EVOLVE-BLOCK-END
