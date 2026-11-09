# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _refine_weights(P, idx_a, idx_b, idx_c, idx_d, deadline):
    """
    Local refinement of the weights on the non-zero support.
    We optimize the weights w_i for i in support to maximize D/M.
    """
    support = np.where(P.ravel() > 1e-12)[0]
    if len(support) == 0:
        return P
    
    w = P.ravel()[support]
    w = w / w.sum()
    
    # We need gradients of D and M w.r.t w
    # D = H(qc), M = max(Ha, Hb, Hd)
    # Let's use a simple iterative projection method or small L-BFGS on weights
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    # Map support indices to bin indices
    sup_a = idx_a[support]
    sup_b = idx_b[support]
    sup_c = idx_c[support]
    sup_d = idx_d[support]
    
    # Sparse bincount logic for speed
    def get_q(idx_map, w, length):
        q = np.zeros(length)
        np.add.at(q, idx_map, w)
        return q
    
    def calc_val_grad(w):
        qa = get_q(sup_a, w, la)
        qb = get_q(sup_b, w, lb)
        qc = get_q(sup_c, w, lc)
        qd = get_q(sup_d, w, ld)
        
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return -1e9, np.zeros_like(w)
        
        val = Hc / M
        
        # Gradients are tricky with max. We'll use numerical gradient for simplicity in this local step
        # Or just use L-BFGS with numerical jac if we define a function of w
        return val, None

    def fun_w(w):
        v, _ = calc_val_grad(w)
        return -v

    # L-BFGS-B on weights (simplex constraint handled by softmax or direct projection?)
    # Direct optimization on weights with simplex constraint is annoying.
    # Let's stick to the z-parameterization but restrict to support?
    # Actually, the current program already does global z optimization.
    # The "Combinatorial" insight is the *initialization* and *structure*.
    # So we just need to ensure the initial P is a good structural guess.
    
    # For now, return P as is, assuming the caller does global optimization.
    # But the directive asks for a DIFFERENT strategy.
    # Let's try a "Greedy Support Construction" first.
    
    return P

def _construct_structural_candidates():
    """
    Generate candidate P matrices based on geometric structures.
    Goal: Maximize H(x-y) while minimizing max(H(x), H(x+y), H(x+2y)).
    
    Structure 1: Vertical lines (x is constant)
    If x is constant, H(x)=0. H(x+y) = H(y). H(x+2y) = H(2y).
    H(x-y) = H(-y) = H(y).
    Ratio = H(y) / H(y) = 1. (Baseline)
    
    Structure 2: Diagonal lines (x-y is constant)
    If x-y=c, H(x-y)=0. Ratio 0. Bad.
    
    Structure 3: "Checkerboard" or specific sparse sets.
    Consider a set of points where x and y are uncorrelated, but x-y is spread out?
    No, x-y is a linear combination.
    
    Let's look at the entropy values for uniform distribution:
    x ~ U(0..9), H(x) = log2(10) ~ 3.32
    y ~ U(0..9), H(y) = log2(10) ~ 3.32
    x+y ~ Triangular(0..18), H(x+y) ~ 3.45
    x-y ~ Triangular(-9..9), H(x-y) ~ 3.45
    x+2y ~ Skewed, H(x+2y) ~ 3.6
    
    D = H(x-y) ~ 3.45
    M = max(3.32, 3.45, 3.6) = 3.6
    Ratio ~ 0.96
    
    To improve ratio, we want D to be high and M to be low.
    M is the max of H(x), H(x+y), H(x+2y).
    
    If we make x uniform, H(x) = 3.32.
    If we make x+y uniform, H(x+y) = 3.32.
    If we make x+2y uniform, H(x+2y) = 3.32.
    
    Can we have all three marginals be "low entropy" (concentrated) while x-y is "high entropy" (spread out)?
    x-y spread out means x and y are different.
    x concentrated means x is near a value.
    x+y concentrated means x+y is near a value.
    
    If x is concentrated at 0, and x+y is concentrated at 0, then y is concentrated at 0.
    Then x-y is concentrated at 0. Low D.
    
    We need x and y to vary such that:
    1. x is "simple" (low entropy)
    2. x+y is "simple" (low entropy)
    3. x+2y is "simple" (low entropy)
    4. x-y is "complex" (high entropy)
    
    This is a linear algebra problem on the support.
    Let's try a support that is a "line" in the (x,y) plane but not axis aligned.
    Line: x + a*y = c.
    Then x+y = c - (a-1)y.
    x+2y = c + (2-a)y.
    x-y = c - (a+1)y.
    
    If we pick a such that the coefficients of y in x, x+y, x+2y are small?
    Coeff of y in x is 0.
    Coeff of y in x+y is a-1.
    Coeff of y in x+2y is 2-a.
    
    If a=1, x=1. Vertical line.
    x=1, x+y=1+y, x+2y=1+2y.
    H(x)=0. H(x+y)=H(y). H(x+2y)=H(2y).
    D = H(1-y) = H(y).
    M = max(0, H(y), H(2y)).
    If y is uniform, H(y)=3.32, H(2y) is slightly higher or lower?
    Values of 2y: 0,2,4,6,8,10,12,14,16,18. All distinct. H(2y)=3.32.
    Ratio = 3.32 / 3.32 = 1.
    
    If a=0, y=0. Horizontal line.
    x= x, x+y=x, x+2y=x.
    H(x)=H(x)=H(x).
    D = H(x-0) = H(x).
    M = H(x).
    Ratio = 1.
    
    What if we use a "grid" of points?
    Suppose we pick points such that x takes few values, x+y takes few values, x+2y takes few values.
    But x-y takes many.
    
    Let's try to minimize M by making the distributions of x, x+y, x+2y be "peaked" (low entropy).
    This suggests using a small number of points.
    
    Candidate: The "Kakeya" set idea.
    A set of points with small measure but large projection?
    Here we are on a discrete grid.
    
    Let's try a specific pattern:
    Points where x+y is constant? No, that's a line.
    
    How about a "Staircase"?
    (0,0), (1,1), (2,2)... Diagonal. x-y=0. D=0. Bad.
    
    How about "Anti-diagonal"?
    (0,9), (1,8)... x+y=9.
    x varies 0..9. H(x)=3.32.
    x+y=9. H(x+y)=0.
    x+2y = x + 2(9-x) = 18-x. H(x+2y)=3.32.
    x-y = x - (9-x) = 2x-9. H(x-y)=3.32.
    M = max(3.32, 0, 3.32) = 3.32.
    D = 3.32.
    Ratio = 1.
    
    It seems hard to beat 1 with simple lines.
    The current score is 2.05. This implies a much better structure exists.
    Wait, the score is D/M. If D/M > 1, then D > M.
    D = H(x-y). M = max(H(x), H(x+y), H(x+2y)).
    So we need H(x-y) > max(H(x), H(x+y), H(x+2y)).
    
    This is possible if x-y is more "uniform" or "spread out" than x, x+y, or x+2y.
    
    Consider x and y independent uniforms.
    H(x)=3.32, H(y)=3.32.
    H(x-y) ~ 3.45.
    H(x+y) ~ 3.45.
    H(x+2y) ~ 3.6.
    M = 3.6. D = 3.45. Ratio < 1.
    
    We need to suppress H(x+2y) and H(x+y) while keeping H(x-y) high.
    This suggests a distribution where x+2y is concentrated.
    
    Let's try to make x+2y constant.
    x = c - 2y.
    Points: (c-2y, y).
    If y=0..4, x=c, c-2, c-4, c-6, c-8.
    Let c=8.
    y=0, x=8. (8,0)
    y=1, x=6. (6,1)
    y=2, x=4. (4,2)
    y=3, x=2. (2,3)
    y=4, x=0. (0,4)
    
    x+2y = 8. H(x+2y) = 0.
    x values: 8,6,4,2,0. H(x) = log2(5) ~ 2.32.
    x+y values: 8,7,6,5,4. H(x+y) = log2(5) ~ 2.32.
    x-y values: 8,5,2,-1,-4. H(x-y) = log2(5) ~ 2.32.
    M = max(2.32, 2.32, 0) = 2.32.
    D = 2.32.
    Ratio = 1.
    
    Still 1.
    
    What if we use a 2D grid?
    Points: (x,y) where x in {0,1}, y in {0,1,2,3,4,5,6,7,8,9}.
    x+2y = 2y or 1+2y.
    If x=0: 0,2,4...18.
    If x=1: 1,3,5...19.
    Combined: 0..19. All distinct?
    0,1,2,3...19. Yes.
    H(x+2y) = log2(20) ~ 4.32.
    H(x) = 1.
    x+y:
    x=0: 0..9.
    x=1: 1..10.
    Combined: 0..10.
    0,1,2...9 (2 counts), 10 (1 count).
    Probabilities: 0:0.05, 1:0.1, ..., 9:0.1, 10:0.05.
    H(x+y) = - (0.05 log 0.05 + 8*0.1 log 0.1 + 0.05 log 0.05)
    = - (2*0.05*(-4.32) + 8*0.1*(-3.32))
    = 0.432 + 2.656 = 3.088.
    
    x-y:
    x=0: 0,-1,-2...-9.
    x=1: 1,0,-1...-8.
    Combined:
    1: 1 count (0.05)
    0: 2 counts (0.1)
    -1: 2 counts (0.1)
    ...
    -8: 2 counts (0.1)
    -9: 1 count (0.05)
    Same distribution as x+y!
    H(x-y) = 3.088.
    
    M = max(1, 3.088, 4.32) = 4.32.
    D = 3.088.
    Ratio = 0.71. Worse.
    
    We need to find a structure where D is large and M is small.
    The current program finds 2.05.
    
    Let's look at the "Inspirations".
    Inspiration 1: Gradient based.
    Inspiration 2: Combinatorial.
    
    Maybe the optimal solution is not a simple line.
    
    Let's try a "Cross" shape.
    x=0 or y=0.
    (0,0) to (0,9) and (1,0) to (9,0).
    19 points.
    x values: 0 (10 times), 1..9 (1 time each).
    H(x) = - (10/19 log(10/19) + 9*(1/19 log(1/19)))
    = - (0.526*(-0.926) + 0.473*(-4.32))
    = 0.487 + 2.043 = 2.53.
    
    y values: Symmetric. H(y) = 2.53.
    
    x+y:
    (0,0)->0
    (0,1)->1 ... (0,9)->9
    (1,0)->1 ... (9,0)->9
    Counts:
    0: 1
    1: 2
    2: 2
    ...
    9: 2
    Total 19.
    H(x+y) = - (1/19 log(1/19) + 9*(2/19 log(2/19)))
    = - (0.0526*(-4.32) + 0.947*(-2.237))
    = 0.227 + 2.119 = 2.346.
    
    x+2y:
    (0,y) -> 2y. 0,2,4...18.
    (x,0) -> x. 1,2...9.
    Values:
    0: 1
    1: 1
    2: 2 (from 0,1 and 2,0)
    3: 1
    4: 2
    ...
    9: 2 (from 0,4.5? No. 0,? and 9,0. 2y=9 no. x=9. So 9 is 1?
    Let's list:
    y=0: x=0..9. x+2y = x. 0..9.
    y=1: x=0. x+2y = 2.
    y=2: x=0. x+2y = 4.
    y=3: x=0. x+2y = 6.
    y=4: x=0. x+2y = 8.
    y=5: x=0. x+2y = 10.
    y=6: x=0. x+2y = 12.
    y=7: x=0. x+2y = 14.
    y=8: x=0. x+2y = 16.
    y=9: x=0. x+2y = 18.
    
    Counts:
    0: 1
    1: 1
    2: 2 (2,0 and 0,1)
    3: 1
    4: 2
    5: 1
    6: 2
    7: 1
    8: 2
    9: 1
    10: 1
    12: 1
    14: 1
    16: 1
    18: 1
    
    Sum: 1+1+2+1+2+1+2+1+2+1+1+1+1+1+1 = 19. Correct.
    
    H(x+2y) = - sum p log p.
    p=1/19: 11 values.
    p=2/19: 4 values.
    H = - (11/19 log(11/19) + 4*(2/19 log(2/19)))
    = - (0.579*(-0.77) + 0.421*(-2.237))
    = 0.446 + 0.942 = 1.388.
    
    M = max(H(x)=2.53, H(x+y)=2.346, H(x+2y)=1.388) = 2.53.
    
    x-y:
    (x,0) -> x. 0..9.
    (0,y) -> -y. 0,-1...-9.
    Values:
    0: 2 (0,0 and 0,0? No. (0,0) is shared.
    Points: (0,0), (1,0)..(9,0), (0,1)..(0,9).
    x-y:
    (0,0) -> 0.
    (1,0) -> 1.
    ...
    (9,0) -> 9.
    (0,1) -> -1.
    ...
    (0,9) -> -9.
    
    Counts:
    0: 1
    1..9: 1 each.
    -1..-9: 1 each.
    Total 19.
    All distinct?
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9.
    Yes, 19 distinct values.
    H(x-y) = log2(19) ~ 4.248.
    
    Ratio = 4.248 / 2.53 = 1.68.
    
    This is better than 1, but current is 2.05.
    We can improve the "Cross" by adding more points to increase D while keeping M low.
    Or by weighting the points.
    
    The "Cross" has H(x) = 2.53.
    If we can reduce H(x) while keeping H(x-y) high, we win.
    H(x) is high because x takes 10 values.
    If we restrict x to fewer values, H(x) drops.
    But x-y needs to be spread out.
    
    Let's try a "Thick Cross" or specific weights.
    The current program uses gradient descent. It likely found a weighted version of such a structure.
    
    Strategy:
    1. Generate a few strong structural candidates (Cross, Thick Cross, etc.).
    2. Use L-BFGS to refine the weights of these candidates.
    
    This is "Combinatorial Initialization + Local Refinement".
    
    Let's define the candidates:
    1. Uniform Cross (x=0 or y=0).
    2. Double Cross (x in {0,4} or y in {0,4}).
    3. Grid (x in {0,4,9} or y in {0,4,9}).
    4. Random sparse sets (greedy selection).
    
    Then refine each.
    """
    candidates = []
    n = 10
    
    # 1. Cross
    P = np.zeros((n, n))
    P[:, 0] = 1.0
    P[0, :] = 1.0
    P[0, 0] = 1.0 # Avoid double count in logic, will normalize anyway
    candidates.append(P)
    
    # 2. Double Cross
    P = np.zeros((n, n))
    P[:, 0] = 1.0
    P[:, 4] = 1.0
    P[0, :] = 1.0
    P[4, :] = 1.0
    candidates.append(P)
    
    # 3. Grid
    P = np.zeros((n, n))
    for i in [0, 4, 9]:
        P[:, i] = 1.0
        P[i, :] = 1.0
    candidates.append(P)
    
    # 4. Random Sparse (Greedy-ish)
    rng = np.random.default_rng(42)
    for _ in range(5):
        P = np.zeros((n, n))
        # Pick 20 random points
        pts = rng.choice(n*n, 20, replace=False)
        P.ravel()[pts] = 1.0
        candidates.append(P)
        
    # 5. Anti-Cross (x+y=0..9 and x-y=0..9? No, just lines)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1.0
        P[i, n-1-i] = 1.0
    candidates.append(P)
    
    return candidates

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_val = -1e9
    best_P = np.full((n, n), 1.0 / 100.0)
    
    candidates = _construct_structural_candidates()
    
    # Add uniform as a fallback
    candidates.append(np.ones((n, n)))
    
    # Refine each candidate
    for P_init in candidates:
        if time.time() > deadline - 60.0:
            break
            
        P_init = P_init.astype(np.float64)
        P_init = P_init / P_init.sum()
        
        # Convert to z for optimization
        # z = log(P + eps)
        eps = 1e-12
        z = np.log(P_init + eps).ravel()
        
        # Define objective
        def fun(z):
            P = np.exp(z - np.max(z))
            s = P.sum()
            if s < 1e-300:
                return 1e9
            P = P / s
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return -v
        
        # L-BFGS-B
        try:
            res = minimize(fun, z, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            z_opt = res.x
            P_opt = np.exp(z_opt - np.max(z_opt))
            P_opt = P_opt / P_opt.sum()
            P_opt = P_opt.reshape(n, n)
            
            v_opt = _value_from_P(P_opt, idx_a, idx_b, idx_c, idx_d)
            
            if v_opt > best_val:
                best_val = v_opt
                best_P = P_opt
                
        except Exception:
            pass
            
    # Final Polish
    if time.time() < deadline - 10.0:
        z_best = np.log(best_P.ravel() + 1e-12)
        def fun(z):
            P = np.exp(z - np.max(z))
            s = P.sum()
            if s < 1e-300:
                return 1e9
            P = P / s
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return -v
        try:
            res = minimize(fun, z_best, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            z_opt = res.x
            P_opt = np.exp(z_opt - np.max(z_opt))
            P_opt = P_opt / P_opt.sum()
            P_opt = P_opt.reshape(n, n)
            v_opt = _value_from_P(P_opt, idx_a, idx_b, idx_c, idx_d)
            if v_opt > best_val:
                best_val = v_opt
                best_P = P_opt
        except Exception:
            pass

    # Finalize
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
    best_P = best_P / s
    
    # Check validity
    pf = best_P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
        
    return best_P.astype(np.float64)

# EVOLVE-BLOCK-END
