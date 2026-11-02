import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Bivariate polynomial multiplication: rank-9 construction
    # P(x,y) = sum_{i=0}^{1} sum_{k=0}^{1} A[i,k] * x^i * y^k  (degree 1 in x, 1 in y)
    # Q(x,y) = sum_{i=0}^{1} sum_{j=0}^{4} B[i,j] * x^i * y^j  (degree 1 in x, 4 in y)
    # Product R(x,y) = P*Q has degree (2,5) with 18 coefficients
    # Evaluate at 9 points (3x3 grid) -> rank 9
    
    # Choose evaluation points
    x_pts = [0, 1, 2]
    y_pts = [0, 1, 2]
    
    eval_points = []
    for xv in x_pts:
        for yv in y_pts:
            eval_points.append((xv, yv))
    
    n_eval = len(eval_points)  # 9
    
    # Build U: shape (4, 9) - U[i,k, r] = x_r^i * y_r^k
    U = np.zeros((4, n_eval))
    for r, (xv, yv) in enumerate(eval_points):
        for i in range(2):
            for k in range(2):
                idx = i * 2 + k
                U[idx, r] = (xv ** i) * (yv ** k)
    
    # Build V: shape (18, 9) - V[i,j, r] = x_r^i * y_r^j
    V = np.zeros((18, n_eval))
    for r, (xv, yv) in enumerate(eval_points):
        for i in range(2):
            for j in range(5):
                idx = i * 9 + j
                V[idx, r] = (xv ** i) * (yv ** j)
    
    # Build W: shape (18, 9) - W[i,j, r] = Lagrange weight for coefficient (i,j) at eval point r
    # We need: R[i,j] = sum_r W[i,j,r] * U[:,r] * V[:,r]
    # where R[i,j] is the (i,j)-th coefficient of the product
    # Using Lagrange interpolation: R[i,j] = sum_r L_r[i,j] * R(x_r, y_r)
    # where L_r[i,j] is the Lagrange basis function evaluated at the coefficient index
    
    # For bivariate Lagrange interpolation with 3x3 grid:
    # L_r(x,y) = prod_{s != r} (x - x_s)/(x_r - x_s) * prod_{t != r} (y - y_t)/(y_r - y_t)
    # But we need to extract specific coefficients, so we use:
    # R[i,j] = sum_r c_{r,i,j} * R(x_r, y_r)
    
    # The coefficient extraction can be done via:
    # R[i,j] = (1/(2*pi*i)) * integral... but for discrete points:
    # Use the fact that if R(x,y) = sum_{a=0}^{2} sum_{b=0}^{5} c_{a,b} x^a y^b
    # then c_{a,b} can be extracted from values at 9 points if we use appropriate weights
    
    # Actually, for a 3x3 grid, we can only uniquely determine a polynomial of degree (2,2) 
    # from 9 points. But our product has degree (2,5), so 9 points are NOT sufficient!
    
    # We need more points. Let's use a different approach:
    # Split into x and y separately.
    # For each fixed y, we have a univariate polynomial in x of degree 2.
    # We need 3 points in x to determine it.
    # For each coefficient in y (0 to 5), we need 3 points in x.
    # Total: 6 * 3 = 18 points? No, that's too many.
    
    # Better approach: Use tensor product structure.
    # P(x,y)Q(x,y) = [P_0(y) + x*P_1(y)] * [Q_0(y) + x*Q_1(y)]
    # = P_0(y)Q_0(y) + x*(P_0(y)Q_1(y) + P_1(y)Q_0(y)) + x^2*P_1(y)Q_1(y)
    
    # So we need to multiply polynomials in y:
    # P_0(y)Q_0(y): deg (1)*(1) = deg 2, needs 3 points
    # P_0(y)Q_1(y): deg (1)*(4) = deg 5, needs 6 points
    # P_1(y)Q_0(y): deg (1)*(1) = deg 2, needs 3 points
    # P_1(y)Q_1(y): deg (1)*(4) = deg 5, needs 6 points
    
    # But we can share evaluation points! Use 6 points in y.
    # At each y-point, compute 3 values (for x=0,1,2) to get the x-coefficients.
    # Total rank = 6 * 3 = 18. Still not 9.
    
    # For rank 9, we need a different strategy.
    # Let's try: evaluate P and Q at 9 points in 2D, but use the fact that
    # the output has special structure.
    
    # Alternative: Use the existing rank-32 fallback but try to improve it.
    # Or accept that rank 9 may not be achievable with simple evaluation.
    
    # Let's just use the fallback rank-32 construction for now, but with
    # a quick search for better rank first.
    
    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)
    
    def build_bivariate_eval(x_pts, y_pts):
        """Build U, V, W for bivariate polynomial multiplication via evaluation."""
        x_vals = np.array(x_pts, dtype=float)
        y_vals = np.array(y_pts, dtype=float)
        n_x = len(x_vals)
        n_y = len(y_vals)
        n_eval = n_x * n_y
        
        # U[i,k,r] where r = ix*n_y + iy, i in [0,1], k in [0,1]
        U = np.zeros((4, n_eval))
        # V[i,j,r] where i in [0,1], j in [0,4]
        V = np.zeros((18, n_eval))
        # W[i,j,r] where i in [0,2], j in [0,5] for output coefficients
        W = np.zeros((18, n_eval))
        
        for ix, xv in enumerate(x_vals):
            for iy, yv in enumerate(y_vals):
                r = ix * n_y + iy
                for i in range(2):
                    for k in range(2):
                        U[i*2+k, r] = (xv ** i) * (yv ** k)
                for i in range(2):
                    for j in range(5):
                        V[i*9+j, r] = (xv ** i) * (yv ** j)
        
        # For W, we need Lagrange interpolation weights
        # R(a,b) = sum_{ix,iy} W[a,b,ix,iy] * R(x_ix, y_iy)
        # where R(a,b) is coefficient of x^a y^b
        
        # Use 2D Lagrange interpolation
        for a in range(3):  # x-degree 0,1,2
            for b in range(6):  # y-degree 0,1,2,3,4,5
                for ix, xv in enumerate(x_vals):
                    for iy, yv in enumerate(y_vals):
                        r = ix * n_y + iy
                        # Compute Lagrange weight for coefficient (a,b)
                        # This requires solving a linear system or using known formulas
                        # For now, leave as 0 - we'll compute properly below
                        W[a*6+b, r] = 0.0  # placeholder
        
        return U, V, W
    
    # Actually, let's compute W properly using linear algebra
    def compute_lagrange_weights(x_pts, y_pts):
        """Compute Lagrange interpolation weights for coefficient extraction."""
        x_vals = np.array(x_pts, dtype=float)
        y_vals = np.array(y_pts, dtype=float)
        n_x = len(x_vals)
        n_y = len(y_vals)
        n_eval = n_x * n_y
        
        # Build the evaluation matrix E where E[r, (a,b)] = x_r^a * y_r^b
        # Then W = E^{-1} (if square) or least squares
        E = np.zeros((n_eval, 18))
        for ix, xv in enumerate(x_vals):
            for iy, yv in enumerate(y_vals):
                r = ix * n_y + iy
                for a in range(3):
                    for b in range(6):
                        E[r, a*6+b] = (xv ** a) * (yv ** b)
        
        # We need W such that coeff = W @ eval_values
        # i.e., W = E^{-1} if E is square (18x18)
        # But we have n_eval x 18, so if n_eval >= 18, use pseudoinverse
        # If n_eval = 18 and E is invertible, W = E^{-1}
        
        if n_eval == 18:
            try:
                W = np.linalg.inv(E)
                return W
            except:
                pass
        
        # Use pseudoinverse
        W = np.linalg.pinv(E)
        return W  # shape (18, n_eval)

    def build_from_block(U_block, V_block, W_block):
        R_block = U_block.shape[1]
        R_total = 3 * R_block
        U_full = np.zeros((4, R_total))
        V_full = np.zeros((18, R_total))
        W_full = np.zeros((18, R_total))
        
        for b in range(3):
            start_r = b * R_block
            end_r = start_r + R_block
            U_full[:, start_r:end_r] = U_block
            for k in range(2):
                for j_local in range(3):
                    j_global = 3*b + j_local
                    v_idx_block = k*3 + j_local
                    v_idx_full = k*9 + j_global
                    V_full[v_idx_full, start_r:end_r] = V_block[v_idx_block, :]
            for j_local in range(3):
                j_global = 3*b + j_local
                for i in range(2):
                    w_idx_block = j_local*2 + i
                    w_idx_full = j_global*2 + i
                    W_full[w_idx_full, start_r:end_r] = W_block[w_idx_block, :]
        return U_full, V_full, W_full
    
    # Try bivariate evaluation with 18 points (3x6 grid)
    if time.time() < deadline:
        x_pts = [0, 1, 2]
        y_pts = [0, 1, 2, 3, 4, 5]
        n_eval = 18
        W_mat = compute_lagrange_weights(x_pts, y_pts)
        
        U = np.zeros((4, n_eval))
        V = np.zeros((18, n_eval))
        for ix, xv in enumerate(x_pts):
            for iy, yv in enumerate(y_pts):
                r = ix * len(y_pts) + iy
                for i in range(2):
                    for k in range(2):
                        U[i*2+k, r] = (xv ** i) * (yv ** k)
                for i in range(2):
                    for j in range(5):
                        V[i*9+j, r] = (xv ** i) * (yv ** j)
        
        W = W_mat.T  # shape (18, 18)
        
        # Snap to integers/half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
        # Try without snapping
        if verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def search_rank_223(R, max_restarts=20):
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+6*R].reshape(6, R)
            W = params[4*R+6*R:4*R+12*R].reshape(6, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, 3, R)
            W_r = W.reshape(3, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return T.flatten() - 1.0
        
        best_sol = None
        best_cost = float('inf')
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            scale = 0.5 + attempt * 0.1
            x0 = np.random.randn(16*R) * scale
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if res.cost < 1e-10:
                        break
            except:
                continue
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+6*R].reshape(6, R)
            W = best_sol[4*R+6*R:4*R+12*R].reshape(6, R)
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                return U_snapped, V_snapped, W_snapped
            U_snapped = np.round(U)
            V_snapped = np.round(V)
            W_snapped = np.round(W)
            if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                return U_snapped, V_snapped, W_snapped
        return None

    # Try to find a low rank decomposition for [2,2,3]
    block_decomp = None
    for R in [9, 10]:
        if time.time() > deadline - 30:
            break
        sol = search_rank_223(R)
        if sol is not None:
            U_b, V_b, W_b = sol
            block_decomp = (U_b, V_b, W_b, R)
            break

    if block_decomp is not None:
        U_b, V_b, W_b, R_b = block_decomp
        U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # First, let's try to find a rank-223 decomposition with more aggressive search
    if time.time() < deadline - 30:
        # Try rank 8 for 2x2x3 (total 24)
        def search_rank_223_aggressive(R, max_restarts=50):
            def residuals(params):
                U = params[:4*R].reshape(4, R)
                V = params[4*R:4*R+6*R].reshape(6, R)
                W = params[4*R+6*R:4*R+12*R].reshape(6, R)
                U_r = U.reshape(2, 2, R)
                V_r = V.reshape(2, 3, R)
                W_r = W.reshape(3, 2, R)
                T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
                return T.flatten() - 1.0
            
            best_sol = None
            best_cost = float('inf')
            
            # Use structured initializations based on Strassen-like patterns
            for attempt in range(max_restarts):
                if time.time() > deadline:
                    break
                
                if attempt < 10:
                    # Random initialization
                    scale = 0.3 + attempt * 0.05
                    x0 = np.random.randn(16*R) * scale
                else:
                    # Try to initialize with known structure
                    x0 = np.zeros(16*R)
                    # Initialize with simple patterns
                    for r in range(R):
                        x0[r] = 1.0 if r % 3 == 0 else (-1.0 if r % 3 == 1 else 0.5)
                        x0[4*R + r] = 1.0 if r % 2 == 0 else -0.5
                        x0[4*R + 6*R + r] = 0.5 if r % 4 == 0 else (-0.5 if r % 4 == 1 else 1.0 if r % 4 == 2 else -1.0)
                
                try:
                    res = least_squares(residuals, x0, method='trf', max_nfev=8000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_sol = res.x
                        if res.cost < 1e-12:
                            break
                except:
                    continue
            
            if best_sol is not None and best_cost < 1e-10:
                U = best_sol[:4*R].reshape(4, R)
                V = best_sol[4*R:4*R+6*R].reshape(6, R)
                W = best_sol[4*R+6*R:4*R+12*R].reshape(6, R)
                
                # Try multiple snapping strategies
                for snap_factor in [2, 4, 8, 16]:
                    U_snapped = np.round(U * snap_factor) / snap_factor
                    V_snapped = np.round(V * snap_factor) / snap_factor
                    W_snapped = np.round(W * snap_factor) / snap_factor
                    if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                        return U_snapped, V_snapped, W_snapped
                
                U_snapped = np.round(U)
                V_snapped = np.round(V)
                W_snapped = np.round(W)
                if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                    return U_snapped, V_snapped, W_snapped
            return None
        
        # Try rank 8 first (would give total rank 24)
        for R in [8, 9, 10]:
            if time.time() > deadline - 10:
                break
            sol = search_rank_223_aggressive(R)
            if sol is not None:
                U_b, V_b, W_b = sol
                U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
                if verify_decomp(U_full, V_full, W_full, n, m, p):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Rank 32 construction
    terms = []
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
        terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
        terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
        terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
        terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
    
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
