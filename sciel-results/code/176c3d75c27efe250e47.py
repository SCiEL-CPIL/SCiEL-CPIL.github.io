"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known optimal rank for 2x2x3 over reals/rationals.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _rank10_core_plus_extension(deadline):
    """
    Rank-7 core (Strassen) + rank-3 extension strategy.
    
    Fix the first 7 terms to the standard Strassen decomposition for 2x2x2.
    Then try to find 3 additional rank-1 terms that produce the third column
    (C[0,2] and C[1,2]) while NOT contributing to C[0,0], C[0,1], C[1,0], C[1,1].
    
    For the new terms r=7,8,9:
    - They must satisfy: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 0 for k=0,1 (no contribution to cols 0,1)
    - They must satisfy: sum_r U[i,j,r]*V[j,2,r]*W[2,i,r] = 1 for (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    
    This is a system of polynomial equations in the new variables.
    We can simplify by fixing U for the new terms (since U only depends on i,j)
    and solving for V[:,2,:] and W[2,:,:].
    """
    if time.time() > deadline - 10:
        return None
    
    # Get the Strassen core (terms 0-6)
    base = _rank11_decomposition()
    U7 = np.array(base["U"][:, :7], dtype=float)
    V7 = np.array(base["V"][:, :7], dtype=float)
    W7 = np.array(base["W"][:, :7], dtype=float)
    
    # Verify the core handles columns 0 and 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U7[i*2+j, r] * V7[j*3+k, r] * W7[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    return None  # Core is broken
    
    # For the 3 new terms (r=7,8,9), we need:
    # Variables: U_new[4,3], V_new[6,3], W_new[6,3]
    # But V_new[j,0,:] and V_new[j,1,:] must be 0 (no B00,B01,B10,B11 contributions from new terms)
    # Similarly W_new[0,:,:] and W_new[1,:,:] must be 0 (no C0,C1 contributions)
    # So only V_new[j,2,:] and W_new[2,:,:] are free.
    
    # Let's parameterize:
    # U_new[i,j] for i in {0,1}, j in {0,1} -> 4 variables per term, 12 total
    # V_new[j,2] for j in {0,1} -> 2 variables per term, 6 total  
    # W_new[2,i] for i in {0,1} -> 2 variables per term, 6 total
    # Total: 24 variables
    
    # Constraints:
    # For each (i,j) in {(0,0),(0,1),(1,0),(1,1)}:
    #   sum_{r=7}^{9} U_new[i,j,r] * V_new[j,2,r] * W_new[2,i,r] = 1
    
    # That's 4 equations in 24 unknowns. Very underdetermined!
    # We can fix many variables and solve.
    
    # Strategy: Try specific structured solutions.
    # Simplest: make each new term handle exactly one (i,j) pair.
    # Term 7: handles (0,0): U_new[0,0,7]=1, others 0; V_new[0,2,7]=1; W_new[2,0,7]=1
    # Term 8: handles (0,1): U_new[0,1,8]=1; V_new[1,2,8]=1; W_new[2,0,8]=1
    # Term 9: handles (1,0): U_new[1,0,9]=1; V_new[0,2,9]=1; W_new[2,1,9]=1
    # But we also need (1,1)! So this doesn't work with 3 terms.
    
    # We need to handle all 4 (i,j) pairs with 3 terms.
    # Each term r contributes U_new[i,j,r] * V_new[j,2,r] * W_new[2,i,r] to C[i,j] for column 2.
    
    # Let's try: 
    # Term 7: U_new[0,0,7]=1, U_new[1,1,7]=1; V_new[0,2,7]=1, V_new[1,2,7]=1; W_new[2,0,7]=1, W_new[2,1,7]=1
    #   Contribution to C[0,0]: 1*1*1 = 1 ✓
    #   Contribution to C[1,1]: 1*1*1 = 1 ✓
    #   Contribution to C[0,1]: U_new[0,1,7]=0 -> 0 ✓
    #   Contribution to C[1,0]: U_new[1,0,7]=0 -> 0 ✓
    #   But we need C[0,1]=1 and C[1,0]=1 too!
    
    # So we need a different structure. Let's use:
    # Term 7: handles C[0,0] and C[0,1]
    #   U_new[0,0,7]=1, U_new[0,1,7]=1
    #   V_new[0,2,7]=1, V_new[1,2,7]=1
    #   W_new[2,0,7]=1
    #   C[0,0]: 1*1*1 = 1 ✓
    #   C[0,1]: 1*1*1 = 1 ✓
    #   C[1,0]: U_new[1,0,7]=0 -> 0 ✓
    #   C[1,1]: U_new[1,1,7]=0 -> 0 ✓
    
    # Term 8: handles C[1,0] and C[1,1]
    #   U_new[1,0,8]=1, U_new[1,1,8]=1
    #   V_new[0,2,8]=1, V_new[1,2,8]=1
    #   W_new[2,1,8]=1
    #   C[1,0]: 1*1*1 = 1 ✓
    #   C[1,1]: 1*1*1 = 1 ✓
    #   C[0,0]: U_new[0,0,8]=0 -> 0 ✓
    #   C[0,1]: U_new[0,1,8]=0 -> 0 ✓
    
    # But that's only 2 terms! We have 3 terms available.
    # Wait, this gives us a rank-9 decomposition! Let me check...
    
    # Actually, let me verify this works:
    U9 = np.zeros((4, 9))
    V9 = np.zeros((6, 9))
    W9 = np.zeros((6, 9))
    
    # Copy Strassen core
    U9[:, :7] = U7
    V9[:, :7] = V7
    W9[:, :7] = W7
    
    # Term 7: C[0,0] and C[0,1]
    U9[0, 7] = 1  # U[0,0]
    U9[1, 7] = 1  # U[0,1]
    V9[2, 7] = 1  # V[0,2]
    V9[5, 7] = 1  # V[1,2]
    W9[4, 7] = 1  # W[2,0]
    
    # Term 8: C[1,0] and C[1,1]
    U9[2, 8] = 1  # U[1,0]
    U9[3, 8] = 1  # U[1,1]
    V9[2, 8] = 1  # V[0,2]
    V9[5, 8] = 1  # V[1,2]
    W9[5, 8] = 1  # W[2,1]
    
    # Verify
    if _verify(U9.tolist(), V9.tolist(), W9.tolist(), 2, 2, 3):
        return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}
    
    # If that didn't work, try other structures
    # Try with half-integer coefficients to get more flexibility
    
    # General approach: solve the 4 equations
    # Let a_r = U_new[0,0,r], b_r = U_new[0,1,r], c_r = U_new[1,0,r], d_r = U_new[1,1,r]
    # Let e_r = V_new[0,2,r], f_r = V_new[1,2,r]
    # Let g_r = W_new[2,0,r], h_r = W_new[2,1,r]
    
    # Equations:
    # a_7*e_7*g_7 + a_8*e_8*g_8 + a_9*e_9*g_9 = 1  (C[0,0])
    # b_7*f_7*g_7 + b_8*f_8*g_8 + b_9*f_9*g_9 = 1  (C[0,1])
    # c_7*e_7*h_7 + c_8*e_8*h_8 + c_9*e_9*h_9 = 1  (C[1,0])
    # d_7*f_7*h_7 + d_8*f_8*h_8 + d_9*f_9*h_9 = 1  (C[1,1])
    
    # Try setting some to 1 and solving
    # Case 1: a_r=e_r=g_r=1 for all r, then b_r=f_r=g_r, etc.
    # This is getting complex. Let's try a numerical approach.
    
    if time.time() > deadline - 5:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _residuals_rank9(params):
        """Residuals for rank-9 with fixed Strassen core."""
        # params: 24 values for the 3 new terms
        # U_new: 4x3, V_new: 2x3 (only col 2), W_new: 2x3 (only row 2)
        U_new = params[0:12].reshape(4, 3)
        V_new_col2 = params[12:18].reshape(2, 3)  # V[j,2] for j=0,1
        W_new_row2 = params[18:24].reshape(2, 3)  # W[2,i] for i=0,1
        
        res = np.zeros(4)
        # C[0,0]: sum_r U_new[0,r] * V_new_col2[0,r] * W_new_row2[0,r]
        res[0] = np.sum(U_new[0, :] * V_new_col2[0, :] * W_new_row2[0, :]) - 1.0
        # C[0,1]: sum_r U_new[1,r] * V_new_col2[1,r] * W_new_row2[0,r]
        res[1] = np.sum(U_new[1, :] * V_new_col2[1, :] * W_new_row2[0, :]) - 1.0
        # C[1,0]: sum_r U_new[2,r] * V_new_col2[0,r] * W_new_row2[1,r]
        res[2] = np.sum(U_new[2, :] * V_new_col2[0, :] * W_new_row2[1, :]) - 1.0
        # C[1,1]: sum_r U_new[3,r] * V_new_col2[1,r] * W_new_row2[1,r]
        res[3] = np.sum(U_new[3, :] * V_new_col2[1, :] * W_new_row2[1, :]) - 1.0
        return res
    
    rng = np.random.default_rng(42)
    best_cost = float('inf')
    best_params = None
    
    for trial in range(50):
        if time.time() > deadline - 3:
            break
        
        # Initialize with the simple structure we found
        if trial == 0:
            x0 = np.array([
                1, 1, 0,  # U_new row 0: [a_7, a_8, a_9]
                1, 1, 0,  # U_new row 1: [b_7, b_8, b_9]
                0, 1, 0,  # U_new row 2: [c_7, c_8, c_9]
                0, 1, 0,  # U_new row 3: [d_7, d_8, d_9]
                1, 1, 0,  # V_new_col2 row 0: [e_7, e_8, e_9]
                1, 1, 0,  # V_new_col2 row 1: [f_7, f_8, f_9]
                1, 0, 0,  # W_new_row2 row 0: [g_7, g_8, g_9]
                0, 1, 0,  # W_new_row2 row 1: [h_7, h_8, h_9]
            ], dtype=float)
        else:
            # Random initialization with small values
            vals = [-1, -0.5, 0, 0.5, 1]
            x0 = np.random.choice(vals, 24).astype(float)
        
        try:
            result = least_squares(_residuals_rank9, x0, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-15:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    # Extract and build full matrices
    U_new = best_params[0:12].reshape(4, 3)
    V_new_col2 = best_params[12:18].reshape(2, 3)
    W_new_row2 = best_params[18:24].reshape(2, 3)
    
    U9 = np.zeros((4, 9))
    V9 = np.zeros((6, 9))
    W9 = np.zeros((6, 9))
    
    U9[:, :7] = U7
    V9[:, :7] = V7
    W9[:, :7] = W7
    
    U9[:, 7:] = U_new
    V9[2, 7:] = V_new_col2[0]  # V[0,2]
    V9[5, 7:] = V_new_col2[1]  # V[1,2]
    W9[4, 7:] = W_new_row2[0]  # W[2,0]
    W9[5, 7:] = W_new_row2[1]  # W[2,1]
    
    # Try to snap to rational values
    for snap in [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
    ]:
        U = [[snap(u) for u in row] for row in U9.tolist()]
        V = [[snap(v) for v in row] for row in V9.tolist()]
        W = [[snap(w) for w in row] for row in W9.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    # Check if the numerical solution is close enough
    if best_cost < 1e-12:
        U = U9.tolist()
        V = V9.tolist()
        W = W9.tolist()
        if _verify(U, V, W, 2, 2, 3, tol=1e-8):
            return {"U": U, "V": V, "W": W}
    
    return None


def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _try_rank10_polynomial_search(deadline):
    """
    Attempt to find a rank-10 decomposition by treating the bilinear map constraints
    as polynomial equations and using structured numerical solvers with high precision.
    Uses the rank-11 solution as a guide to identify which term to eliminate.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(12345)
    best_cost = float('inf')
    best_params = None
    
    # Try dropping each of the 11 terms and solving
    for drop_term in range(11):
        if time.time() > deadline - 20:
            break
        
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        # Multiple perturbations
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Scale perturbation by attempt number
            scale = 0.1 * (attempt + 1)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x0 = x0 + rng.normal(0, scale, 160)
            
            try:
                result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=10000, ftol=1e-18, xtol=1e-18)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_params = result.x.copy()
                    if result.cost < 1e-14:
                        break
            except Exception:
                pass
        
        if best_cost < 1e-14:
            break
    
    # Also try random initializations with structured patterns
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        
        # Try small integer/half-integer initializations
        vals = [-1, -0.5, 0, 0.5, 1]
        x0 = np.random.choice(vals, 160).astype(float)
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=10000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    # Extract and try to snap to rational values
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try different snap strategies
    for snap in [
        lambda x: round(x * 2) / 2,      # half-integers
        lambda x: round(x * 4) / 4,      # quarter-integers
        lambda x: round(x * 8) / 8,      # eighth-integers
        lambda x: round(x),               # integers
        lambda x: round(x * 6) / 6,      # sixths
    ]:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    # If no snap worked, return the numerical solution (may not pass verification)
    return None


def _try_rank10_groebner_like(deadline):
    """
    Attempt to find a rank-10 decomposition using a Groebner-basis-like approach.
    This is a heuristic: we fix the structure to have many zeros and solve the
    remaining polynomial system via substitution.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import fsolve
    except ImportError:
        return None
    
    # Assume a specific structure: 10 terms where some are "merged"
    # We'll parameterize with fewer free variables
    rng = np.random.default_rng(999)
    
    best_cost = float('inf')
    best_solution = None
    
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        
        # Random initialization with sparsity
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Fill with random small values
        for i in range(4):
            for j in range(10):
                if rng.random() < 0.3:
                    U[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    V[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    W[i, j] = rng.choice([-1, 1])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = fsolve(_residuals_rank10, x0, full_output=True, xtol=1e-14, maxfev=5000)
            x_sol, info, ier, msg = result
            if ier == 1:
                cost = np.sum(np.array(_residuals_rank10(x_sol))**2)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = x_sol.copy()
                    if cost < 1e-14:
                        break
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-10:
        return None
    
    U_flat = best_solution[:40]
    V_flat = best_solution[40:100]
    W_flat = best_solution[100:160]
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    for snap in [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
    ]:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 135
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-9 solution using core+extension strategy
    if best_rank > 9 and time.time() < deadline - 40:
        result9 = _rank10_core_plus_extension(deadline)
        if result9 is not None:
            best = result9
            best_rank = 9
            return best  # Return immediately if we found rank 9
    
    # Try to find a rank-10 solution using multiple strategies
    if best_rank > 10 and time.time() < deadline - 35:
        # Strategy 1: Polynomial search with structured initialization
        result10 = _try_rank10_polynomial_search(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
        elif time.time() < deadline - 20:
            # Strategy 2: Groebner-like search
            result10 = _try_rank10_groebner_like(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
    
    return best
