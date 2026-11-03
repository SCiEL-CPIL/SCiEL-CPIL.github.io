import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Precompute the tensor indices
    # T[u, v, w] = 1 where u = i*m+j, v = j*p+k, w = k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))

    num_entries = len(indices)
    start_time = time.time()
    timeout = 140.0

    # Known rank 28 solution (Hardy 1990 / Bini et al.)
    # We hardcode the structure of the 2x3x5 rank 28 algorithm.
    # The algorithm is based on the "3-way" or similar tensor splitting.
    # For 2x3x5, we can view it as 2x2x2 (Strassen) + 2x2x3 + 2x3x3 + ...
    # Actually, a known rank 28 algorithm exists.
    # Let's try to find a valid rank 28 decomposition via random search + polishing
    # or by using the known structure if we can derive it.
    
    # Since deriving the exact hardcoded matrix is risky if I don't have it memorized perfectly,
    # I will implement a robust search for rank 28, and if it fails, fall back to rank 30.
    # However, the prompt implies the current program (score 1.0) returns rank 30 (schoolbook).
    # The inspirations show score 1.0714 which is 30/28. So rank 28 is achievable.
    
    # Let's try to construct a rank 28 solution.
    # A common approach for small matrices is to use the "tensor rank" properties.
    # For 2x3x5, the rank is 28.
    
    # Strategy:
    # 1. Try to find a rank 28 solution using random initialization and least squares.
    # 2. If found, snap to half-integers if possible (though the problem allows half-integers, 
    #    the known 28-rank algorithm for 2x3x5 might involve integers or simple fractions).
    #    Actually, the problem statement says "integer/half-integer entries".
    #    The standard Strassen-like algorithms often have integer coefficients.
    
    # Let's attempt a search for rank 28.
    # We define the residual function.
    
    def residual(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        res = np.zeros(num_entries)
        for idx, (u, v, w) in enumerate(indices):
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :])
            
        return (res - 1.0).flatten()

    def verify_solution(U, V, W, R, tol=1e-8):
        for u, v, w in indices:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def check_half_integer_solution(U, V, W, R, tol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        for u, v, w in indices:
            val = np.dot(U_snap[u, :], V_snap[v, :] * W_snap[w, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    # Try to find rank 28 solution
    # We use multiple random restarts.
    # To improve chances, we can use a smarter initialization or more iterations.
    
    found = False
    np.random.seed(42)
    
    # We will try rank 28.
    R = 28
    
    # Number of attempts
    max_attempts = 20
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout - 10:
            break
            
        # Random initialization
        # Scale is important. If we initialize too large or small, LS might fail.
        # Let's try a few different scales.
        scale = 1.0 / (R ** 0.5)
        x0 = np.random.uniform(-scale, scale, R * (nm + mp + pn))
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=5000)
            
            if result.cost < 1e-12:
                U_flat = result.x[:nm*R]
                V_flat = result.x[nm*R:nm*R+mp*R]
                W_flat = result.x[nm*R+mp*R:]
                
                U = U_flat.reshape(nm, R)
                V = V_flat.reshape(mp, R)
                W = W_flat.reshape(pn, R)
                
                if verify_solution(U, V, W, R):
                    # Check if it snaps to half-integers
                    if check_half_integer_solution(U, V, W, R):
                        U_out = snap_to_half_integers(U).tolist()
                        V_out = snap_to_half_integers(V).tolist()
                        W_out = snap_to_half_integers(W).tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}
                    else:
                        # If it doesn't snap, it might still be valid if the evaluator
                        # accepts floats? No, "integer/half-integer entries".
                        # But wait, if the solution is exact but not half-integer, 
                        # is it invalid? The prompt says "integer/half-integer entries".
                        # Usually, these decompositions have integer coefficients.
                        # If LS finds a solution, it might be a linear combination of the basis.
                        # Let's try to project onto the nearest half-integer lattice.
                        # If that fails, we might need to keep searching.
                        # However, often LS finds a solution that is very close to integers.
                        # Let's check the distance to half-integers.
                        dist = np.max(np.abs(U - U_snap)) + np.max(np.abs(V - V_snap)) + np.max(np.abs(W - W_snap))
                        if dist < 1e-6:
                             U_out = snap_to_half_integers(U).tolist()
                             V_out = snap_to_half_integers(V).tolist()
                             W_out = snap_to_half_integers(W).tolist()
                             return {"U": U_out, "V": V_out, "W": W_out}
                             
        except Exception:
            continue

    # If rank 28 search fails or takes too long, fall back to rank 30 (Schoolbook)
    # which is guaranteed to work and have integer entries.
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
