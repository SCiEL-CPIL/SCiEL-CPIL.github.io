import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _residual(U, V, W, n, m, p):
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)

def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U

def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V

def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W

def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen-like terms for k=0,1 (rank 7)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # k=2 terms (4 more)
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _try_slice_absorption(deadline):
    """
    Slice absorption: Use 7 Strassen terms for k=0,1 and find 3 additional terms
    that handle k=2 by expressing the needed bilinear forms as linear combinations
    of the Strassen forms plus 3 new rank-1 terms.
    
    The k=2 output needs:
      C[2,0] = A[0,0]*B[0,2] + A[1,0]*B[1,2]
      C[2,1] = A[0,1]*B[0,2] + A[1,1]*B[1,2]
    
    We have 7 Strassen bilinear forms (products of linear forms in A and B).
    We want to find 3 more rank-1 terms such that the k=2 outputs can be
    expressed as linear combinations of all 10 bilinear forms.
    
    Each rank-1 term c has factors U_c (4-vector), V_c (6-vector), W_c (6-vector).
    The bilinear form for term c is: (U_c^T * A_flat) * (V_c^T * B_flat)
    where A_flat = [A00, A01, A10, A11] and B_flat = [B00, B01, B02, B10, B11, B12].
    
    For the k=2 outputs, we need:
      C[2,0] = sum_c alpha_c * (U_c^T A) * (V_c^T B)  where the (U_c^T A)*(V_c^T B)
               must equal A00*B02 + A10*B12 when combined with W[2,0][c]
      C[2,1] = sum_c beta_c * (U_c^T A) * (V_c^T B)  where combined with W[2,1][c]
               must equal A01*B02 + A11*B12
    
    Actually, let me think differently. The 10 rank-1 terms give 10 bilinear forms.
    Each output entry C[k,i] is a linear combination of these bilinear forms via W[k*n+i][c].
    
    For k=0,1: the 7 Strassen terms already handle these (W[0,i] and W[1,i] for i=0,1).
    For k=2: we need W[2,0] and W[2,1] such that the linear combination of all 10
    bilinear forms gives the correct results.
    
    The 7 Strassen bilinear forms (in terms of A entries a00,a01,a10,a11 and B entries b00,b01,b02,b10,b11,b12):
    
    Term 0: (a00+a11)*(b00+b11)  -> W[0,0]=1, W[1,1]=1
    Term 1: (a10+a11)*b00         -> W[0,1]=1, W[1,1]=-1
    Term 2: a00*(b01-b11)         -> W[1,0]=1, W[1,1]=1
    Term 3: a11*(b10-b00)         -> W[0,0]=1, W[0,1]=1
    Term 4: (a00+a01)*b11         -> W[0,0]=-1, W[1,0]=1
    Term 5: (a10-a00)*(b00+b01)   -> W[1,1]=1
    Term 6: (a01-a11)*(b10+b11)   -> W[0,0]=1
    
    For k=2, we need:
      C[2,0] = a00*b02 + a10*b12
      C[2,1] = a01*b02 + a11*b12
    
    The 7 Strassen forms don't involve b02 or b12 at all (they only use b00,b01,b10,b11).
    So we need the 3 new terms to provide the b02 and b12 dependence.
    
    Let the 3 new terms be:
      Term 7: u7^T A * v7^T B
      Term 8: u8^T A * v8^T B  
      Term 9: u9^T A * v9^T B
    
    We need to find u7,u8,u9 (each 4-dim), v7,v8,v9 (each 6-dim), and w7,w8,w9 (each 6-dim)
    such that:
      For k=0,1: W[0,i][7..9] and W[1,i][7..9] should be 0 (to not disturb Strassen)
      For k=2:
        W[2,0][7]*u7^T A * v7^T B + W[2,0][8]*u8^T A * v8^T B + W[2,0][9]*u9^T A * v9^T B = a00*b02 + a10*b12
        W[2,1][7]*u7^T A * v7^T B + W[2,1][8]*u8^T A * v8^T B + W[2,1][9]*u9^T A * v9^T B = a01*b02 + a11*b12
    
    This is a system where we have 3 bilinear forms (from terms 7,8,9) that must span
    the 2 needed bilinear forms (a00*b02+a10*b12) and (a01*b02+a11*b12).
    
    Simple choice: 
      Term 7: u7 = [1,0,0,0], v7 = [0,0,1,0,0,0]  -> a00*b02
      Term 8: u8 = [0,0,1,0], v8 = [0,0,0,0,0,1]  -> a10*b12
      Term 9: u9 = [0,1,0,0], v9 = [0,0,1,0,0,0]  -> a01*b02
      
    Then:
      C[2,0] = W[2,0][7]*a00*b02 + W[2,0][8]*a10*b12 + W[2,0][9]*a01*b02
      We need: W[2,0][7]=1, W[2,0][8]=1, W[2,0][9]=0
      
      C[2,1] = W[2,1][7]*a00*b02 + W[2,1][8]*a10*b12 + W[2,1][9]*a01*b02
      We need: W[2,1][7]=0, W[2,1][8]=0, W[2,1][9]=1
      
    But wait - we also need a11*b12 for C[2,1]. With only 3 new terms, we can't get all 4
    monomials a00*b02, a10*b12, a01*b02, a11*b12.
    
    Let me reconsider. We need:
      C[2,0] = a00*b02 + a10*b12
      C[2,1] = a01*b02 + a11*b12
    
    With 3 rank-1 terms, we have 3 bilinear forms. Each is a product of a linear form in A
    and a linear form in B. We need these 3 to span a 2-dimensional subspace of the space
    of bilinear forms that includes both target forms.
    
    Actually, the space of bilinear forms in (a00,a01,a10,a11) and (b02,b12) is 4x2=8 dimensional.
    The two target forms span a 2-dim subspace. We need 3 rank-1 forms whose span contains this subspace.
    
    Let me try:
      Term 7: (a00+a10) * b02  -> u7=[1,0,1,0], v7=[0,0,1,0,0,0]
      Term 8: (a01+a11) * b02  -> u8=[0,1,0,1], v8=[0,0,1,0,0,0]
      Term 9: (a00+a10) * b12  -> u9=[1,0,1,0], v9=[0,0,0,0,0,1]
      
    Then:
      C[2,0] = W[2,0][7]*(a00+a10)*b02 + W[2,0][8]*(a01+a11)*b02 + W[2,0][9]*(a00+a10)*b12
      = (W[2,0][7]+W[2,0][9])*a00*b02 + W[2,0][7]*a10*b02 + W[2,0][8]*a01*b02 + W[2,0][8]*a11*b02 + W[2,0][9]*a10*b12
      
    Hmm, this is getting complicated. Let me try a different approach.
    
    Actually, I realize the issue. Let me try:
      Term 7: a00 * b02
      Term 8: a10 * b12  
      Term 9: a01 * b02 + a11 * b12  (but this isn't rank 1!)
    
    A rank-1 bilinear form must be (linear in A) * (linear in B). So a01*b02 + a11*b12 is NOT rank 1.
    
    With 3 rank-1 forms, we can at most span a 3-dim subspace. The two target forms are in an 8-dim space.
    We need the 3 rank-1 forms to include both targets in their span.
    
    Let me try:
      Term 7: a00 * (b02 + b12)  -> u7=[1,0,0,0], v7=[0,0,1,0,0,1]
      Term 8: a10 * b12           -> u8=[0,0,1,0], v8=[0,0,0,0,0,1]
      Term 9: (a01 + a11) * b02   -> u9=[0,1,0,1], v9=[0,0,1,0,0,0]
      
    C[2,0] = W[2,0][7]*a00*(b02+b12) + W[2,0][8]*a10*b12 + W[2,0][9]*(a01+a11)*b02
    Need: a00*b02 + a10*b12
    So: W[2,0][7]=1 (gives a00*b02 + a00*b12), W[2,0][8]=1 (gives a10*b12), W[2,0][9]=0
    But we get extra a00*b12 term. Not good.
    
    Let me try yet another approach. What if we use half-integer coefficients?
    
    Term 7: (a00 + a01) * b02  -> u7=[1,1,0,0], v7=[0,0,1,0,0,0]
    Term 8: (a10 + a11) * b12  -> u8=[0,0,1,1], v8=[0,0,0,0,0,1]
    Term 9: a00 * b12           -> u9=[1,0,0,0], v9=[0,0,0,0,0,1]
    
    C[2,0] = W[2,0][7]*(a00+a01)*b02 + W[2,0][8]*(a10+a11)*b12 + W[2,0][9]*a00*b12
    Need: a00*b02 + a10*b12
    W[2,0][7]=1: gives a00*b02 + a01*b02
    W[2,0][8]=1: gives a10*b12 + a11*b12
    W[2,0][9]=0
    Result: a00*b02 + a01*b02 + a10*b12 + a11*b12. Extra terms.
    
    This is tricky. Let me try a numerical approach instead - set up the full system and solve.
    
    Actually, let me reconsider the problem. Maybe rank 10 is not achievable with this simple approach.
    Let me try a more systematic numerical search.
    """
    n, m, p = 2, 2, 3
    r = 10
    
    # Start from rank-11 and try to find a rank-10 via optimization
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    # Try various strategies
    rng = np.random.RandomState(42)
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Strategy: random restarts with ALS
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt < 50:
            # Truncated rank-11 + perturbation
            U_init = (U11[:, :r] + rng.normal(0, 0.05, (n*m, r))).tolist()
            V_init = (V11[:, :r] + rng.normal(0, 0.05, (m*p, r))).tolist()
            W_init = (W11[:, :r] + rng.normal(0, 0.05, (p*n, r))).tolist()
        elif attempt < 100:
            # Random
            U_init = rng.uniform(-1, 1, (n*m, r)).tolist()
            V_init = rng.uniform(-1, 1, (m*p, r)).tolist()
            W_init = rng.uniform(-1, 1, (p*n, r)).tolist()
        else:
            # Structured: 7 Strassen + 3 random for k=2
            U_init = [[0.0]*r for _ in range(n*m)]
            V_init = [[0.0]*r for _ in range(m*p)]
            W_init = [[0.0]*r for _ in range(p*n)]
            
            # Copy Strassen part (first 7 terms)
            for i in range(n*m):
                for c in range(7):
                    U_init[i][c] = U11[i][c]
            for i in range(m*p):
                for c in range(7):
                    V_init[i][c] = V11[i][c]
            for i in range(p*n):
                for c in range(7):
                    W_init[i][c] = W11[i][c]
            
            # Random for last 3 terms, but constrain W for k=0,1 to be 0
            for c in range(7, r):
                for i in range(n*m):
                    U_init[i][c] = rng.uniform(-1, 1)
                for i in range(m*p):
                    V_init[i][c] = rng.uniform(-1, 1)
                for i in range(p*n):
                    k = i // n
                    if k < 2:
                        W_init[i][c] = 0.0
                    else:
                        W_init[i][c] = rng.uniform(-1, 1)
        
        U, V, W = _als_optimize(U_init, V_init, W_init, n, m, p, r, iters=60, deadline=deadline)
        
        # Round and check
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
        
        res = _residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(res)
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U, V, W
    
    # Try scipy on best result
    if best_U is not None and time.time() < deadline:
        try:
            def _flatten(U, V, W):
                return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
            
            def _unflatten(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                return U, V, W
            
            def _objective(x):
                U, V, W = _unflatten(x)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                            res.append(s - 1.0)
                return res
            
            x0 = _flatten(best_U, best_V, best_W)
            result = least_squares(_objective, x0, method='lm', max_nfev=20000, xtol=1e-15, ftol=1e-15)
            U, V, W = _unflatten(result.x)
            Ur, Vr, Wr = _round_half(U.tolist()), _round_half(V.tolist()), _round_half(W.tolist())
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    best = _rank11_solution()
    
    if time.time() < deadline:
        result10 = _try_slice_absorption(deadline)
        if result10 is not None:
            best = {"U": result10[0], "V": result10[1], "W": result10[2]}
    
    return best
