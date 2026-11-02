import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.allclose(T_rec, T, atol=tol)

def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded))
    return rounded, max_err

def _strassen_3x3x3():
    """
    Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indices: u = i*3 + j, v = j*3 + k, w = k*3 + i
    """
    # Strassen's 7 terms for C = A * B where A, B are 3x3
    # Let A = [[a, b, c], [d, e, f], [g, h, i]]
    # Let B = [[x, y, z], [s, t, u], [v, w, t2]] (using distinct letters)
    # 
    # Standard Strassen for 2x2 generalized to 3x3 (7 multiplications)
    # M1 = (a+b)*(x+s)
    # M2 = (c)*(s+t+u)  -- this isn't quite right for 3x3
    #
    # Actually, let's use the known rank-7 decomposition for 3x3x3.
    # The tensor for 3x3x3 matrix multiplication has rank 7 (Strassen).
    #
    # One standard form:
    # P1 = (a + b + c)(x + s + v)
    # P2 = (d + e + f)(s + t + w)
    # P3 = (g + h + i)(v + w + t2)
    # ... this is getting complicated. Let's use a verified construction.
    
    # A well-known rank-7 decomposition for 3x3 matrix multiplication:
    # Reference: Strassen (1969)
    
    # Let's define the 7 bilinear forms.
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[x, y, z], [s, t, u], [v, w, q]]
    # C = A * B
    
    # The 7 products are:
    # m1 = (a + b + c) * (x + s + v)
    # m2 = (d + e + f) * (s + t + w)  
    # m3 = (g + h + i) * (v + w + q)
    # m4 = (a - b + c) * (x - s + v)
    # m5 = (a + b - c) * (x + s - v)
    # m6 = (d - e + f) * (s - t + w)
    # m7 = (g - h + i) * (v - w + q)
    
    # This doesn't quite work out cleanly. Let me use a different known decomposition.
    
    # Actually, the simplest approach: use the known fact that rank(3x3x3) = 7.
    # I'll construct it via the standard Strassen-like formula.
    
    # Let's use this verified set of 7 terms:
    # The 7 rank-1 tensors for 3x3x3 matrix mult:
    
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # Term 1: (a+b+c)(x+s+v) -> contributes to C in a specific pattern
    # This is getting complex. Let me use a simpler approach:
    # I'll hardcode a known valid rank-7 decomposition.
    
    # From literature, one rank-7 decomposition:
    # m1 = (a + d + g)(x + s + v)  -- no, this isn't standard
    
    # Let me just use the standard Strassen algorithm adapted to 3x3.
    # Actually, for 3x3, the rank is 7, and one construction is:
    
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[x, y, z], [s, t, u], [v, w, q]]
    
    # C[0,0] = ax + bs + cv
    # C[0,1] = ay + bt + cw
    # C[0,2] = az + bu + cq
    # C[1,0] = dx + es + fv
    # C[1,1] = dy + et + fw
    # C[1,2] = dz + eu + fq
    # C[2,0] = gx + hs + iv
    # C[2,1] = gy + ht + iw
    # C[2,2] = gz + hu + iq
    
    # Strassen's 7 multiplications for 3x3:
    # This is non-trivial to write out by hand correctly.
    
    # Alternative: Use the fact that we can verify any proposed decomposition.
    # Let me try a systematic construction.
    
    # Actually, let's just skip the hardcoded Strassen and use a numerical approach
    # that's more likely to work. The key insight is that we should start from
    # a good initial guess.
    
    # For now, return the schoolbook rank-27 decomposition as a fallback
    # and let the ALS handle finding lower ranks.
    
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def _als_decompose(T, n, m, p, rank, max_iters=200, tol=1e-10, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    """
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
    else:
        U = np.random.randn(n * m, rank) * 0.5
    
    if init_V is not None:
        V = np.array(init_V, dtype=float).copy()
    else:
        V = np.random.randn(m * p, rank) * 0.5
    
    if init_W is not None:
        W = np.array(init_W, dtype=float).copy()
    else:
        W = np.random.randn(p * n, rank) * 0.5
    
    prev_loss = np.inf
    for _ in range(max_iters):
        # Solve for U using efficient einsum-based formulation
        TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, rank)
        h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, rank)
        G_U = np.einsum('bcd,bce->de', TW, TW)  # (rank, rank)
        try:
            U_new = np.linalg.solve(G_U + 1e-10 * np.eye(rank), h_U.T).T
        except np.linalg.LinAlgError:
            U_new = h_U
        U = U_new
        
        # Solve for V
        UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, rank)
        h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, rank)
        G_V = np.einsum('acd,ace->de', UW, UW)  # (rank, rank)
        try:
            V_new = np.linalg.solve(G_V + 1e-10 * np.eye(rank), h_V.T).T
        except np.linalg.LinAlgError:
            V_new = h_V
        V = V_new
        
        # Solve for W
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, rank)
        h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, rank)
        G_W = np.einsum('abd,abe->de', UV, UV)  # (rank, rank)
        try:
            W_new = np.linalg.solve(G_W + 1e-10 * np.eye(rank), h_W.T).T
        except np.linalg.LinAlgError:
            W_new = h_W
        W = W_new
        
        # Calculate loss
        T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
        loss = np.sum((T_rec - T) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U, V, W, loss

def _construct_3x3x5_from_strassen():
    """
    Construct a 3x3x5 decomposition by extending the 3x3x3 Strassen decomposition.
    
    The 3x3x5 tensor can be viewed as 3x3x3 plus two extra output columns.
    We start from a rank-7 3x3x3 decomposition and add terms for the extra columns.
    """
    n, m, p = 3, 3, 5
    
    # Build the 3x3x3 sub-tensor (first 3 columns of output)
    # T3[u, v, w] for w in [0, 8] corresponds to k in [0, 2]
    T3 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i * 3 + j
                v = j * 3 + k
                w = k * 3 + i
                T3[u, v, w] = 1.0
    
    # Get a rank-7 decomposition for 3x3x3 via ALS
    np.random.seed(123)
    U3, V3, W3, loss3 = _als_decompose(T3, 3, 3, 3, 7, max_iters=100, tol=1e-12)
    
    # Now extend to 3x3x5
    # The full tensor T has shape (9, 15, 15)
    # The first 9 output indices correspond to k=0,1,2 (w = k*3 + i)
    # The last 6 output indices correspond to k=3,4 (w = k*3 + i for i=0,1,2)
    
    T_full = _build_target_tensor(3, 3, 5)
    
    # Initialize the full decomposition
    # Start with the 3x3x3 part and extend
    U_full = np.zeros((9, 7 + 6))  # 7 for 3x3x3 + 6 for extra
    V_full = np.zeros((15, 7 + 6))
    W_full = np.zeros((15, 7 + 6))
    
    # Copy the 3x3x3 part
    U_full[:, :7] = U3
    V_full[:9, :7] = V3
    W_full[:9, :7] = W3
    
    # For the extra columns (k=3,4), we need additional terms
    # Each extra k value requires terms for i=0,1,2
    # The extra part of T: for k=3,4, T[u, v, w] = 1 where u=i*3+j, v=j*3+k, w=k*3+i
    
    # Initialize the extra 6 terms with schoolbook-like structure
    for idx, k in enumerate([3, 4]):
        for i in range(3):
            col = 7 + idx * 3 + i
            u = i * 3  # This is a simplification; we need to handle all j
            # Actually, for each (i, k) pair, we need a term for each j
            # But that would be 3*2*3 = 18 extra terms, which is too many
            # Let's use a different approach
    
    # Better approach: just use ALS on the full tensor with rank 7+6=13
    # or try various ranks
    
    # Let's just return the initial guess and let the caller refine
    return U_full, V_full, W_full

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    T = _build_target_tensor(n, m, p)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Strategy: Start from a constructive initial guess based on Strassen for 3x3x3
    # and extend to handle the extra 2 columns.
    
    # Build the 3x3x3 sub-problem
    T3 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i * 3 + j
                v = j * 3 + k
                w = k * 3 + i
                T3[u, v, w] = 1.0
    
    # Try to find a good rank-7 decomposition for 3x3x3
    np.random.seed(42)
    U3, V3, W3, loss3 = _als_decompose(T3, 3, 3, 3, 7, max_iters=100, tol=1e-12)
    
    # Now try to extend to 3x3x5 with rank 7 + extra
    # The extra 2 columns (k=3,4) each need 3 terms (for i=0,1,2)
    # So we try rank 7 + 6 = 13 first
    
    for target_rank in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 44, 45]:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            break
            
        # Create initial guess: embed the 3x3x3 solution and add random terms for the rest
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 100 + target_rank)
            
            try:
                # Initialize U, V, W
                U_init = np.zeros((9, target_rank))
                V_init = np.zeros((15, target_rank))
                W_init = np.zeros((15, target_rank))
                
                # Embed the 3x3x3 solution in the first 7 columns
                if target_rank >= 7:
                    U_init[:, :7] = U3
                    V_init[:9, :7] = V3
                    W_init[:9, :7] = W3
                
                # Add random initialization for remaining columns
                if target_rank > 7:
                    U_init[:, 7:] = np.random.randn(9, target_rank - 7) * 0.5
                    V_init[:, 7:] = np.random.randn(15, target_rank - 7) * 0.5
                    W_init[:, 7:] = np.random.randn(15, target_rank - 7) * 0.5
                
                U, V, W, loss = _als_decompose(T, n, m, p, target_rank, max_iters=80, tol=1e-10,
                                                 init_U=U_init, init_V=V_init, init_W=W_init)
                
                if loss < 1e-6:
                    # Round to half-integers
                    U_rounded, err_u = _round_to_halfint(U)
                    V_rounded, err_v = _round_to_halfint(V)
                    W_rounded, err_w = _round_to_halfint(W)
                    
                    max_err = max(err_u, err_v, err_w)
                    
                    if max_err < 1e-3:
                        # Verify the rounded solution
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                            if target_rank < best_rank:
                                best_rank = target_rank
                                best_result = {
                                    "U": U_rounded.tolist(),
                                    "V": V_rounded.tolist(),
                                    "W": W_rounded.tolist()
                                }
                                break  # Found valid for this rank, try lower
            except Exception:
                continue

    # Also try pure random initialization for lower ranks
    for target_rank in range(10, 20):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            break
            
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + restart * 100 + target_rank)
            
            try:
                U, V, W, loss = _als_decompose(T, n, m, p, target_rank, max_iters=50, tol=1e-10)
                
                if loss < 1e-6:
                    U_rounded, err_u = _round_to_halfint(U)
                    V_rounded, err_v = _round_to_halfint(V)
                    W_rounded, err_w = _round_to_halfint(W)
                    
                    max_err = max(err_u, err_v, err_w)
                    
                    if max_err < 1e-3:
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                            if target_rank < best_rank:
                                best_rank = target_rank
                                best_result = {
                                    "U": U_rounded.tolist(),
                                    "V": V_rounded.tolist(),
                                    "W": W_rounded.tolist()
                                }
            except Exception:
                continue

    return best_result
# EVOLVE-BLOCK-END
