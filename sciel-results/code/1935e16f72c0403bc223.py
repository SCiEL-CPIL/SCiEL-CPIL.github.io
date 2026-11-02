import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposition tensor
    # T_dec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_dec, T, atol=tol)

def _build_rank_39():
    """
    Build the known rank 39 decomposition.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n     # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n     # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]

        # Strassen terms
        # m1 = (a00+a11)(b00+b11)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2      # B[0, 10]
    v_g2 = p + k2  # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]

    # Q1: a00 * B[0, 10]
    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    # Q2: a01 * B[1, 10]
    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    # Q3: a10 * B[0, 10]
    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    # Q4: a11 * B[1, 10]
    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1
    
    return U, V, W

def _try_rank_38_search(start_time, deadline):
    """
    Attempt to find a rank-38 decomposition using the splitting method.
    
    Key insight: The tensor T[a,b,c] for a=0,1,2,3 (where a = i*m+j) can be
    split into two parts:
    - Part 0: a=0,1 (i=0, j=0,1) - first row of A
    - Part 1: a=2,3 (i=1, j=0,1) - second row of A
    
    Each part is a 2 x 22 x 22 tensor (2 rows of A, 22 B entries, 22 C entries).
    
    We search for a common set of bilinear forms (V and W vectors) that can
    represent both parts with minimal total rank.
    """
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Split into two 2 x 22 x 22 tensors
    T0 = T[0:2, :, :]  # i=0: a=0 (j=0), a=1 (j=1)
    T1 = T[2:4, :, :]  # i=1: a=2 (j=0), a=3 (j=1)
    
    # Strategy: Try to find a shared basis of bilinear forms.
    # Start with the rank-39 solution and try to merge terms.
    
    U39, V39, W39 = _build_rank_39()
    
    # The rank-39 solution has 5 Strassen blocks (7 terms each) + 4 schoolbook terms.
    # Each Strassen block is independent. Can we share structure between blocks?
    
    # Alternative: Try a different block structure.
    # What if we use a 2x2x3 block (rank 11 or 12) instead of separate 2x2x2 + 2x2x1?
    
    # Let's try: 3 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x5
    # 3*7 + R(2,2,5) = 21 + R(2,2,5)
    # R(2,2,5) <= 7*2 + 4 = 18 (using 2 Strassen + 1 schoolbook)
    # Total: 21 + 18 = 39. Same.
    
    # What about 2 blocks of 2x2x3 (rank 11 each) + 1 block of 2x2x5 (rank 18)?
    # 2*11 + 18 = 40. Worse.
    
    # What about a completely different approach?
    # Use the fact that we have 2 rows of A and 11 columns of B.
    # The output is 2x11.
    
    # Let's try a greedy approach: build the decomposition term by term,
    # choosing terms that maximize coverage of the remaining tensor.
    
    T_remaining = T.copy()
    U_new = []
    V_new = []
    W_new = []
    
    best_rank = 39
    best_U, best_V, best_W = U39.copy(), V39.copy(), W39.copy()
    
    # Try different target ranks
    for target_rank in range(38, 25, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a rank-target_rank decomposition
        success = _greedy_decomposition(T, target_rank, deadline - 10)
        if success:
            U_g, V_g, W_g = success
            if _verify(U_g, V_g, W_g, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_g, V_g, W_g
                break
    
    if best_rank < 39:
        return best_U, best_V, best_W
    
    return None

def _greedy_decomposition(T, target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition using a greedy approach.
    
    At each step, choose a rank-1 tensor u v^T w^T that best approximates
    the remaining tensor.
    """
    n, m, p = DIMENSIONS
    T_remaining = T.copy()
    
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(target_rank):
        if time.time() > deadline:
            return None
        
        # Find the best rank-1 approximation
        # We want to maximize <T_remaining, u v^T w^T> / (||u|| ||v|| ||w||)
        # This is equivalent to finding the best rank-1 approximation.
        
        # Use a simple approach: try all combinations of basis vectors
        # For efficiency, use random sampling + local optimization
        
        best_score = -1
        best_u, best_v, best_w = None, None, None
        
        # Try random initializations
        n_trials = 200
        for trial in range(n_trials):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # Normalize
            u /= np.linalg.norm(u) + 1e-12
            v /= np.linalg.norm(v) + 1e-12
            w /= np.linalg.norm(w) + 1e-12
            
            # Local optimization: iterate over u, v, w
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update u: u = (T_remaining * v * w) / ||...||
                Tvw = np.einsum('abc,b,c->a', T_remaining, v, w)
                u_new = Tvw / (np.linalg.norm(Tvw) + 1e-12)
                
                # Update v: v = (T_remaining * u * w) / ||...||
                Twu = np.einsum('abc,a,c->b', T_remaining, u_new, w)
                v_new = Twu / (np.linalg.norm(Twu) + 1e-12)
                
                # Update w: w = (T_remaining * u * v) / ||...||
                Twu = np.einsum('abc,a,b->c', T_remaining, u_new, v_new)
                w_new = Twu / (np.linalg.norm(Twu) + 1e-12)
                
                u, v, w = u_new, v_new, w_new
            
            # Score: inner product with T_remaining
            score = np.einsum('abc,a,b,c', T_remaining, u, v, w)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            return None
        
        # Subtract the best rank-1 tensor
        T_remaining -= np.outer(best_u, np.outer(best_v, best_w)).reshape(n*m, m*p, p*n)
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        # Check if we've converged
        if np.linalg.norm(T_remaining) < 1e-8:
            break
    
    # Check if the decomposition is exact
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    if len(U_list) < target_rank:
        # Pad with zeros
        while U.shape[1] < target_rank:
            U = np.column_stack([U, np.zeros(n*m)])
            V = np.column_stack([V, np.zeros(m*p)])
            W = np.column_stack([W, np.zeros(p*n)])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # 3. Try to find rank 38 (or lower)
    result = _try_rank_38_search(start_time, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
