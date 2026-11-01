import numpy as np
import time

def _strassen_28():
    """Fallback: rank-28 Strassen-based decomposition."""
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            U[0, col] = U_strassen[r][0]
            U[1, col] = U_strassen[r][1]
            U[2, col] = U_strassen[r][2]
            U[3, col] = U_strassen[r][3]

            v00 = 0 * p + 2 * t
            v01 = 0 * p + (2 * t + 1)
            v10 = 1 * p + 2 * t
            v11 = 1 * p + (2 * t + 1)
            V[v00, col] = V_strassen[r][0]
            V[v01, col] = V_strassen[r][1]
            V[v10, col] = V_strassen[r][2]
            V[v11, col] = V_strassen[r][3]

            w00 = (2 * t) * n + 0
            w01 = (2 * t + 1) * n + 0
            w10 = (2 * t) * n + 1
            w11 = (2 * t + 1) * n + 1
            W[w00, col] = W_strassen[r][0]
            W[w01, col] = W_strassen[r][1]
            W[w10, col] = W_strassen[r][2]
            W[w11, col] = W_strassen[r][3]

            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # must equal the matrix multiplication tensor:
    # T[a,b,c] = 1 if a corresponds to A[i,j], b to B[j,k], c to C[i,k]
    # A is 2x2: a = i*2 + j (i=row, j=col of A)
    # B is 2x8: b = j*8 + k (j=row of B, k=col of B)  
    # Wait, B is m x p = 2 x 8, so B[j,k] with j in {0,1}, k in {0..7}
    # b = j*p + k = j*8 + k
    # C is p x n = 8 x 2, so C[k,i] with k in {0..7}, i in {0,1}
    # c = k*n + i = k*2 + i
    
    # T[a,b,c] = sum_j A[i,j]*B[j,k] where a=i*2+j, b=j*8+k, c=k*2+i
    # So T[a,b,c] = 1 if there exists j such that a=i*2+j, b=j*8+k, c=k*2+i
    # i.e., j = a % 2, i = a // 2, j' = b // 8, k = b % 8, i' = c % 2, k' = c // 2
    # T = 1 if j == j' and i == i' and k == k'
    
    for a in range(n * m):
        i = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                # Expected value: 1 if j_a == j_b and i == i_c and k == k_c
                expected = 1.0 if (j_a == j_b and i == i_c and k == k_c) else 0.0
                # Computed value
                computed = 0.0
                for t in range(rank):
                    computed += U[a, t] * V[b, t] * W[c, t]
                if abs(computed - expected) > tol:
                    return False
    return True

def _try_rank_r_search(r, time_budget, rng):
    """Try to find a rank-r decomposition via random search + local refinement."""
    n, m, p = 2, 2, 8
    start_time = time.time()
    
    # Initialize with random half-integer values
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_iters = 5000
    for it in range(num_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization from {-1, 0, 1}
        U = rng.integers(-1, 2, size=(n*m, r))
        V = rng.integers(-1, 2, size=(m*p, r))
        W = rng.integers(-1, 2, size=(p*n, r))
        
        # Quick check: compute tensor and compare
        # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Use matrix operations for speed
        # For each t, the contribution is outer(U[:,t], outer(V[:,t], W[:,t]))
        # This is expensive, so do a partial check
        
        # Compute for a subset of (a,b,c) triples
        # Actually, let's compute the full tensor efficiently
        # T = sum_t np.einsum('at,bt,ct->abc', U, V, W)
        # But this creates a 4x16x16 tensor
        
        # Faster: for each a, compute M_a[b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # M_a = V @ diag(U[a,:]) @ W.T  ... no
        # M_a[b,c] = sum_t V[b,t]*U[a,t]*W[c,t] = (V * U[a,:]) @ W.T  element-wise
        # Actually: M_a = V @ np.diag(U[a,:]) @ W.T
        # Or: M_a = (V * U[a,:][None,:]) @ W.T
        
        # Let's just do the full check with einsum for speed
        T = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Build expected tensor
        # T_exp[a,b,c] = 1 if j_a==j_b and i==i_c and k==k_c
        T_exp = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            i = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and i == i_c and k == k_c:
                        T_exp[a, b, c] = 1.0
        
        err = np.max(np.abs(T - T_exp))
        if err < 1e-9:
            return U, V, W, True
        
        if err < best_score:
            best_score = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    if best_U is not None and best_score < 1e-9:
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses a time-bounded search to find the lowest rank decomposition.
    
    Interface:
      U: shape (n*m, rank) = (4, rank), integer entries
      V: shape (m*p, rank) = (16, rank), integer entries
      W: shape (p*n, rank) = (16, rank), integer entries, following W[k*n+i, r] convention
    """
    n, m, p = 2, 2, 8
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Start with the known rank-28 solution
    fallback = _strassen_28()
    
    # Try to find lower rank solutions
    # Theoretical lower bound: at least max(4, 16, 16) = 16? No, that's not right.
    # The rank must be at least the "slice rank" lower bound.
    # For now, try ranks from 16 upward
    
    rng = np.random.default_rng(42)
    
    # Try rank 16 first (theoretical minimum based on dimensions)
    for r in range(16, 28):
        if time.time() > deadline - 5:
            break
        
        time_budget = min(10, max(2, deadline - time.time() - 5))
        U, V, W, success = _try_rank_r_search(r, time_budget, rng)
        
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no lower rank found, return the rank-28 Strassen solution
    return fallback
