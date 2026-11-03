"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor from decomposition using vectorized ops
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        # Outer product: T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_vec[i, j] * v_vec[j, :] * w_vec[:, i]
    # Target tensor is all ones
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-9):
        return False
    return True

def _verify_fast(n, m, p, U, V, W, rank):
    """Faster verification using einsum."""
    U = np.array(U, dtype=float).reshape(n, m, rank)
    V = np.array(V, dtype=float).reshape(m, p, rank)
    W = np.array(W, dtype=float).reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_structured(n, m, p):
    """Try structured decompositions exploiting the all-ones structure."""
    # Strategy: For each (i,j) pair, we need sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all k
    # If we fix W[k,i,c] = 1 for all k,i,c (use rank m*p columns), then we need
    # sum_c U[i,j,c] * V[j,k,c] = 1 for all (i,j,k)
    # This means for each j, the vector [U[i,j,c] for c] dotted with [V[j,k,c] for c] = 1 for all i,k
    
    # Simpler: use the fact that we can write 1 = sum of rank-1 terms
    # Let's try: assign each (i,j,k) triple to a column, but merge where possible
    
    # Approach 1: Use n*m*p columns but with values that might allow merging
    # Approach 2: Use a "grid" decomposition
    
    # Let's try: for each j, use p columns where V[j,k,c] is nonzero only for specific k
    # and W[k,i,c] handles the i-dependence
    
    # Actually, let's try a simple structured approach:
    # Use rank = n*p columns. For each (i,k) pair, have one column c(i,k)
    # U[i,j,c(i,k)] = 1/m for all j (so sum over j... no that doesn't work)
    
    # Let me think differently. The constraint is:
    # For all i,j,k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # 
    # If we set W[k,i,c] = 1 for all (k,i,c) that are used, and V[j,k,c] = 1 for all j,
    # then we need sum_c U[i,j,c] = 1 for all (i,j), which gives rank >= n*m with U having
    # n*m independent constraints... but V and W also need to be consistent.
    
    # Let's try: rank = m*p. For each (j,k), one column c(j,k).
    # U[i,j,c(j,k)] = 1/n (so that for fixed i,j, sum over k of U[i,j,c(j,k)] * ... = 1)
    # V[j,k,c(j,k)] = 1
    # W[k,i,c(j,k)] = 1
    # Then sum_c U[i,j,c] V[j,k,c] W[k,i,c] = U[i,j,c(j,k)] * V[j,k,c(j,k)] * W[k,i,c(j,k)] = (1/n)*1*1 = 1/n
    # That's not 1. Need U[i,j,c(j,k)] = 1. But then for fixed (i,j), we have p columns each with U=1,
    # so sum_c U[i,j,c] V[j,k,c] W[k,i,c] = sum over all (j',k') of U[i,j,c(j',k')] V[j,k,c(j',k')] W[k,i,c(j',k')]
    # = sum over k' of V[j,k,c(j,k')] * W[k,i,c(j,k')] (since U[i,j,c(j',k')] = 1 only when j'=j)
    # = sum over k' of (1 if k'=k else 0) * 1 = 1. 
    # Wait, V[j,k,c(j',k')] is 1 only when j'=j and k'=k, so V[j,k,c(j,k')] = 1 if k'=k else 0.
    # So the sum is 1. This works!
    
    # So rank = m*p = 21 should work with:
    # For each (j,k) in 0..m-1 x 0..p-1, column c = j*p+k
    # U[i*m+j, c] = 1 for all i
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1 for all i
    
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j][c] = 1
            V[j * p + k][c] = 1
            for i in range(n):
                W[k * n + i][c] = 1
    
    if _verify_fast(n, m, p, U, V, W, rank):
        return U, V, W, rank
    
    # Try even lower: rank = n*p with symmetric construction
    rank2 = n * p
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            for j in range(m):
                U2[i * m + j][c] = 1
            for j in range(m):
                V2[j * p + k][c] = 1
            W2[k * n + i][c] = 1
    
    if _verify_fast(n, m, p, U2, V2, W2, rank2):
        return U2, V2, W2, rank2
    
    return None

def _try_random_decomposition(n, m, p, rank, seed=0):
    """Try to find a random integer/half-integer decomposition of given rank."""
    rng = random.Random(seed)
    
    for attempt in range(30):
        vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)])
        
        if _verify_fast(n, m, p, U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_structured_decomposition(n, m, p):
    """Try structured approaches to find lower rank."""
    return _try_structured(n, m, p)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin before 150s kill
    
    # First try structured decompositions
    best_rank = n * m * p  # schoolbook fallback
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j][c] = 1
                best_V[j * p + k][c] = 1
                best_W[k * n + i][c] = 1
    
    # Try structured approaches
    struct_result = _try_structured(n, m, p)
    if struct_result is not None:
        U_s, V_s, W_s, rank_s = struct_result
        if rank_s < best_rank:
            best_rank = rank_s
            best_U, best_V, best_W = U_s, V_s, W_s
    
    # Try random search for lower ranks
    min_rank_to_try = max(m * p, n * p, 20)  # lower bound
    max_rank_to_try = min(best_rank - 1, 50)
    
    if time.time() < deadline - 30:
        seed_counter = 0
        for rank in range(min_rank_to_try, max_rank_to_try + 1):
            if time.time() > deadline - 30:
                break
            
            for seed in range(seed_counter, seed_counter + 15):
                if time.time() > deadline - 30:
                    break
                
                result = _try_random_decomposition(n, m, p, rank, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(n, m, p, U, V, W, rank):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        break
            
            seed_counter += 15
    
    # Clean values to exact integers or half-integers
    def clean_values(val):
        rounded = round(val * 2) / 2
        if abs(val - rounded) < 1e-10:
            return rounded
        return val
    
    U_final = [[clean_values(x) for x in row] for row in best_U]
    V_final = [[clean_values(x) for x in row] for row in best_V]
    W_final = [[clean_values(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
