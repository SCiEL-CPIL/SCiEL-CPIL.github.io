import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _verify_vec(U, V, W, n, m, p, tol=1e-9):
    """
    Vectorized verification.
    T[i,j,k] should equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    # Reshape for efficient einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) < tol)


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each i: compute M[i] = U[i,:,:] * W[:,i,:]  (m x p matrix)
    # M[i][j,k] = sum_r U[i,j,r]*W[k,i,r]
    # Then T[i,j,k] = sum_r M[i][j,k] * V[j,k,r]... no that's not right.
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let A[i,j,r] = U[i,j,r], B[j,k,r] = V[j,k,r], C[k,i,r] = W[k,i,r]
    # T[i,j,k] = sum_r A[i,j,r]*B[j,k,r]*C[k,i,r]
    # For fixed i: T[i,j,k] = sum_r A[i,j,r]*B[j,k,r]*C[k,i,r]
    # Define D[i,j,k] = sum_r A[i,j,r]*C[k,i,r]  (this is m x p for each i)
    # Then T[i,j,k] = sum_k' ... no.
    # Actually: T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # For fixed i: let E[j,k] = sum_r B[j,k,r]*C[k,i,r]  (m x p)
    # Then T[i,j,k] = sum_r A[i,j,r] * B[j,k,r] * C[k,i,r]
    # This is not simply A[i,j,:] dot E[j,k] because C depends on k.
    # Let's just use einsum, it's fast enough.
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) < 1e-9)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96 schoolbook
    
    # Schoolbook decomposition
    sb = _schoolbook(n, m, p)
    best_rank = rank
    best_U = np.array(sb['U'], dtype=np.int64)
    best_V = np.array(sb['V'], dtype=np.int64)
    best_W = np.array(sb['W'], dtype=np.int64)
    
    # Try to find lower rank decompositions
    # Strategy: use structured approaches and efficient verification
    
    np.random.seed(42)
    
    # Try ranks from low to high, but be smart about it
    # Lower bound is max(n*m, m*p, p*n) = 32
    # Try a few structured and random approaches
    
    # First, try some structured decompositions
    # For 3x4x8, we can try to exploit the factorization
    
    # Try random search with vectorized verification
    # Key: verification via einsum is fast, so we can try many candidates
    
    tried_ranks = set()
    
    for try_rank in range(32, rank):  # from lower bound up to schoolbook
        if time.time() > deadline - 10:
            break
        
        if try_rank in tried_ranks:
            continue
        tried_ranks.add(try_rank)
        
        # Try a few random initializations
        n_attempts = 20 if try_rank < 60 else 10
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U_rand = np.random.randint(-1, 2, (n * m, try_rank))
            V_rand = np.random.randint(-1, 2, (m * p, try_rank))
            W_rand = np.random.randint(-1, 2, (p * n, try_rank))
            
            if _verify_vec(U_rand, V_rand, W_rand, n, m, p):
                best_rank = try_rank
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                break
        
        if best_rank < rank:
            break
    
    # Try local refinement: try to reduce rank by removing redundant columns
    if best_rank < rank:
        # Try to remove columns
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        cur_rank = best_rank
        
        for _ in range(50):  # max 50 column removals
            if time.time() > deadline - 5:
                break
            
            removed = False
            for r in range(cur_rank - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Try removing column r
                U_test = np.delete(U_cur, r, axis=1)
                V_test = np.delete(V_cur, r, axis=1)
                W_test = np.delete(W_cur, r, axis=1)
                
                if _verify_vec(U_test, V_test, W_test, n, m, p):
                    U_cur = U_test
                    V_cur = V_test
                    W_cur = W_test
                    cur_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        best_rank = cur_rank
        best_U = U_cur
        best_V = V_cur
        best_W = W_cur
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
