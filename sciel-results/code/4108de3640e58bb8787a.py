"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses a structured approach:
    1. Start with schoolbook (rank 96)
    2. Try to reduce rank using a combinatorial/structured search
    3. Use verification to ensure exactness
    
    The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    T[i, j, k] = 1 for all valid (i, j, k)
    where i in [0,n), j in [0,m), k in [0,p)
    
    We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s limit

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_decomposition_numpy(U_mat, V_mat, W_mat, rank):
        """
        Vectorized verification using numpy.
        U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        Returns True if the decomposition is exact.
        """
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        result = np.zeros((n, m, p))
        for c in range(rank):
            result += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])
        
        return np.allclose(result, np.ones((n,m,p)), atol=1e-9)

    def verify_decomposition_exact(U_mat, V_mat, W_mat, rank):
        """
        Exact verification for integer/half-integer entries.
        U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        Returns True if the decomposition is exact.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if total != 1:
                        return False
        return True

    # Strategy: Try to find lower rank decompositions
    # For matrix multiplication, known lower bounds and results:
    # - Schoolbook: n*m*p = 96
    # - Strassen-style reductions may help
    # - For (4,4,6), we can try ranks from ~24 to 95
    
    # The trivial lower bound is max(n*p, m*p, n*m) = max(24, 24, 16) = 24
    # But we can potentially do better
    
    # Try ranks from lower bound up to schoolbook
    # Use a structured search with random initialization and verification
    
    def random_search_structured(rank, num_tries=100, seed=42):
        """
        Try random initializations with structured patterns and check if they form a valid decomposition.
        Uses a mix of random and structured approaches.
        """
        np_rng = np.random.RandomState(seed)
        
        for attempt in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                # Pure random
                U_rand = np_rng.randn(n*m, rank) * 0.5
                V_rand = np_rng.randn(m*p, rank) * 0.5
                W_rand = np_rng.randn(p*n, rank) * 0.5
            elif attempt % 3 == 1:
                # Sparse random - many zeros
                U_rand = np.zeros((n*m, rank))
                V_rand = np.zeros((m*p, rank))
                W_rand = np.zeros((p*n, rank))
                for c in range(rank):
                    # Each column has a few non-zero entries
                    for _ in range(3):
                        U_rand[np.random.randint(n*m), c] = np.random.choice([-1, 1])
                        V_rand[np.random.randint(m*p), c] = np.random.choice([-1, 1])
                        W_rand[np.random.randint(p*n), c] = np.random.choice([-1, 1])
            else:
                # Block-structured
                U_rand = np.zeros((n*m, rank))
                V_rand = np.zeros((m*p, rank))
                W_rand = np.zeros((p*n, rank))
                for c in range(rank):
                    # Assign to blocks
                    i_block = c % n
                    j_block = (c // n) % m
                    k_block = (c // (n*m)) % p
                    for di in range(1):
                        for dj in range(1):
                            for dk in range(1):
                                i = (i_block + di) % n
                                j = (j_block + dj) % m
                                k = (k_block + dk) % p
                                U_rand[i*m+j, c] = 1
                                V_rand[j*p+k, c] = 1
                                W_rand[k*n+i, c] = 1
            
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            # Verify with numpy first (faster)
            if verify_decomposition_numpy(U_rand, V_rand, W_rand, rank):
                # Double-check with exact verification
                if verify_decomposition_exact(U_list, V_list, W_list, rank):
                    return U_list, V_list, W_list, True
        
        return None, None, None, False

    def try_strassen_style(rank):
        """
        Try a Strassen-style decomposition pattern.
        For 2x2 * 2x2, Strassen uses rank 7 instead of 8.
        Generalize to 4x4 * 4x6.
        """
        if time.time() > deadline - 10:
            return None, None, None, False
        
        # This is a heuristic - try patterns inspired by Strassen
        U_try = np.zeros((n*m, rank))
        V_try = np.zeros((m*p, rank))
        W_try = np.zeros((p*n, rank))
        
        # For now, this is a placeholder - a real Strassen generalization
        # would require careful design
        return None, None, None, False

    # Try decreasing ranks from a reasonable lower bound
    # Lower bound: at least max(n, m, p) * min(n, m, p) / something
    # For (4,4,6), let's try from 24 up to schoolbook
    
    # First, try some structured approaches at various ranks
    for rank in range(24, min(40, rank_schoolbook)):
        if time.time() > deadline - 15:
            break
        
        if rank >= best_rank:
            continue
        
        # Try random search
        U_try, V_try, W_try, found = random_search_structured(rank, num_tries=30, seed=rank*1000+42)
        
        if found:
            best_rank = rank
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            # Continue searching for even lower rank
            continue
    
    # If we haven't found anything better, try a few more ranks with more tries
    for rank in range(24, min(35, rank_schoolbook)):
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            continue
        
        U_try, V_try, W_try, found = random_search_structured(rank, num_tries=50, seed=rank*2000+123)
        
        if found:
            best_rank = rank
            best_result = {"U": U_try, "V": V_try, "W": W_try}
    
    return best_result
# EVOLVE-BLOCK-END
