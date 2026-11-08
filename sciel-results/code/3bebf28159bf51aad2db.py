import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    
    n_m = n * m  # 10
    m_p = m * p  # 25
    p_n = p * n  # 10
    
    # Target tensor indices: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_indices = np.array(target_indices)
    target_set = set(map(tuple, target_indices))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n_m, rank), dtype=int)
        V = np.zeros((m_p, rank), dtype=int)
        W = np.zeros((p_n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_exact(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T = np.zeros((n_m, m_p, p_n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        for a, b, c in target_indices:
            if abs(T[a, b, c] - 1) > 1e-9:
                return False
        mask = np.zeros((n_m, m_p, p_n), dtype=bool)
        for a, b, c in target_indices:
            mask[a, b, c] = True
        if not np.all(abs(T[~mask]) < 1e-9):
            return False
        return True
    
    def verify_fast(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        vals = np.array([np.einsum('r,r,r->', U[a], V[b], W[c]) for a, b, c in target_indices])
        if not np.all(np.abs(vals - 1) < 1e-6):
            return False
        # Check a sample of zeros
        for _ in range(30):
            a = random.randint(0, n_m - 1)
            b = random.randint(0, m_p - 1)
            c = random.randint(0, p_n - 1)
            if (a, b, c) in target_set:
                continue
            val = np.einsum('r,r,r->', U[a], V[b], W[c])
            if abs(val) > 1e-6:
                return False
        return True
    
    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to reduce rank using structured approaches
    # Approach 1: Try to find a lower rank decomposition using iterative methods
    
    # For matrix multiplication, there are known algorithms with lower rank
    # Let's try a more systematic search
    
    # First, let's try to use the fact that we can combine terms
    # Try ranks from best_rank-1 down to 40
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Use a smarter initialization: start from schoolbook and try to merge
        # Create a random linear combination approach
        
        num_attempts = 2000
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Strategy: Start with schoolbook and apply random linear transformations
            # to the rank dimensions, then try to zero out some rows/cols
            
            # Generate random mixing matrices
            # We'll work with a subset of the schoolbook terms
            
            # Alternative: Use a random sparse structure
            # Create random matrices with structure
            
            # For matrix multiplication, we can use the fact that
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # Each output element needs m multiplications
            
            # Let's try a random bilinear algorithm approach
            # Generate random U, V, W with small integers
            
            # Use a smarter distribution: many zeros, some ones, few twos
            def gen_sparse_matrix(rows, cols, density=0.3, max_val=2):
                M = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < density:
                            M[i, j] = random.choice([-max_val, -1, 1, max_val])
                return M
            
            U = gen_sparse_matrix(n_m, rank, 0.25, 2)
            V = gen_sparse_matrix(m_p, rank, 0.25, 2)
            W = gen_sparse_matrix(p_n, rank, 0.25, 2)
            
            if verify_fast(U, V, W):
                if verify_exact(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
    
    # If we found something better, try to improve further
    if best_rank < n * m * p:
        for rank in range(best_rank - 1, 39, -1):
            if time.time() > deadline:
                break
            
            num_attempts = 3000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                def gen_sparse_matrix(rows, cols, density=0.2, max_val=2):
                    M = np.zeros((rows, cols))
                    for i in range(rows):
                        for j in range(cols):
                            if random.random() < density:
                                M[i, j] = random.choice([-max_val, -1, 1, max_val])
                    return M
                
                U = gen_sparse_matrix(n_m, rank, 0.2, 2)
                V = gen_sparse_matrix(m_p, rank, 0.2, 2)
                W = gen_sparse_matrix(p_n, rank, 0.2, 2)
                
                if verify_fast(U, V, W):
                    if verify_exact(U, V, W):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        break
    
    return best
# EVOLVE-BLOCK-END
