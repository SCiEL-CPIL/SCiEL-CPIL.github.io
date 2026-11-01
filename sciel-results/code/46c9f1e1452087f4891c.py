"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import random
import numpy as np

def construct():
    """Return the schoolbook tensor decomposition for 2x3x9 matrix multiplication.
    
    We attempt to find a lower-rank decomposition by first finding a low-rank
    decomposition for 2x3x3 and then replicating it for the 9 columns (3 blocks).
    """
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_233_full(U, V, W, R):
        """Verify 2x3x3 decomposition including zeros.
        U: (6, R), V: (9, R), W: (6, R)
        Check all (u, v, w) in 6x9x6 space.
        """
        # Convert to numpy for speed
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Target tensor T of shape (6, 9, 6)
        # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k
        # Indices: u = i*3+j, v = j*3+k, w = k*2+i
        T = np.zeros((6, 9, 6), dtype=int)
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    u = i * 3 + j
                    v = j * 3 + k
                    w = k * 2 + i
                    T[u, v, w] = 1
        
        # Compute decomposition tensor
        # D[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a tensor product sum.
        # Can be computed as:
        # D = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        
        try:
            D = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        except Exception:
            return False
            
        # Check exact match
        if not np.array_equal(D, T):
            return False
            
        return True

    def find_233_decomp(target_rank, max_trials=1000):
        """Search for 2x3x3 decomposition of given rank."""
        for _ in range(max_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Generate sparse random matrices
            # Entries from {-1, 0, 1} with p=0.4 for non-zero
            # Higher density might help find solutions that satisfy zeros?
            # Actually, sparsity helps satisfy zeros.
            U = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice([-1, 0, 1]) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(6)]
            
            if verify_233_full(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    def build_239_from_233(decomp_233):
        """Build 2x3x9 decomposition from 2x3x3 decomposition."""
        R = len(decomp_233["U"][0])
        total_rank = 3 * R
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(27)]
        W = [[0] * total_rank for _ in range(18)]
        
        U_233 = decomp_233["U"]
        V_233 = decomp_233["V"]
        W_233 = decomp_233["W"]
        
        for block in range(3):
            for c in range(R):
                col = block * R + c
                for row in range(6):
                    U[row][col] = U_233[row][c]
                
                for j in range(3):
                    for k_local in range(3):
                        k_actual = 3 * block + k_local
                        V[j * 9 + k_actual][col] = V_233[j * 3 + k_local][c]
                
                for k_local in range(3):
                    for i in range(2):
                        k_actual = 3 * block + k_local
                        W[k_actual * 2 + i][col] = W_233[k_local * 2 + i][c]
        
        return {"U": U, "V": V, "W": W}

    # Try ranks for 2x3x3
    # Schoolbook is 18.
    # Known lower bound is likely 12?
    # Try 12, 13, 14, 15, 16, 17.
    
    best_result = None
    
    for R in [12, 13, 14, 15, 16, 17]:
        if time.time() - start_time > DEADLINE:
            break
        
        trials = 1000
        
        decomp_233 = find_233_decomp(R, trials)
        if decomp_233:
            result = build_239_from_233(decomp_233)
            return result

    # Fallback to schoolbook
    return build_schoolbook()
