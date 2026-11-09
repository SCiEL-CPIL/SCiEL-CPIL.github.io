import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 36
    
    # 1. Generate Schoolbook Solution (Rank 36)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][col] = 1
                V_sb[v_idx][col] = 1
                W_sb[w_idx][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    def verify_fast(U, V, W, R):
        """
        Fast verification using precomputed arrays
        """
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        # Use multiple random tests
        for _ in range(5):
            A = np.random.randint(-3, 4, size=(n, m), dtype=np.int32)
            B = np.random.randint(-3, 4, size=(m, p), dtype=np.int32)
            
            C_direct = A @ B
            
            LA = np.einsum('ij,ijr->ir', A, U_reshaped)
            LB = np.einsum('jk,jkr->kr', B, V_reshaped)
            C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
            
            if not np.array_equal(C_direct, C_decomp):
                return False
                
        return True

    # 2. Search for lower ranks with structured approach
    # Try to find if rank 35 or lower is possible
    
    # For small dimensions, we can try a more systematic search
    # Focus on ranks that might have structure
    
    for rank in range(schoolbook_rank - 1, 25, -1):
        if time.time() > deadline - 20.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Try different strategies for each rank
        max_trials = min(50, int(remaining * 5))
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
                
            # Strategy: Try to perturb schoolbook or use structured random
            if trial % 3 == 0:
                # Sparse random
                U = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(n * m)]
                V = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(m * p)]
                W = [[random.choice([-1, 0, 1]) if random.random() < 0.3 else 0 for _ in range(rank)] for _ in range(p * n)]
            elif trial % 3 == 1:
                # Dense random with small values
                U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
                W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            else:
                # Try to use first rank columns of schoolbook and randomize rest
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                for col in range(min(rank, schoolbook_rank)):
                    if col < schoolbook_rank:
                        i = col // (m * p)
                        j = (col // p) % m
                        k = col % p
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                # Randomize some entries
                for _ in range(10):
                    u_row = random.randint(0, n * m - 1)
                    u_col = random.randint(0, rank - 1)
                    U[u_row][u_col] = random.choice([-1, 0, 1])
                    v_row = random.randint(0, m * p - 1)
                    v_col = random.randint(0, rank - 1)
                    V[v_row][v_col] = random.choice([-1, 0, 1])
                    w_row = random.randint(0, p * n - 1)
                    w_col = random.randint(0, rank - 1)
                    W[w_row][w_col] = random.choice([-1, 0, 1])
            
            if verify_fast(U, V, W, rank):
                best_rank = rank
                best_solution = {"U": U, "V": V, "W": W}
                return best_solution
                
    return best_solution
# EVOLVE-BLOCK-END
