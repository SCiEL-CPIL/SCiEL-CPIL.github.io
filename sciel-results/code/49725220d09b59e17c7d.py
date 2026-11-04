"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Schoolbook decomposition (rank 75)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        results = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    results[i, j, :] += u_reshaped[i, j] * v_reshaped[j, :] * w_reshaped[:, i]
        
        return np.allclose(results, 1.0, atol=1e-8)
    
    def build_linear_system(U_rand, V_rand, R):
        """Build the linear system A * W_flat = b for given U, V."""
        num_eqs = n * m * p
        num_vars = p * n * R
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = i * m + j
                for k in range(p):
                    v_row = j * p + k
                    w_row = k * n + i
                    for r in range(R):
                        A[idx, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]
                    idx += 1
        
        return A, b
    
    def try_solve_for_W(U_rand, V_rand, R):
        """Solve for W and check if it has half-integer entries."""
        A, b = build_linear_system(U_rand, V_rand, R)
        
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            W_rand = W_flat.reshape(p * n, R)
            
            # Check if W is close to half-integers
            W_check = W_rand * 2
            dist = np.abs(W_check - np.round(W_check))
            max_dist = np.max(dist)
            
            if max_dist < 1e-6:
                W_int = np.round(W_check) / 2.0
                return W_int
        except Exception:
            pass
        
        return None
    
    def generate_patterns(R, attempt):
        """Generate diverse random patterns for U and V."""
        patterns = [
            lambda: (np.random.randint(-1, 2, (n * m, R)).astype(float),
                     np.random.randint(-1, 2, (m * p, R)).astype(float)),
            lambda: (np.random.choice([-1.0, 1.0], size=(n * m, R)),
                     np.random.choice([-1.0, 1.0], size=(m * p, R))),
            lambda: (np.random.randint(-2, 3, (n * m, R)).astype(float),
                     np.random.randint(-2, 3, (m * p, R)).astype(float)),
            lambda: (np.random.randint(-3, 4, (n * m, R)).astype(float),
                     np.random.randint(-3, 4, (m * p, R)).astype(float)),
            # Sparse patterns
            lambda: (np.random.choice([0, 0, 0, 1, -1], size=(n * m, R)).astype(float),
                     np.random.choice([0, 0, 0, 1, -1], size=(m * p, R)).astype(float)),
            # Binary with bias
            lambda: (np.random.choice([0, 1], size=(n * m, R), p=[0.3, 0.7]).astype(float),
                     np.random.choice([0, 1], size=(m * p, R), p=[0.3, 0.7]).astype(float)),
        ]
        
        idx = attempt % len(patterns)
        U_rand, V_rand = patterns[idx]()
        return U_rand, V_rand
    
    # Try ranks from 74 down to 65
    for R in range(rank_schoolbook - 1, 64, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = max(10, int((deadline - 20 - time.time()) / 0.5))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            U_rand, V_rand = generate_patterns(R, attempt)
            W_int = try_solve_for_W(U_rand, V_rand, R)
            
            if W_int is not None:
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_int.tolist()
                
                if validate(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    print(f"Found valid decomposition with rank {R}")
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
# EVOLVE-BLOCK-END
