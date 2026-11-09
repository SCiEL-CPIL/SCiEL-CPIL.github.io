import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Known rank 17 solution
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Term 1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Term 2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Term 3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Term 4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Term 5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Term 6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Term 7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Term 8
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    # Term 9
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    # Term 10
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    # Term 11
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    # Term 12
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    # Term 13
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    # Term 14
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # Term 15
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    # Term 16
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    # Term 17
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def to_list(mat):
        return mat.tolist()
    
    # Attempt to find rank 16 solution
    # Use a more robust search strategy with a strict time limit
    start_time = time.time()
    deadline = start_time + 110.0 # 110 seconds for search, 10s buffer
    
    R = 16
    # Pre-compute the tensor indices for efficiency
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_indices.append((u, v, w))
                
    # We can also pre-compute the zero constraints
    # But the linear system approach only uses the non-zero constraints (18 equations)
    # and relies on the fact that if the non-zero constraints are satisfied with integer W,
    # the zero constraints are *likely* satisfied if the structure is preserved.
    # However, strictly speaking, we must check all 54*6=324 constraints.
    
    # To speed up the check, we can vectorize the verification.
    
    for _ in range(50000):
        if time.time() > deadline:
            break
            
        # Random sparse initialization
        # Using small integers
        U = np.random.randint(-1, 2, size=(n*m, R))
        V = np.random.randint(-1, 2, size=(m*p, R))
        
        # Build linear system for W
        # 18 non-zero entries in tensor
        A_mat = np.zeros((18, R * p * n)) 
        b_vec = np.zeros(18)
        
        for idx, (u, v, w) in enumerate(tensor_indices):
            b_vec[idx] = 1.0
            for r in range(R):
                coeff = U[u, r] * V[v, r]
                # Unknown is W[w, r]. Index in A_mat is w*R + r
                A_mat[idx, w * R + r] = coeff
        
        # Solve least squares
        try:
            W_sol, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_sol = W_sol.reshape((p*n, R))
            
            # Check if W_sol is close to integer or half-integer
            # Try integers
            W_int = np.round(W_sol)
            if np.allclose(W_sol, W_int, atol=1e-6):
                W_candidate = W_int
                is_half = False
            else:
                # Try half-integers
                W_half = np.round(W_sol * 2) / 2.0
                if np.allclose(W_sol, W_half, atol=1e-6):
                    W_candidate = W_half
                    is_half = True
                else:
                    continue
            
            # Full verification
            # Reconstruct tensor and check
            # C[u, v] = sum_r U[u, r] * V[v, r] * W[w, r] for the specific w?
            # No, the decomposition is:
            # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
            # We need to check if this holds for ALL u, v, w.
            
            # Vectorized check:
            # Create tensor T_recon of shape (n*m, m*p, p*n)
            # This is 6 * 9 * 6 = 324 elements.
            # T_recon[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
            
            # We can compute this efficiently.
            # Let's just loop over the 324 entries, it's small.
            valid = True
            for u in range(n * m):
                for v in range(m * p):
                    for w in range(p * n):
                        # Expected value
                        # Find i, j, k
                        i = u // m
                        j = u % m
                        j_v = v // p
                        k = v % p
                        i_w = w % n
                        k_w = w // n
                        
                        if j == j_v and i == i_w and k == k_w:
                            expected = 1.0
                        else:
                            expected = 0.0
                        
                        # Computed value
                        # sum_r U[u, r] * V[v, r] * W[w, r]
                        val = np.dot(U[u, :], V[v, :] * W[w, :])
                        
                        if abs(val - expected) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                # Found a valid rank 16 solution
                # Convert to list
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W_candidate)}
                
        except Exception:
            pass

    # Return the known rank 17 solution
    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
