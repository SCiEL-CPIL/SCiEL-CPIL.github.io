# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Total rank: 7 outer Strassen steps * 11 inner terms (7 Strassen + 4 Schoolbook)
    initial_rank = 77
    
    U = [[0] * initial_rank for _ in range(n * m)]
    V = [[0] * initial_rank for _ in range(m * p)]
    W = [[0] * initial_rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Inner Decomposition for a 2x2 * 2x3 block
    inner_strassen = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    inner_schoolbook = [
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]
    
    # Construct the decomposition
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        # Process Inner Strassen (7 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            c_global_rank = base_rank + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

        # Process Inner Schoolbook (4 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            c_global_rank = base_rank + 7 + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k_local, coeff) in B_inner:
                k_global_in_block = 2 
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k_global_in_block)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k_local, coeff) in C_inner:
                k_global_in_block = 2
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k_global_in_block)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    # Convert to NumPy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check time
    if time.time() - start_time > 100:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }

    # Attempt simple reduction: remove zero columns
    # A column r is zero if U[:,r], V[:,r], W[:,r] are all zero?
    # Or if the term is zero.
    # We check if the tensor term is zero.
    # Term r is u_r (outer) v_r (outer) w_r.
    # If any is zero, the term is zero.
    
    active_indices = []
    for r in range(initial_rank):
        if not (np.allclose(U_np[:, r], 0) or np.allclose(V_np[:, r], 0) or np.allclose(W_np[:, r], 0)):
            active_indices.append(r)
            
    if len(active_indices) < initial_rank:
        U_new = U_np[:, active_indices]
        V_new = V_np[:, active_indices]
        W_new = W_np[:, active_indices]
        return {
            "U": U_new.tolist(),
            "V": V_new.tolist(),
            "W": W_new.tolist()
        }
    else:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }

# EVOLVE-BLOCK-END
