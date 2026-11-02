import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B block indices: 00, 01, 10, 11 -> local 0, 1, 2, 3
    # C block indices: 00, 01, 10, 11 -> local 0, 1, 2, 3
    
    # (A_coeffs, B_coeffs, C_coeffs)
    # C_coeffs are added to the output
    # Note: B_coeffs are for the 2x2 block
    strassen_steps = [
        # 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00 += 1, C11 += 1
        # 2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C10 += 1, C11 += -1
        # 3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # C01 += 1, C11 += 1
        # 4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # C00 += 1, C10 += 1
        # 5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # C00 += -1, C01 += 1
        # 6
        ([0, 1, -1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),  # C11 += 1
        # 7
        ([0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 0, 1]),  # C00 += 1, C11 += 1
    ]
    
    num_blocks = p // 2
    
    for l in range(num_blocks):
        for s in range(7):
            c_idx = l * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_steps[s]
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    U[u_idx, c_idx] = a_coeffs[i * m + j]
                    
            # Fill V
            # B is m x p. Block l covers columns 2l and 2l+1.
            # b_coeffs are for local indices 00, 01, 10, 11.
            # Local 00 -> j=0, k=2l
            # Local 01 -> j=0, k=2l+1
            # Local 10 -> j=1, k=2l
            # Local 11 -> j=1, k=2l+1
            for local_j in range(m):
                for local_k in range(2):
                    b_local_idx = local_j * 2 + local_k
                    b_global_k = 2 * l + local_k
                    v_idx = local_j * p + b_global_k
                    V[v_idx, c_idx] = b_coeffs[b_local_idx]
                    
            # Fill W
            # C is p x n.
            # c_coeffs are for local indices 00, 01, 10, 11.
            # Local 00 -> i=0, k=2l
            # Local 01 -> i=0, k=2l+1
            # Local 10 -> i=1, k=2l
            # Local 11 -> i=1, k=2l+1
            for local_i in range(n):
                for local_k in range(2):
                    c_local_idx = local_i * 2 + local_k
                    c_global_k = 2 * l + local_k
                    w_idx = c_global_k * n + local_i
                    W[w_idx, c_idx] = c_coeffs[c_local_idx]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
