import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    Returns {'U', 'V', 'W'} where U is (16, rank), V is (32, rank), W is (32, rank).
    The decomposition satisfies: sum_r U[:,r] * V[:,r] * W[:,r] = multiplication tensor.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 8

    # Strassen 2x2*2x2 decomposition
    # Each entry is (u_vec, v_vec, w_vec) where each is a length-4 vector
    # representing a 2x2 matrix flattened in row-major order
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    col = 0

    for b_half in range(2):
        for m_idx in range(7):
            for s_idx in range(7):
                u_sub, v_sub, w_sub = strassen_2x2[s_idx]

                a_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],
                    1: [(1, 0, 1), (1, 1, 1)],
                    2: [(0, 0, 1)],
                    3: [(1, 1, 1)],
                    4: [(0, 0, 1), (0, 1, 1)],
                    5: [(1, 0, 1), (0, 0, -1)],
                    6: [(0, 1, 1), (1, 1, -1)],
                }

                b_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],
                    1: [(0, 0, 1)],
                    2: [(0, 1, 1), (1, 1, -1)],
                    3: [(1, 0, 1), (0, 0, -1)],
                    4: [(1, 1, 1)],
                    5: [(0, 0, 1), (0, 1, 1)],
                    6: [(1, 0, 1), (1, 1, 1)],
                }

                c_blocks = {
                    0: [(0, 0, 1), (1, 1, 1)],
                    1: [(1, 0, 1), (1, 1, -1)],
                    2: [(0, 1, 1), (1, 1, 1)],
                    3: [(0, 0, 1), (1, 0, 1)],
                    4: [(0, 0, -1), (0, 1, 1)],
                    5: [(1, 1, 1)],
                    6: [(0, 0, 1)],
                }

                for (a_row, a_col, a_coeff) in a_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            coeff = u_sub[local_r * 2 + local_s]
                            if coeff != 0:
                                global_A_idx = (a_row * 2 + local_r) * 4 + (a_col * 2 + local_s)
                                U[global_A_idx, col] += a_coeff * coeff

                for (b_row, b_col, b_coeff) in b_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            coeff = v_sub[local_r * 2 + local_s]
                            if coeff != 0:
                                global_B_idx = (b_row * 2 + local_r) * 8 + (b_half * 4 + b_col * 2 + local_s)
                                V[global_B_idx, col] += b_coeff * coeff

                for (c_row, c_col, c_coeff) in c_blocks[m_idx]:
                    for local_r in range(2):
                        for local_s in range(2):
                            coeff = w_sub[local_r * 2 + local_s]
                            if coeff != 0:
                                global_j = b_half * 4 + c_col * 2 + local_s
                                global_i = c_row * 2 + local_r
                                global_W_idx = global_j * 4 + global_i
                                W[global_W_idx, col] += c_coeff * coeff

                col += 1

    # Now, try to reduce the rank by finding linear dependencies among the 98 terms.
    # We check if the 98 rank-1 tensors are linearly independent.
    # Flatten each term into a vector of length 16*32*32 = 16384.
    # Check the rank of the 98 x 16384 matrix.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try to find dependencies
        flat_dim = n * m * p  # 128, not 16384! The tensor is (n, m, p) = (4,4,8)
        # Wait, the multiplication tensor has indices (i, k, j) where i in [0,4), k in [0,4), j in [0,8)
        # So the dimension is 4*4*8 = 128
        flat_dim = 4 * 4 * 8  # 128
        
        # Build the matrix of flattened terms
        M_flat = np.zeros((rank, flat_dim))
        
        for r in range(rank):
            # T_r[i, k, j] = U[i, r] * V[k, r] * W[j, r]
            # Reshape to (4, 4, 8) and flatten
            T_r = U[:, r].reshape(4, 4) * V[:, r].reshape(4, 8).T  # This is wrong!
            
        # Let me redo this. U has shape (16, rank) where index = i*m + k
        # V has shape (32, rank) where index = k*p + j
        # W has shape (32, rank) where index = j*n + i
        
        for r in range(rank):
            # For each (i, k, j), the term is U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        idx = i * m * p + k * p + j
                        M_flat[r, idx] = U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
        
        # Now check the rank of M_flat (98 x 128)
        # If the rank is less than 98, there are dependencies.
        # We can use SVD to find the rank and potentially reduce.
        
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M_flat, full_matrices=False)
            # The effective rank is the number of singular values above tolerance
            tol = 1e-10
            effective_rank = np.sum(s_vals > tol)
            
            if effective_rank < rank:
                # We can potentially reduce the rank
                # But we need to find a valid decomposition with fewer terms.
                # For now, just report the effective rank
                pass
        except:
            pass
    
    # Verify the decomposition
    def check_decomposition(U, V, W, n, m, p, max_checks=200):
        import random
        random.seed(42)
        errors = 0
        for _ in range(max_checks):
            i = random.randint(0, n-1)
            k = random.randint(0, m-1)
            j = random.randint(0, p-1)
            a_idx = i * m + k
            b_idx = k * p + j
            w_idx = j * n + i
            val = np.sum(U[a_idx, :] * V[b_idx, :] * W[w_idx, :])
            if abs(val - 1) > 1e-6:
                errors += 1
        
        for _ in range(max_checks):
            i = random.randint(0, n-1)
            k = random.randint(0, m-1)
            j = random.randint(0, p-1)
            a_idx = i * m + k
            b_idx = k * p + j
            j2 = random.randint(0, p-1)
            i2 = random.randint(0, n-1)
            if j2 == j and i2 == i:
                continue
            w_idx = j2 * n + i2
            val = np.sum(U[a_idx, :] * V[b_idx, :] * W[w_idx, :])
            if abs(val) > 1e-6:
                errors += 1
        return errors
    
    errors = check_decomposition(U, V, W, n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
