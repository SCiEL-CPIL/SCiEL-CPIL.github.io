import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
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

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        terms = []
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        terms.append((u, v, w))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        terms.append((u, v, w))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        terms.append((u, v, w))

        return terms

    def build_4x4x7_optimized():
        all_terms = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3 of B)
        # Recursive Strassen: 7 * 7 = 49 terms
        
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices(block_idx, local_idx, dim_a, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                all_terms.append((u, v, w))

        # Part 2: 4x4 * 4x2 (columns 4-5 of B)
        # 4 Strassen 2x2x2 blocks
        all_terms.extend(strassen_2x2_block(0, 0, 4))
        all_terms.extend(strassen_2x2_block(0, 2, 4))
        all_terms.extend(strassen_2x2_block(2, 0, 4))
        all_terms.extend(strassen_2x2_block(2, 2, 4))
        
        # Part 3: 4x4 * 4x1 (column 6 of B)
        # 16 schoolbook terms
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                all_terms.append((u, v, w))

        rank = len(all_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(all_terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]
        
        return {"U": U, "V": V, "W": W}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        # Verification with random test matrices
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        # Test with several random matrices
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randint(-3, 4, size=(n, m)).astype(float)
            B = rng.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            
            # Compute using the decomposition
            # P_r = (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k])
            # C[i,k] = sum_r W[k*n+i, r] * P_r
            
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            # P_r = einsum('ijr,jkr->r', A, V_r) * ... no, that's not right.
            # P_r = (U_r @ A.T flattened) * (V_r @ B.T flattened)
            # Actually: sum_{i,j} U[i,j,r] * A[i,j] = np.sum(U_r * A[:,:,None], axis=(0,1))
            # sum_{j,k} V[j,k,r] * B[j,k] = np.sum(V_r * B[:,:,None], axis=(0,1))
            
            U_flat = U_r * A[:, :, np.newaxis]
            V_flat = V_r * B[:, :, np.newaxis]
            P = np.sum(U_flat, axis=(0,1)) * np.sum(V_flat, axis=(0,1))
            C_computed = np.sum(W_r * P[np.newaxis, np.newaxis, :], axis=2)
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return build_schoolbook()
        
        return result
    except Exception:
        return build_schoolbook()
