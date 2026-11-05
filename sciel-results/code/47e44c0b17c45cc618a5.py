import time
import numpy as np
from fractions import Fraction

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

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        terms.append((u, v, w))

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        terms.append((u, v, w))

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

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

        all_terms.extend(strassen_2x2_block(0, 0, 4))
        all_terms.extend(strassen_2x2_block(0, 2, 4))
        all_terms.extend(strassen_2x2_block(2, 0, 4))
        all_terms.extend(strassen_2x2_block(2, 2, 4))
        
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

    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=1000, tol=1e-8):
        """
        Try to reduce the rank using alternating least squares.
        """
        U = np.array(U_init, dtype=float).copy()
        V = np.array(V_init, dtype=float).copy()
        W = np.array(W_init, dtype=float).copy()
        
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return None
        
        # Try to project to lower rank
        # Use SVD to find a lower-rank approximation
        best_result = None
        
        for attempt in range(5):
            U_r = U[:, :target_rank].copy()
            V_r = V[:, :target_rank].copy()
            W_r = W[:, :target_rank].copy()
            
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                    
                # Update U: for each c, U[:,c] should minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||
                # This is a linear least squares problem
                # Reshape for efficient computation
                
                U_r = U_r.reshape(n, m, target_rank)
                V_r = V_r.reshape(m, p, target_rank)
                W_r = W_r.reshape(p, n, target_rank)
                
                # Compute current tensor
                tensor = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                residual = np.ones((n, m, p), dtype=float) - tensor
                
                # Update U: for each i,j, find U[i,j,:] that best fits
                for i in range(n):
                    for j in range(m):
                        # residual[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                        # For fixed i,j, this is linear in U[i,j,:]
                        # Build the system: for each k, sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = residual[i,j,k]
                        # This is a system of p equations in target_rank unknowns
                        A_mat = np.zeros((p, target_rank))
                        b_vec = np.zeros(p)
                        for k in range(p):
                            for c in range(target_rank):
                                A_mat[k, c] = V_r[j, k, c] * W_r[k, i, c]
                            b_vec[k] = residual[i, j, k]
                        
                        # Solve least squares
                        try:
                            U_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            U[i, j, :] = U_new
                        except:
                            pass
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                # Update V similarly
                tensor = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                residual = np.ones((n, m, p), dtype=float) - tensor
                
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        b_vec = np.zeros(n)
                        for i in range(n):
                            for c in range(target_rank):
                                A_mat[i, c] = U_r[i, j, c] * W_r[k, i, c]
                            b_vec[i] = residual[i, j, k]
                        
                        try:
                            V_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            V[j, k, :] = V_new
                        except:
                            pass
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                # Update W similarly
                tensor = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                residual = np.ones((n, m, p), dtype=float) - tensor
                
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        b_vec = np.zeros(m)
                        for j in range(m):
                            for c in range(target_rank):
                                A_mat[j, c] = U_r[i, j, c] * V_r[j, k, c]
                            b_vec[j] = residual[i, j, k]
                        
                        try:
                            W_new = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            W[k, i, :] = W_new
                        except:
                            pass
                
                if time.time() - start_time > DEADLINE:
                    break
            
            # Check if the reduced-rank decomposition is valid
            U_final = U.reshape(n*m, target_rank)
            V_final = V.reshape(m*p, target_rank)
            W_final = W.reshape(p*n, target_rank)
            
            U_r = U_final.reshape(n, m, target_rank)
            V_r = V_final.reshape(m, p, target_rank)
            W_r = W_final.reshape(p, n, target_rank)
            
            tensor = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            error = np.max(np.abs(tensor - np.ones((n, m, p), dtype=float)))
            
            if error < tol:
                # Convert to list format
                U_list = [[0] * target_rank for _ in range(n*m)]
                V_list = [[0] * target_rank for _ in range(m*p)]
                W_list = [[0] * target_rank for _ in range(p*n)]
                for r in range(n*m):
                    for c in range(target_rank):
                        if U_final[r, c] != 0:
                            U_list[r][c] = round(U_final[r, c], 10)
                for r in range(m*p):
                    for c in range(target_rank):
                        if V_final[r, c] != 0:
                            V_list[r][c] = round(V_final[r, c], 10)
                for r in range(p*n):
                    for c in range(target_rank):
                        if W_final[r, c] != 0:
                            W_list[r][c] = round(W_final[r, c], 10)
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        return best_result

    def rank_1_peeling():
        initial = build_4x4x7_optimized()
        U_init = np.array(initial["U"], dtype=float)
        V_init = np.array(initial["V"], dtype=float)
        W_init = np.array(initial["W"], dtype=float)
        
        # Try to reduce rank
        current_rank = U_init.shape[1]
        best_result = initial
        
        # Try various target ranks
        for target_rank in range(current_rank - 1, max(0, current_rank - 10), -1):
            if time.time() - start_time > DEADLINE:
                break
            
            result = try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=200)
            if result is not None:
                best_result = result
                break
        
        return best_result

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        # First try the optimized structural decomposition
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()
        
        # Try to reduce rank
        if time.time() - start_time < DEADLINE - 10:
            reduced = rank_1_peeling()
            if reduced is not None:
                U_red = np.array(reduced["U"], dtype=float)
                V_red = np.array(reduced["V"], dtype=float)
                W_red = np.array(reduced["W"], dtype=float)
                rank_red = U_red.shape[1]
                
                if rank_red < rank:
                    U_r = U_red.reshape(n, m, rank_red)
                    V_r = V_red.reshape(m, p, rank_red)
                    W_r = W_red.reshape(p, n, rank_red)
                    
                    tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                    if np.allclose(tensor_decomp, target, atol=1e-8):
                        return reduced
        
        return result
    except Exception:
        return build_schoolbook()
