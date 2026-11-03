import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_fast(U, V, W, rank, tol=1e-9):
        """Verify decomposition using numpy matrix operations for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i, j, k] == 1 if i, j, k form a valid multiplication index, else 0?
        # No, the tensor for matrix multiplication is T[i, j, k] = 1 if the indices match?
        # The task says: "exactly decompose the matrix multiplication tensor".
        # The tensor is defined by T[i, j, k] = 1 if (i, j, k) is a term in the sum?
        # Actually, for matrix multiplication C = AB, C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
        # T[i, j, k] = 1.
        # The decomposition is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
        # Wait, the shapes are U(n*m, rank), V(m*p, rank), W(p*n, rank).
        # So U is indexed by (i, j), V by (j, k), W by (k, i).
        # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # We want sum_r ... = 1 for all i, j, k.
        
        # Let's construct the full tensor T of size (n*m) x (m*p) x (p*n)?
        # No, T is (n, m, p).
        # T[i, j, k] = 1.
        # Check: sum_r U[i, j, r] V[j, k, r] W[k, i, r] == 1.
        
        # Efficient check:
        # For each r, form outer product? No.
        # We can iterate over i, j, k.
        
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        num_blocks = len(j_pairs) * len(k_pairs)
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(15):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify_fast(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 10:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10:
                    return U_list, V_list, W_list, rank
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][idx1] = 0
                    U_new[i][idx2] = 0
                for j in range(m*p):
                    V_new[j][idx1] = 0
                    V_new[j][idx2] = 0
                for k in range(p*n):
                    W_new[k][idx1] = 0
                    W_new[k][idx2] = 0
                    
                if verify_fast(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
            
        if rank > 45:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 15:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 15:
                    return U_list, V_list, W_list, rank
                for idx3 in range(idx2 + 1, rank):
                    if time_left() < 15:
                        return U_list, V_list, W_list, rank
                    
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][idx1] = 0
                        U_new[i][idx2] = 0
                        U_new[i][idx3] = 0
                    for j in range(m*p):
                        V_new[j][idx1] = 0
                        V_new[j][idx2] = 0
                        V_new[j][idx3] = 0
                    for k in range(p*n):
                        W_new[k][idx1] = 0
                        W_new[k][idx2] = 0
                        W_new[k][idx3] = 0
                        
                    if verify_fast(U_new, V_new, W_new, rank):
                        keep = [j for j in range(rank) if j != idx1 and j != idx2 and j != idx3]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 3
                        return try_triple_elimination(U_list, V_list, W_list, rank)
                        
        return U_list, V_list, W_list, rank

    def try_linear_combination_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate a term by expressing it as a linear combination of others."""
        if time_left() < 25:
            return U_list, V_list, W_list, rank
            
        if rank > 40:
            return U_list, V_list, W_list, rank
            
        for term_idx in range(rank - 1, -1, -1):
            if time_left() < 25:
                break
            
            all_rows = []
            all_rhs = []
            
            for i in range(n*m):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(U_list[i][t])
                all_rows.append(row)
                all_rhs.append(U_list[i][term_idx])
            
            for j in range(m*p):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(V_list[j][t])
                all_rows.append(row)
                all_rhs.append(V_list[j][term_idx])
            
            for k in range(p*n):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(W_list[k][t])
                all_rows.append(row)
                all_rhs.append(W_list[k][term_idx])
            
            A = np.array(all_rows, dtype=float)
            b = np.array(all_rhs, dtype=float)
            
            try:
                coeffs, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    residual = np.linalg.norm(A @ coeffs - b)
                    if residual > 1e-8:
                        continue
                
                # Check for half-integer or integer coefficients
                coeffs_rounded = np.round(coeffs * 2) / 2
                if np.linalg.norm(coeffs - coeffs_rounded) > 1e-6:
                    continue
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify_fast(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
            except:
                continue

        return U_list, V_list, W_list, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_fast(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
        
        # Try triple elimination if time permits
        if time_left() > 30:
            U_list, V_list, W_list, new_rank = try_triple_elimination(U_list, V_list, W_list, new_rank)
        
        # Try linear combination elimination if time permits
        if time_left() > 40:
            U_list, V_list, W_list, new_rank = try_linear_combination_elimination(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
