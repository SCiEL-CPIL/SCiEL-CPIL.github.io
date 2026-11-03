import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
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
        """Try to eliminate redundant terms using null-space analysis."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        def term_sparsity(idx):
            u_zeros = sum(1 for i in range(n*m) if U_list[i][idx] == 0)
            v_zeros = sum(1 for j in range(m*p) if V_list[j][idx] == 0)
            w_zeros = sum(1 for k in range(p*n) if W_list[k][idx] == 0)
            return u_zeros + v_zeros + w_zeros
        
        for _ in range(20):
            if time_left() < 5:
                break
            progress = False
            order = sorted(range(rank), key=term_sparsity, reverse=True)
            for term_idx in order:
                if time_left() < 5:
                    break
                if term_idx >= rank:
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
                
                if verify(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once using fast verification."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
        
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        # Precompute contribution of each term to each (i,j,k)
        total_dim = n * m * p
        contrib = np.zeros((total_dim, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    contrib[idx, :] = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
        
        total_contrib = contrib.sum(axis=1)
        
        max_pairs = min(2000, rank * (rank - 1) // 2)
        pairs_checked = 0
        
        for idx1 in range(rank):
            if time_left() < 10 or pairs_checked >= max_pairs:
                break
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10 or pairs_checked >= max_pairs:
                    break
                pairs_checked += 1
                
                residual = total_contrib - contrib[:, idx1] - contrib[:, idx2]
                if np.allclose(residual, 1.0, rtol=1e-8, atol=1e-10):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def try_gaussian_elimination(U_list, V_list, W_list, rank):
        """
        Use null-space analysis to find linear dependencies among terms.
        """
        if time_left() < 20:
            return U_list, V_list, W_list, rank
        
        total_dim = n * m * p
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        M = np.zeros((total_dim, rank), dtype=float)
        for t in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i*m+j, t]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_arr[j*p+k, t]
                        w_val = W_arr[k*n+i, t]
                        if v_val == 0 or w_val == 0:
                            continue
                        M[i*m*p + j*p + k, t] = u_val * v_val * w_val
        
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            null_space = Vt[s < tol]
            nullity = len(null_space)
            
            if nullity == 0:
                return U_list, V_list, W_list, rank
            
            for ns_vec in null_space:
                if time_left() < 10:
                    break
                max_idx = int(np.argmax(np.abs(ns_vec)))
                max_val = ns_vec[max_idx]
                if abs(max_val) < 1e-8:
                    continue
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][max_idx] = 0
                for j in range(m*p):
                    V_new[j][max_idx] = 0
                for k in range(p*n):
                    W_new[k][max_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != max_idx]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 1
        except Exception:
            pass
        
        return U_list, V_list, W_list, rank

    def try_linear_combination_merge(U_list, V_list, W_list, rank):
        """
        Try to find two terms a, b such that their tensor vectors are proportional.
        """
        if time_left() < 15:
            return U_list, V_list, W_list, rank
        
        total_dim = n * m * p
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        vecs = []
        for t in range(rank):
            vec = np.zeros(total_dim, dtype=float)
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i*m+j, t]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_arr[j*p+k, t]
                        w_val = W_arr[k*n+i, t]
                        if v_val == 0 or w_val == 0:
                            continue
                        vec[i*m*p + j*p + k] = u_val * v_val * w_val
            vecs.append(vec)
        
        for a in range(rank):
            if time_left() < 10:
                break
            for b in range(a + 1, rank):
                if time_left() < 10:
                    break
                nz = np.nonzero(vecs[a])[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                if abs(vecs[a][idx]) < 1e-12:
                    continue
                c = vecs[b][idx] / vecs[a][idx]
                
                if np.allclose(vecs[b], c * vecs[a], rtol=1e-8, atol=1e-10):
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][a] *= (1 + c)
                    
                    for i in range(n*m):
                        U_new[i][b] = 0
                    for j in range(m*p):
                        V_new[j][b] = 0
                    for k in range(p*n):
                        W_new[k][b] = 0
                    
                    keep = [j for j in range(rank) if j != b]
                    U_list = [[row[j] for j in keep] for row in U_new]
                    V_list = [[row[j] for j in keep] for row in V_new]
                    W_list = [[row[j] for j in keep] for row in W_new]
                    rank -= 1
                    
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list, rank
        
        return U_list, V_list, W_list, rank

    def try_local_optimization(U_list, V_list, W_list, rank):
        """
        Attempt to reduce rank by 1 by solving a linear system for a new term.
        This is a heuristic approach: try to find a term that can replace two existing terms.
        """
        if time_left() < 30:
            return U_list, V_list, W_list, rank
            
        # This is a simplified version of a more complex local search
        # For now, just return the same result as we don't have a robust implementation
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
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,2),(4,5)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
        ([(0,1),(2,3)], [(0,5),(1,3),(2,4)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,4)], [(0,3),(1,2),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,4),(3,5)]),
        ([(0,2),(1,3)], [(0,4),(1,2),(3,5)]),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,2)], [(0,1),(2,4),(3,5)]),
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,4),(3,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try Gaussian elimination if time permits
        if time_left() > 50:
            U_list, V_list, W_list, new_rank = try_gaussian_elimination(U_list, V_list, W_list, new_rank)
        
        # Try linear combination merge if time permits
        if time_left() > 40:
            U_list, V_list, W_list, new_rank = try_linear_combination_merge(U_list, V_list, W_list, new_rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 25:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
            
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
