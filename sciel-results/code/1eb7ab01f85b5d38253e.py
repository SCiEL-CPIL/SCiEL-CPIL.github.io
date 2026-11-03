import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_full(U, V, W, rank):
        """Full verification: check all 60 tensor entries."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
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
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        changed = True
        while changed and time_left() > 5:
            changed = False
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
                
                if verify_full(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    changed = True
                    break
        
        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U, V, W, rank):
        """Try to eliminate pairs of terms whose combined contribution can be absorbed."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        improved = True
        while improved and time_left() > 10:
            improved = False
            for a in range(rank - 1, -1, -1):
                if time_left() < 10:
                    break
                for b in range(a - 1, -1, -1):
                    if time_left() < 10:
                        break
                    # Try removing both terms a and b
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][a] = 0
                        U_new[i][b] = 0
                    for j in range(m*p):
                        V_new[j][a] = 0
                        V_new[j][b] = 0
                    for k in range(p*n):
                        W_new[k][a] = 0
                        W_new[k][b] = 0
                    
                    if verify_full(U_new, V_new, W_new, rank):
                        # Remove both columns
                        keep = [j for j in range(rank) if j != a and j != b]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 2
                        improved = True
                        break
                if improved:
                    break
        
        return U_list, V_list, W_list, rank

    def solve_for_W(U, V, rank):
        """Given U and V, solve for W such that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        num_unknowns = p * n * rank
        num_eqs = n * m * p
        
        if num_unknowns < num_eqs:
            return None
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        w_idx = (k * n + i) * rank + r
                        A[eq_idx, w_idx] = U_arr[i*m+j, r] * V_arr[j*p+k, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.max(np.abs(A @ W_flat - b)) > 1e-8:
                return None
            W_arr = W_flat.reshape(p * n, rank)
            W_snapped = np.round(W_arr * 2) / 2
            if np.max(np.abs(A @ W_snapped.flatten() - b)) > 1e-8:
                return None
            return W_snapped.tolist()
        except:
            return None

    def build_strassen_and_optimize(j_pairs, k_pairs):
        """Build Strassen decomposition and try to reduce rank."""
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            return None, None, None, rank
        
        # Try single term elimination
        U_list, V_list, W_list, rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination
        U_list, V_list, W_list, rank = try_pairwise_elimination(
            np.array(U_list), np.array(V_list), np.array(W_list), rank
        )
        
        return U_list, V_list, W_list, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(1,2),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(3,4),(2,5)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U_list, V_list, W_list, rank = build_strassen_and_optimize(j_pairs, k_pairs)
        
        if U_list is not None and rank < best_rank:
            best_rank = rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try linear algebra approach with structured patterns
    if time_left() > 30:
        np.random.seed(42)
        for attempt in range(30):
            if time_left() < 20:
                break
            
            for test_rank in range(max(40, best_rank - 3), best_rank):
                if time_left() < 15:
                    break
                
                U_test = np.zeros((n*m, test_rank))
                V_test = np.zeros((m*p, test_rank))
                
                for i in range(n):
                    for j in range(m):
                        cols = np.random.choice(test_rank, size=min(3, test_rank), replace=False)
                        for c in cols:
                            U_test[i*m+j, c] = np.random.choice([1, -1])
                
                for j in range(m):
                    for k in range(p):
                        cols = np.random.choice(test_rank, size=min(3, test_rank), replace=False)
                        for c in cols:
                            V_test[j*p+k, c] = np.random.choice([1, -1])
                
                W_test = solve_for_W(U_test, V_test, test_rank)
                if W_test is not None:
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, test_rank):
                        if test_rank < best_rank:
                            best_rank = test_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test}
                        break
    
    # Fallback to schoolbook
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
