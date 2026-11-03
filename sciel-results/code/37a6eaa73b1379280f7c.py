import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

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
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_single_term_removal(U, V, W, rank, term_idx):
        """Check if removing term_idx still gives valid decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each tensor entry, check if the sum without term_idx still equals 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute dot product excluding term_idx
                    s = 0.0
                    for r in range(rank):
                        if r != term_idx:
                            s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
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
        """Try to eliminate redundant terms using fast single-term check."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        changed = True
        while changed and time_left() > 5:
            changed = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                if verify_single_term_removal(U_arr, V_arr, W_arr, rank, term_idx):
                    # Remove term_idx from all arrays
                    U_arr = np.delete(U_arr, term_idx, axis=1)
                    V_arr = np.delete(V_arr, term_idx, axis=1)
                    W_arr = np.delete(W_arr, term_idx, axis=1)
                    rank -= 1
                    changed = True
                    break
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

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

    def structured_random_search(U_base, V_base, W_base, base_rank, target_rank, num_trials=200):
        """
        Structured random search: try random sparse patterns and solve for W.
        """
        best_result = None
        best_r = base_rank
        
        for R in range(min(target_rank, base_rank - 1), min(target_rank, base_rank - 1) + 1):
            if time_left() < 10:
                break
            
            for attempt in range(100):
                if time_left() < 5:
                    break
                
                # Initialize with random sparse patterns
                U_test = np.zeros((n*m, R))
                V_test = np.zeros((m*p, R))
                
                # For U: each row should have 1-3 nonzeros
                for i in range(n):
                    for j in range(m):
                        num_nz = np.random.randint(1, 4)
                        cols = np.random.choice(R, size=num_nz, replace=False)
                        for c in cols:
                            U_test[i*m+j, c] = np.random.choice([1, -1])
                
                for j in range(m):
                    for k in range(p):
                        num_nz = np.random.randint(1, 4)
                        cols = np.random.choice(R, size=num_nz, replace=False)
                        for c in cols:
                            V_test[j*p+k, c] = np.random.choice([1, -1])
                
                W_test = solve_for_W(U_test, V_test, R)
                if W_test is not None:
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, R):
                        return U_test.tolist(), V_test.tolist(), W_test, R
        
        return None

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
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),  # duplicate but safe
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms (now faster)
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try structured random search to find lower rank
    if time_left() > 20 and best_rank > 50:
        # Use the best Strassen decomposition as a starting point
        # and try to find a lower rank decomposition
        for j_pairs, k_pairs in configurations[:5]:
            if time_left() < 10:
                break
            U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
            
            # Try to find a decomposition with rank < current best
            target_rank = max(40, best_rank - 3)
            result = structured_random_search(U.tolist(), V.tolist(), W.tolist(), rank, target_rank)
            if result is not None:
                U_r, V_r, W_r, r = result
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
    
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
