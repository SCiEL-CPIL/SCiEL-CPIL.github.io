import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Also check that cross terms vanish
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i,j,k) != (i2,j2,k2):
                                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                                    # This is already checked above; the real check is:
                                    # For each term r, U[i,j,r]*V[j,k,r]*W[k,i,r] should contribute
                                    # only to the (i,j,k) entry
                                    pass
        # More careful: verify the full tensor decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal 1 if (i,j,k) is a valid triple, 0 otherwise
        # But for matrix mult, T[i,j,k] = 1 for all valid (i,j,k) since each entry is a single product
        # Wait - for matrix multiplication C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = A[i,j]*B[j,k] which gives C[i,k]
        # So the structure is: for each (i,j,k), the term is U[i,j,r]*V[j,k,r]*W[k,i,r]
        # and we need sum_r to give 1 for the "correct" contribution
        # Actually the decomposition is: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This should equal sum_j A[i,j]*B[j,k]
        # So for the decomposition to be exact, we need:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{j,j} for each (i,j,k)
        # i.e., it should be 1 for every (i,j,k) triple (since each A[i,j]*B[j,k] is a separate product)
        # Wait no. Let me re-read the task.
        # The tensor for matrix mult 2x5 * 5x6 has entries indexed by (i,j,k) where i in [0,2), j in [0,5), k in [0,6)
        # T[i,j,k] = A[i,j] * B[j,k]
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # And that's it! There are no "cross terms" to worry about because the tensor only has these 60 entries.
        return True

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

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        # Each 2x2x2 block: rank 7
        # Remaining j: standard 2*p terms per j
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
        """Try to eliminate redundant terms by checking if removing a term still gives valid decomposition."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        changed = True
        while changed and time_left() > 5:
            changed = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Check if term is redundant
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
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    changed = True
                    break  # Restart the loop since indices shifted
        
        return U_list, V_list, W_list, rank

    def solve_for_W(U, V, rank):
        """Given U and V, solve for W such that the decomposition is exact.
        For each (i,j,k), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        This is a linear system in W entries."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        # Number of unknowns: p*n*rank = 6*2*rank = 12*rank
        # Number of equations: n*m*p = 2*5*6 = 60
        num_unknowns = p * n * rank
        num_eqs = n * m * p
        
        if num_unknowns < num_eqs:
            return None  # Underdetermined in wrong direction
        
        # Build the linear system: A * w = b
        # where w is the vector of W entries
        # For each equation (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
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
        
        # Solve using least squares (or exact if square/overdetermined)
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if solution is exact
            if np.max(np.abs(A @ W_flat - b)) > 1e-8:
                return None
            # Check if entries are close to integers or half-integers
            W_arr = W_flat.reshape(p * n, rank)
            # Snap to nearest half-integer
            W_snapped = np.round(W_arr * 2) / 2
            if np.max(np.abs(A @ W_snapped.flatten() - b)) > 1e-8:
                return None
            return W_snapped.tolist()
        except:
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
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try linear algebra approach: fix structured U,V and solve for W
    if time_left() > 20:
        # Try various structured patterns for U and V with {0,1,-1} entries
        # Pattern 1: Each U[i*m+j, r] is 1 only for certain r
        # We'll try random structured patterns and solve for W
        
        for attempt in range(50):
            if time_left() < 15:
                break
            
            # Try different rank values below current best
            for test_rank in range(max(40, best_rank - 5), best_rank):
                if time_left() < 15:
                    break
                
                # Generate structured U: each row has ~2-3 nonzeros
                U_test = np.zeros((n*m, test_rank))
                V_test = np.zeros((m*p, test_rank))
                
                # For U: row i*m+j should have nonzeros in columns related to j
                # For V: row j*p+k should have nonzeros in columns related to j
                for i in range(n):
                    for j in range(m):
                        # Assign 2-3 random columns
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
                    # Verify
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, test_rank):
                        if test_rank < best_rank:
                            best_rank = test_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test}
                        break  # Found a solution at this rank, try lower
    
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
