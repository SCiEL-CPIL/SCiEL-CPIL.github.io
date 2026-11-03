import time
import numpy as np
from itertools import product, combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_fast(U, V, W, rank, tol=1e-9):
        """Verify decomposition using fully vectorized numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorize all (i,j,k) checks at once
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # For each t: sum over t of U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Reshape to get all combinations
        # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
        # We need to sum over t for each (i,j,k)
        
        # Create index arrays
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_rows = U_arr[i_idx*m + j_idx, :]  # (n*m*p, rank)
        v_rows = V_arr[j_idx*p + k_idx, :]  # (n*m*p, rank)
        w_rows = W_arr[k_idx*n + i_idx, :]  # (n*m*p, rank)
        
        # Sum over rank dimension
        sums = np.sum(u_rows * v_rows * w_rows, axis=1)
        
        return np.all(np.abs(sums - 1.0) <= tol)

    def verify_batch(U, V, W, rank, indices_to_check, tol=1e-9):
        """Check only specific (i,j,k) indices for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i, j, k in indices_to_check:
            s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
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

    def try_eliminate_terms_fast(U, V, W, rank):
        """Try to eliminate redundant terms using fast verification."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(5):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 3:
                    break
                
                # Create modified arrays (only zero out the column)
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
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination_fast(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 5:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 5:
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
                    return try_pairwise_elimination_fast(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    # Generate all possible pairings
    def generate_pairings(indices):
        """Generate all ways to pair up indices."""
        if not indices:
            return [[]]
        if len(indices) % 2 != 0:
            return [[]]
        first = indices[0]
        rest = indices[1:]
        results = []
        for i, second in enumerate(rest):
            remaining = rest[:i] + rest[i+1:]
            for sub in generate_pairings(remaining):
                results.append([(first, second)] + sub)
        return results

    # Try different configurations
    j_indices = list(range(m))
    k_indices = list(range(p))
    
    # All possible pairings for j (5 elements - one left over)
    # We need to choose which 4 j's to pair, leaving 1
    j_pair_configs = []
    for leftover in range(m):
        paired = [j for j in range(m) if j != leftover]
        for pairing in generate_pairings(paired):
            j_pair_configs.append((pairing, [leftover]))
    
    # All possible pairings for k (6 elements)
    k_pair_configs = []
    for pairing in generate_pairings(list(range(p))):
        k_pair_configs.append(pairing)
    
    best_result = None
    best_rank = float('inf')
    
    # First try the known good config
    known_configs = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,2),(1,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(2,5),(3,4)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,5),(3,4)]),
        ([(0,1),(2,3)], [(0,3),(1,5),(2,4)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
        ([(0,1),(2,3)], [(0,5),(1,2),(3,4)]),
        ([(0,1),(2,3)], [(0,5),(1,4),(2,3)]),
    ]
    
    tried = set()
    
    for j_pairs, k_pairs in known_configs:
        if time_left() < 8:
            break
        key = (tuple(sorted(j_pairs)), tuple(sorted(k_pairs)))
        if key in tried:
            continue
        tried.add(key)
        
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_fast(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms_fast(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 15:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination_fast(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # If time permits, try more systematic search
    if time_left() > 30 and best_rank > 50:
        for j_pairing, leftover_j in j_pair_configs:
            if time_left() < 10:
                break
            for k_pairing in k_pair_configs:
                if time_left() < 5:
                    break
                key = (tuple(sorted(j_pairing)), tuple(sorted(k_pairing)))
                if key in tried:
                    continue
                tried.add(key)
                
                U, V, W, rank = build_strassen_decomp(j_pairing, k_pairing)
                
                if not verify_fast(U.tolist(), V.tolist(), W.tolist(), rank):
                    continue
                
                U_list, V_list, W_list, new_rank = try_eliminate_terms_fast(U, V, W, rank)
                
                if time_left() > 10:
                    U_list, V_list, W_list, new_rank = try_pairwise_elimination_fast(U_list, V_list, W_list, new_rank)
                    
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
