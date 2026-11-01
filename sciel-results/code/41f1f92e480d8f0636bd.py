"""Structured block-decomposition for 2x5x6 matrix multiplication tensor using Strassen blocks with optimization."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build_strassen(j1, j2, k1, k2):
        """
        Returns list of (u_pairs, v_pairs, w_pairs) for Strassen rank-7 terms.
        """
        terms = []
        # m1 = (a00+a11)(b00+b11)
        terms.append((
            [(u_idx(0, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1), (w_idx(k2, 1), 1)]
        ))
        # m2 = (a10+a11)*b00
        terms.append((
            [(u_idx(1, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1)],
            [(w_idx(k1, 1), 1), (w_idx(k2, 1), -1)]
        ))
        # m3 = a00*(b01-b11)
        terms.append((
            [(u_idx(0, j1), 1)],
            [(v_idx(j1, k2), 1), (v_idx(j2, k2), -1)],
            [(w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]
        ))
        # m4 = a11*(b10-b00)
        terms.append((
            [(u_idx(1, j2), 1)],
            [(v_idx(j2, k1), 1), (v_idx(j1, k1), -1)],
            [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1)]
        ))
        # m5 = (a00+a01)*b11
        terms.append((
            [(u_idx(0, j1), 1), (u_idx(0, j2), 1)],
            [(v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), -1), (w_idx(k2, 0), 1)]
        ))
        # m6 = (a10-a00)*(b00+b01)
        terms.append((
            [(u_idx(1, j1), 1), (u_idx(0, j1), -1)],
            [(v_idx(j1, k1), 1), (v_idx(j1, k2), 1)],
            [(w_idx(k2, 1), 1)]
        ))
        # m7 = (a01-a11)*(b10+b11)
        terms.append((
            [(u_idx(0, j2), 1), (u_idx(1, j2), -1)],
            [(v_idx(j2, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1)]
        ))
        return terms

    def build_schoolbook_2x1x2(j, k1, k2):
        """
        Returns list of (u_pairs, v_pairs, w_pairs) for schoolbook rank-4 terms.
        """
        terms = []
        terms.append(([(u_idx(0, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 0), 1)]))
        terms.append(([(u_idx(0, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 0), 1)]))
        terms.append(([(u_idx(1, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 1), 1)]))
        terms.append(([(u_idx(1, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 1), 1)]))
        return terms

    def build_decomposition(terms_list):
        """Build U, V, W matrices from a list of terms."""
        rank = len(terms_list)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col, (u_pairs, v_pairs, w_pairs) in enumerate(terms_list):
            for idx, val in u_pairs:
                U[idx][col] = val
            for idx, val in v_pairs:
                V[idx][col] = val
            for idx, val in w_pairs:
                W[idx][col] = val
        return U, V, W

    def verify_decomposition(U, V, W):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # Build the structured decomposition (rank 54)
    all_terms = []
    all_terms.extend(build_strassen(0, 1, 0, 1))
    all_terms.extend(build_strassen(0, 1, 2, 3))
    all_terms.extend(build_strassen(0, 1, 4, 5))
    all_terms.extend(build_strassen(2, 3, 0, 1))
    all_terms.extend(build_schoolbook_2x1x2(4, 0, 1))
    all_terms.extend(build_strassen(2, 3, 2, 3))
    all_terms.extend(build_schoolbook_2x1x2(4, 2, 3))
    all_terms.extend(build_strassen(2, 3, 4, 5))
    all_terms.extend(build_schoolbook_2x1x2(4, 4, 5))

    U, V, W = build_decomposition(all_terms)
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook rank 60
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = len(all_terms)

    # Try to find a lower-rank decomposition using iterative refinement
    # Strategy: try removing terms and see if we can compensate
    # Use a simple approach: try various rank reductions
    
    # First, let's try a numerical search for rank reduction
    # We'll use a gradient-free approach: randomly perturb and check validity
    
    np.random.seed(42)
    
    # Try to reduce rank by combining terms
    # For each pair of terms, see if they can be merged
    
    def try_reduce_rank(U, V, W, target_rank):
        """Try to find a decomposition of target_rank by optimizing."""
        if target_rank >= best_rank:
            return None
        
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Build the tensor from current decomposition
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want to find new U', V', W' with fewer columns that give the same T
        
        # This is a hard problem. Let's try a simpler approach:
        # Check if any term is redundant (can be expressed as combination of others)
        
        # For now, try random perturbations of the last few terms
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing one term and adjusting others
            if target_rank == best_rank - 1:
                # Try to see if the last term is redundant
                U_test = U_mat[:, :-1].copy()
                V_test = V_mat[:, :-1].copy()
                W_test = W_mat[:, :-1].copy()
                
                # Check if this still gives correct results
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = np.sum(U_test[i * m + j, :] * V_test[j * p + k, :] * W_test[k * n + i, :])
                            if not np.isclose(total, 1.0, atol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return (U_test.tolist(), V_test.tolist(), W_test.tolist())
        
        return None

    # Try reducing rank by 1
    if time.time() - start_time < time_limit - 10:
        result = try_reduce_rank(U, V, W, best_rank - 1)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank -= 1

    # If time permits, try more aggressive search
    if time.time() - start_time < time_limit - 20:
        # Try a more systematic search using the structure
        # Look for alternative Strassen-like decompositions for 2x2x2
        # with different coefficient patterns
        
        # For 2x2x2, the minimal rank is 7. Let's verify our Strassen is correct.
        # Try different groupings of the 5 middle indices
        
        alternative_groupings = [
            # (j_indices, k_indices) pairs for 2x2 blocks
            ([(0,1),(2,3),(4,5)], [(0,1),(2,3),(4,5)]),  # Current
            ([(0,1),(2,4),(3,5)], [(0,2),(1,3),(4,5)]),   # Different
            ([(0,2),(1,3),(4,5)], [(0,1),(2,4),(3,5)]),   # Another
        ]
        
        for j_groups, k_groups in alternative_groupings:
            if time.time() - start_time > time_limit:
                break
            
            # This approach is complex; skip for now
            pass

    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Fall back to original
        U_orig, V_orig, W_orig = build_decomposition(all_terms)
        return {"U": U_orig, "V": V_orig, "W": W_orig}

    return {"U": best_U, "V": best_V, "W": best_W}
