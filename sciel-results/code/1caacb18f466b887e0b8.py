"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 140.0

    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template (rank 7).
    # w_offsets ordered as [X00, X01, X10, X11] -> (k, i)
    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    def verify_fast(Ul, Vl, Wl, R):
        """Verify using vectorized tensor contraction."""
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U3 = Ua.reshape(n, m, R)
        V3 = Va.reshape(m, p, R)
        W3 = Wa.reshape(p, n, R)
        # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: T[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-7)

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if not verify_fast(base_U, base_V, base_W, 36):
        # Fallback: return the base 36-rank solution
        return {"U": base_U, "V": base_V, "W": base_W}

    # Precompute numpy arrays for fast verification
    base_Ua = np.array(base_U, dtype=float)
    base_Va = np.array(base_V, dtype=float)
    base_Wa = np.array(base_W, dtype=float)

    def apply_merges(merge_list):
        """Apply a list of (c_target, c_source, sign) merges and verify.
        Returns (rank, U, V, W) if valid, else None.
        merge_list: list of (c1, c2, sgn) meaning add sgn*col_c2 to col_c1, delete col_c2
        """
        if time.time() > deadline - 20:
            return None
        # Build new matrices by applying all merges
        R = 36 - len(merge_list)
        if R < 1:
            return None
        # Track which original columns are deleted
        deleted = set()
        new_U = np.zeros((n*m, R))
        new_V = np.zeros((m*p, R))
        new_W = np.zeros((p*n, R))
        # Map old col -> new col
        col_map = {}
        new_idx = 0
        for c in range(36):
            if c not in deleted:
                col_map[c] = new_idx
                new_idx += 1
        
        # Apply merges in order
        # First, build the combined columns
        # For each merge (c1, c2, sgn): new_col[c1] += sgn * old_col[c2], then delete c2
        # We need to handle sequential merges carefully
        # Simpler: start with all 36 columns, apply each merge, track which are alive
        
        alive = list(range(36))
        col_data_U = [base_Ua[:, c].copy() for c in range(36)]
        col_data_V = [base_Va[:, c].copy() for c in range(36)]
        col_data_W = [base_Wa[:, c].copy() for c in range(36)]
        
        for (c1, c2, sgn) in merge_list:
            if c2 not in alive or c1 not in alive:
                return None
            # Add sgn * col_data[c2] to col_data[c1]
            col_data_U[c1] += sgn * col_data_U[c2]
            col_data_V[c1] += sgn * col_data_V[c2]
            col_data_W[c1] += sgn * col_data_W[c2]
            # Remove c2 from alive
            alive.remove(c2)
        
        # Build final arrays
        R = len(alive)
        new_U = np.zeros((n*m, R))
        new_V = np.zeros((m*p, R))
        new_W = np.zeros((p*n, R))
        for new_i, old_c in enumerate(alive):
            new_U[:, new_i] = col_data_U[old_c]
            new_V[:, new_i] = col_data_V[old_c]
            new_W[:, new_i] = col_data_W[old_c]
        
        # Verify
        if not verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), R):
            return None
        return (R, new_U.tolist(), new_V.tolist(), new_W.tolist())

    # Compatible merge pairs: (c1, c2) where c1 is from AL-block and c2 is from AR-block
    # or vice versa, sharing the same (B_block, C_block)
    # Block structure:
    #   Block 0: cols 0-6,  (AL, B11, C11)
    #   Block 1: cols 7-13, (AR, B21, C11)  -- same C as block 0, different B!
    #   Wait, let me re-examine.
    #   Block 0: strassen_block(AL, B11, C11) -> A rows {0,1}, B cols {0,1}, C rows {0,1}
    #   Block 1: strassen_block(AR, B21, C11) -> A rows {0,1} (but cols 2,3), B rows {2,3} cols {0,1}, C rows {0,1}
    #   Block 2: strassen_block(AL, B12, C12) -> A rows {0,1} cols {0,1}, B rows {0,1} cols {2,3}, C rows {2,3}
    #   Block 3: strassen_block(AR, B22, C12) -> A rows {0,1} cols {2,3}, B rows {2,3} cols {2,3}, C rows {2,3}
    #
    # For a merge to be valid, the two columns must contribute to the same set of (i,j,k) tensor entries
    # with the same sign pattern, so that adding one to the other cancels out.
    # 
    # Actually, the merge works when: for all (i,j,k), the product U[c1][i,j]*V[c1][j,k]*W[c1][k,i]
    # equals -sgn * U[c2][i,j]*V[c2][j,k]*W[c2][k,i] summed over the merge.
    # This is hard to check a priori. Let's just try all reasonable pairs.
    #
    # The original code tried pairs where one column is from AL-block and the other from AR-block
    # at the same Strassen position. Let's expand to try all pairs within the same output C-block.
    
    # Pairs: (c_from_block_A, c_from_block_B) where they share C-block
    # C-block 0 (C11): blocks 0 and 1 -> cols 0-6 and 7-13
    # C-block 1 (C12): blocks 2 and 3 -> cols 14-20 and 21-27
    
    pairs_c0 = [(i, i+7) for i in range(7)]  # (0,7), (1,8), ..., (6,13)
    pairs_c1 = [(i, i+7) for i in range(14, 21)]  # (14,21), (15,22), ..., (20,27)
    all_pairs = pairs_c0 + pairs_c1
    
    # Also try reverse direction
    all_pairs_rev = [(c2, c1) for (c1, c2) in all_pairs]
    
    # Try single merges first
    found_35 = None
    for (c1, c2) in all_pairs + all_pairs_rev:
        if time.time() > deadline - 30:
            break
        for sgn in (1.0, -1.0):
            result = apply_merges([(c1, c2, sgn)])
            if result is not None:
                found_35 = result
                break
        if found_35:
            break
    
    if found_35:
        best = found_35
        R35, U35, V35, W35 = found_35
    
    # Now try to find rank 34 by applying two merges
    if best[0] == 35 and time.time() < deadline - 40:
        # Use the rank-35 solution as base
        U35a = np.array(U35, dtype=float)
        V35a = np.array(V35, dtype=float)
        W35a = np.array(W35, dtype=float)
        
        def apply_merges_35(merge_list, base_Ua, base_Va, base_Wa, total_cols):
            """Apply merges to a 35-col base."""
            if time.time() > deadline - 20:
                return None
            R = total_cols - len(merge_list)
            if R < 1:
                return None
            alive = list(range(total_cols))
            col_data_U = [base_Ua[:, c].copy() for c in range(total_cols)]
            col_data_V = [base_Va[:, c].copy() for c in range(total_cols)]
            col_data_W = [base_Wa[:, c].copy() for c in range(total_cols)]
            
            for (c1, c2, sgn) in merge_list:
                if c2 not in alive or c1 not in alive:
                    return None
                col_data_U[c1] += sgn * col_data_U[c2]
                col_data_V[c1] += sgn * col_data_V[c2]
                col_data_W[c1] += sgn * col_data_W[c2]
                alive.remove(c2)
            
            R = len(alive)
            new_U = np.zeros((n*m, R))
            new_V = np.zeros((m*p, R))
            new_W = np.zeros((p*n, R))
            for new_i, old_c in enumerate(alive):
                new_U[:, new_i] = col_data_U[old_c]
                new_V[:, new_i] = col_data_V[old_c]
                new_W[:, new_i] = col_data_W[old_c]
            
            if not verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), R):
                return None
            return (R, new_U.tolist(), new_V.tolist(), new_W.tolist())
        
        # Try pairs of merges on the 35-col solution
        # We need to figure out which columns in the 35-col solution correspond to which original columns
        # This is complex; instead, try all pairs of the original 14 compatible pairs
        # and see if applying two of them simultaneously works on the 36-col base
        
        # Try all pairs of compatible merges from the 36-col base
        # Limit to pairs that don't conflict (don't share a column)
        compatible_merge_pairs = []
        for idx1 in range(len(all_pairs)):
            for idx2 in range(idx1 + 1, len(all_pairs)):
                c1a, c2a = all_pairs[idx1]
                c1b, c2b = all_pairs[idx2]
                # Check no column overlap
                if len({c1a, c2a, c1b, c2b}) == 4:
                    compatible_merge_pairs.append(((c1a, c2a), (c1b, c2b)))
        
        # Also try reverse directions
        for (pa, pb) in compatible_merge_pairs[:200]:  # Limit to 200 to stay within time
            if time.time() > deadline - 25:
                break
            for sgn_a in (1.0, -1.0):
                for sgn_b in (1.0, -1.0):
                    result = apply_merges([(pa[0], pa[1], sgn_a), (pb[0], pb[1], sgn_b)])
                    if result is not None:
                        best = result
                        break
                if best[0] < 35:
                    break
            if best[0] < 35:
                break
        
        # Also try with reversed pairs
        if best[0] >= 35:
            for (pa, pb) in compatible_merge_pairs[:200]:
                if time.time() > deadline - 25:
                    break
                pa_rev = (pa[1], pa[0])
                pb_rev = (pb[1], pb[0])
                for sgn_a in (1.0, -1.0):
                    for sgn_b in (1.0, -1.0):
                        result = apply_merges([(pa_rev[0], pa_rev[1], sgn_a), (pb_rev[0], pb_rev[1], sgn_b)])
                        if result is not None:
                            best = result
                            break
                    if best[0] < 35:
                        break
                if best[0] < 35:
                    break

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
