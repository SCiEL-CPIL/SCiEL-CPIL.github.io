import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check non-zeros: T[i,j,k] should be 1 if i,j,k are consistent (which they are by construction of the tensor)
        # The tensor for matrix multiplication is T[i,j,k] = 1 if the indices form a valid path?
        # No, the tensor is T[i,j,k] = 1 if the multiplication is defined?
        # Actually, the tensor M_{i,j,k} is 1 if the indices correspond to the multiplication term A_{i,j} B_{j,k} -> C_{i,k}.
        # In the flattened indices:
        # A index: i*m + j
        # B index: j*p + k
        # C index: k*n + i (trace convention W[k*n+i, r])
        # The tensor component T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
        # Otherwise 0.
        
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        # Iterate over all possible (a,b,c) and check if they are NOT a valid (i,j,k) triple
        # There are nm*mp*pn = 8*24*12 = 2304 combinations.
        # Valid ones are n*m*p = 48.
        # So we check 2304 - 48 = 2256 zeros.
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    # 1. Construct Rank 42 (Strassen blocks)
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # A indices
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        # B indices
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        # W indices
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        # Term 1: M2 = (A10+A11)B00
        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        # Term 2: M3 = A00(B01-B11)
        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        # Term 3: M4 = A11(B10-B00)
        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        # Term 4: M5 = (A00+A01)B11
        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        # Term 5: M6 = (A10-A00)(B00+B01)
        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        # Term 6: M7 = (A01-A11)(B10+B11)
        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if verify(U42, V42, W42):
        # Attempt to reduce rank from 42
        # Strategy: Randomized Linear Dependency Check
        # We have a set of 42 vectors in the tensor space.
        # If the rank is R < 42, then there exist linear dependencies.
        # We can try to find a subset of size R that spans the same tensor.
        # Or simply check if we can remove columns one by one while maintaining validity?
        # No, removing a column changes the tensor unless we adjust others.
        
        # Better: Use the fact that we can linearly combine columns.
        # If column c is a linear combination of other columns in a way that preserves the tensor?
        # Actually, the tensor is T = sum_r u_r v_r w_r.
        # If we can find coefficients alpha_r such that sum alpha_r u_r v_r w_r = T and fewer non-zero alphas?
        # This is NP-hard in general.
        
        # Heuristic: Check for duplicate or proportional columns.
        # In Strassen, are there proportional columns?
        # Within a block, the 7 terms are linearly independent (rank 7 for 2x2x2).
        # Across blocks, the supports are disjoint in A, B, C indices?
        # Let's check supports.
        # Block (i_start, j_start, k_start) affects:
        # U: rows in [i_start, i_start+1] x [j_start, j_start+1]
        # V: rows in [j_start, j_start+1] x [k_start, k_start+1]
        # W: rows in [k_start, k_start+1] x [i_start, i_start+1]
        
        # Since i_start is always 0 (n=2), the U rows are 0,1,2,3 for j_start=0 and 4,5,6,7 for j_start=1?
        # Wait, n=2. i_start is 0.
        # j_start is 0, 1.
        # k_start is 0, 1, 2.
        
        # Block (0,0,0):
        # U rows: 0,1,2,3 (i=0,1; j=0,1) -> indices 0,1,2,3
        # V rows: 0,1,2,3 (j=0,1; k=0,1) -> indices 0,1,2,3
        # W rows: 0,1,2,3 (k=0,1; i=0,1) -> indices 0,1,2,3
        
        # Block (0,0,1):
        # U rows: 0,1,2,3 (same A block)
        # V rows: 2,3,4,5 (j=0,1; k=2,3) -> indices 2,3,4,5
        # W rows: 4,5,6,7 (k=2,3; i=0,1) -> indices 4,5,6,7
        
        # Notice that U rows for block (0,0,0) and (0,0,1) are the SAME.
        # This means we might be able to combine terms across k_start blocks for the same (i_start, j_start).
        
        # Let's try a specific optimization:
        # For a fixed (i_start, j_start), we have 3 blocks (k_start=0,1,2).
        # Each uses 7 terms. Total 21 terms for the 2x2x6 sub-problem.
        # Can we do 2x2x6 in rank < 21?
        # 2x2x2 is 7. 2x2x4 is 14? 2x2x6 is 21?
        # If we use Strassen for 2x2x2, we get 7 per 2x2x2 block.
        # Is there a faster 2x2x6?
        # Known rank for 2x2x6 is 15? No, 2x2xn is 7*ceil(n/2) + ...?
        # Actually, for 2x2xn, the rank is known to be 7*ceil(n/2) for small n?
        # Let's check if we can merge the 3 blocks of 2x2x2 into a single 2x2x6 of rank < 21.
        
        # If we can find a rank 15 decomposition for 2x2x6, then total rank would be 2 * 15 = 30.
        # Or if we can find rank 14, total 28.
        
        # Let's try to construct a 2x2x6 decomposition using random search or known algorithms.
        # Given time constraints, we will attempt a random linear algebra approach to reduce the 42 columns.
        
        # We will try to find a basis of the tensor space spanned by the 42 terms.
        # Flatten the tensor? No, the tensor is fixed. We want to represent the SAME tensor with fewer terms.
        # This is equivalent to finding a rank decomposition of the tensor.
        
        # We can try to remove columns by solving for dependencies.
        # Let T be the target tensor.
        # We have T = sum_{r=0}^{41} u_r v_r w_r.
        # We want T = sum_{r=0}^{R-1} u'_r v'_r w'_r.
        
        # Heuristic: Try to remove one column at a time.
        # If we remove column c, can we adjust the remaining columns to still sum to T?
        # This requires that the vector (u_c, v_c, w_c) is in the span of the others in the tensor product space.
        
        # A simpler check: If two columns have the same U and V, but different W, can we merge?
        # If u_c = u_d and v_c = v_d, then u_c v_c w_c + u_c v_c w_d = u_c v_c (w_c + w_d).
        # So we can merge them into one column with W = w_c + w_d.
        
        # Let's check for such merges in the Strassen construction.
        
        # Group columns by (U, V) signature.
        # Since U and V are sparse, we can use their non-zero patterns and values.
        
        best_U, best_V, best_W = U42, V42, W42
        best_rank = 42
        
        # Attempt merging
        changed = True
        while changed:
            changed = False
            current_rank = best_U.shape[1]
            if current_rank <= 24:
                break
            
            # Find pairs (c, d) with c < d such that U[:,c] == U[:,d] and V[:,c] == V[:,d]
            # Note: They must be exactly equal, not just proportional, unless we scale.
            # If U[:,c] = alpha U[:,d] and V[:,c] = beta V[:,d], then
            # u_c v_c = alpha beta u_d v_d.
            # So we can merge them if we scale appropriately.
            
            # Let's just look for exact matches first.
            merged = False
            new_U = np.zeros((nm, current_rank - 1), dtype=float)
            new_V = np.zeros((mp, current_rank - 1), dtype=float)
            new_W = np.zeros((pn, current_rank - 1), dtype=float)
            
            # This is O(R^2) which is 42^2 = 1764. Fast enough.
            # But we need to be careful with indices.
            
            # Let's try a different approach: 
            # Iterate through columns, and if we find a duplicate (up to scalar) of a previous column,
            # merge it into the previous one and skip adding it.
            
            # However, "up to scalar" for U and V must be consistent.
            # If U_c = s U_d and V_c = t V_d, then the product term is s t U_d V_d W_c.
            # We can absorb s t into W.
            # So we need U_c proportional to U_d AND V_c proportional to V_d.
            
            cols_to_keep = []
            used = [False] * current_rank
            
            for i in range(current_rank):
                if used[i]:
                    continue
                
                # Try to merge all other unused columns into i if possible
                # Or just find one merge at a time to be safe.
                
                # Find j > i such that U_i ~ U_j and V_i ~ V_j
                merge_found = False
                for j in range(i + 1, current_rank):
                    if used[j]:
                        continue
                    
                    # Check proportionality
                    # U
                    nz_u_i = np.nonzero(best_U[:, i])[0]
                    nz_u_j = np.nonzero(best_U[:, j])[0]
                    if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                        if len(nz_u_i) != len(nz_u_j):
                            continue
                        if len(nz_u_i) > 0:
                            continue # one is zero, other not
                    else:
                        if not np.array_equal(nz_u_i, nz_u_j):
                            continue
                        # Check ratio
                        ratios_u = best_U[nz_u_i, i] / best_U[nz_u_i, j]
                        if np.std(ratios_u) > 1e-10:
                            continue
                        s = ratios_u[0]
                    
                    # V
                    nz_v_i = np.nonzero(best_V[:, i])[0]
                    nz_v_j = np.nonzero(best_V[:, j])[0]
                    if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                        if len(nz_v_i) != len(nz_v_j):
                            continue
                        if len(nz_v_i) > 0:
                            continue
                    else:
                        if not np.array_equal(nz_v_i, nz_v_j):
                            continue
                        ratios_v = best_V[nz_v_i, i] / best_V[nz_v_i, j]
                        if np.std(ratios_v) > 1e-10:
                            continue
                        t = ratios_v[0]
                    
                    # If we get here, U_i = s U_j and V_i = t V_j
                    # So U_j V_j = (1/s) (1/t) U_i V_i
                    # Term j is U_j V_j W_j = (1/(st)) U_i V_i W_j
                    # We can add (1/(st)) W_j to W_i.
                    
                    scale = 1.0 / (s * t)
                    best_W[:, i] += scale * best_W[:, j]
                    used[j] = True
                    merge_found = True
                    changed = True
                    break # Restart the loop to rebuild arrays? Or just mark used?
                    # Marking used is fine if we rebuild at the end.
                
                if not used[i]:
                    cols_to_keep.append(i)
            
            if merge_found:
                # Rebuild arrays
                new_rank = len(cols_to_keep)
                best_U = best_U[:, cols_to_keep]
                best_V = best_V[:, cols_to_keep]
                best_W = best_W[:, cols_to_keep]
                
                # Verify
                if not verify(best_U, best_V, best_W):
                    # Revert? This shouldn't happen if math is correct.
                    # But let's be safe.
                    best_U, best_V, best_W = U42, V42, W42
                    break
                else:
                    best_rank = new_rank
                    print(f"Merged to rank {best_rank}", file=sys.stderr)
            else:
                break
        
        if best_rank < 42 and verify(best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

        # If merging didn't help, try random search for lower rank?
        # Given the structure, it's unlikely to find rank < 42 easily without specific knowledge.
        # But let's try a simple random restart with rank 30?
        # No, that's too risky and slow.
        
        # Return the best found (likely 42)
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
