"""
Rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.

Combines the clean schoolbook baseline with a structured search:
1. Build the rank-18 schoolbook decomposition.
2. Attempt to merge pairs of columns by solving a small linear system
   for the third factor's entries, checking the resulting rank-17 (etc.)
   decomposition for exactness.
3. Use a deadline guard to ensure we finish well within 120s.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3

    # Build schoolbook decomposition of rank n*m*p = 18
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1.0
                V_sb[v, col] = 1.0
                W_sb[w, col] = 1.0
                col += 1

    def validate(U_mat, V_mat, W_mat, rank_test):
        """Check that U,V,W of given rank exactly decompose the tensor."""
        # Check all valid (i,j,k) entries sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = sum(U_mat[u, c] * V_mat[v, c] * W_mat[w, c]
                              for c in range(rank_test))
                    if abs(val - 1.0) > 1e-9:
                        return False

        # Check all off-diagonal entries are 0
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    # If indices are consistent, skip (already checked)
                    if i_u == i_w and j_u == j_v and k_v == k_w:
                        continue
                    val = sum(U_mat[u, c] * V_mat[v, c] * W_mat[w, c]
                              for c in range(rank_test))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Convert to list-of-lists with half-integer support
    def to_list(fac):
        scaled = np.rint(fac * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    # Fallback: return schoolbook
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find a lower-rank decomposition by merging pairs of schoolbook columns
    # Strategy: for each pair (a, b) of schoolbook columns, try to find W entries
    # such that the combined column represents both terms exactly.
    
    # Each schoolbook column c corresponds to a triple (i_c, j_c, k_c) where:
    #   U has 1 at position (i_c * m + j_c)
    #   V has 1 at position (j_c * p + k_c)
    #   W has 1 at position (k_c * n + i_c)
    
    # When merging columns a and b into one column, we need:
    #   U_new[u] * V_new[v] * W_new[w] = U_a[u]*V_a[v]*W_a[w] + U_b[u]*V_b[v]*W_b[w]
    #   for all (u,v,w)
    
    # Since U_a and U_b have single 1s, the merged column's U, V entries are determined.
    # The constraint is on W_new.
    
    # Let's enumerate triples for each column
    triples = []
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
                col += 1
    
    # For each pair of distinct triples, check if they can be merged
    # Two triples (i1,j1,k1) and (i2,j2,k2) can be merged if:
    #   - They don't conflict in U: (i1*m+j1) != (i2*m+j2) OR if same, the U entries add
    #   - They don't conflict in V: (j1*p+k1) != (j2*p+k2) OR if same, the V entries add
    #   - The resulting W entries must be consistent
    
    # Actually, let's think more carefully. The merged column has:
    #   U_new[i*m+j] = U_a[i*m+j] + U_b[i*m+j]  (which is 0, 1, or 2)
    #   V_new[j*p+k] = V_a[j*p+k] + V_b[j*p+k]  (which is 0, 1, or 2)
    #   W_new[k*n+i] = ?  (to be determined)
    
    # The product U_new[u] * V_new[v] * W_new[w] must equal
    #   U_a[u]*V_a[v]*W_a[w] + U_b[u]*V_b[v]*W_b[w]
    
    # This is a system of linear equations in the W_new entries.
    
    # Let's try all pairs and see if a solution exists with half-integer entries.
    
    deadline = START_TIME + TIME_LIMIT
    found_lower = False
    
    for rank_test in range(rank_sb - 1, max(rank_sb - 5, 12), -1):
        if time.time() > deadline:
            break
        
        # Try to find a set of rank_test merged columns that cover all 18 terms
        # This is complex, so let's try a simpler approach:
        # Start with schoolbook and try to merge pairs one at a time.
        
        # For each pair (a, b), check if merging is possible
        # If yes, create a new decomposition with one fewer column
        
        # We'll do a greedy approach: repeatedly find a mergeable pair and merge it.
        
        U_cur = U_sb.copy()
        V_cur = V_sb.copy()
        W_cur = W_sb.copy()
        cur_rank = rank_sb
        active = list(range(rank_sb))
        
        merged_any = False
        while len(active) > 1 and time.time() < deadline:
            merged_this_round = False
            for idx_a in range(len(active)):
                if time.time() > deadline:
                    break
                for idx_b in range(idx_a + 1, len(active)):
                    a = active[idx_a]
                    b = active[idx_b]
                    
                    # Try to merge columns a and b
                    # Determine the U and V entries of the merged column
                    U_merge = U_cur[:, a] + U_cur[:, b]
                    V_merge = V_cur[:, a] + V_cur[:, b]
                    
                    # Check if U_merge and V_merge have at most one non-zero entry each
                    # (since each schoolbook column has exactly one 1 in U and one in V)
                    nz_U = np.nonzero(U_merge)[0]
                    nz_V = np.nonzero(V_merge)[0]
                    
                    if len(nz_U) > 2 or len(nz_V) > 2:
                        continue
                    
                    # The merged column's U and V are determined.
                    # Now solve for W_merge.
                    # For each (u, v, w), we need:
                    #   U_merge[u] * V_merge[v] * W_merge[w] = 
                    #     U_cur[u, a] * V_cur[v, a] * W_cur[w, a] + 
                    #     U_cur[u, b] * V_cur[v, b] * W_cur[w, b]
                    
                    # Let's collect the constraints on W_merge.
                    # Each constraint is of the form:
                    #   coeff * W_merge[w] = rhs
                    # where coeff = U_merge[u] * V_merge[v]
                    
                    w_constraints = {}
                    feasible = True
                    for u in range(n * m):
                        if U_merge[u] == 0:
                            continue
                        for v in range(m * p):
                            if V_merge[v] == 0:
                                continue
                            coeff = U_merge[u] * V_merge[v]
                            if coeff == 0:
                                continue
                            # For each w, the constraint is:
                            #   coeff * W_merge[w] = 
                            #     U_cur[u, a] * V_cur[v, a] * W_cur[w, a] + 
                            #     U_cur[u, b] * V_cur[v, b] * W_cur[w, b]
                            for w in range(p * n):
                                rhs = (U_cur[u, a] * V_cur[v, a] * W_cur[w, a] +
                                       U_cur[u, b] * V_cur[v, b] * W_cur[w, b])
                                if w in w_constraints:
                                    # Check consistency
                                    if abs(w_constraints[w] - rhs / coeff) > 1e-9:
                                        feasible = False
                                        break
                                else:
                                    w_constraints[w] = rhs / coeff
                            if not feasible:
                                break
                        if not feasible:
                            break
                    
                    if not feasible:
                        continue
                    
                    # Check if all W_merge entries are half-integers
                    w_merge = np.zeros(p * n)
                    for w, val in w_constraints.items():
                        w_merge[w] = val
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        continue
                    
                    # Also check that entries not in w_constraints are 0
                    for w in range(p * n):
                        if w not in w_constraints:
                            # Verify that the constraint is satisfied with W_merge[w] = 0
                            # i.e., U_cur[u, a] * V_cur[v, a] * 0 + U_cur[u, b] * V_cur[v, b] * 0 = 0
                            # This is automatically satisfied since W_cur entries are 0 or 1
                            # and the product must be 0 when coeff > 0
                            # Actually, we need to verify that for all (u,v) with coeff > 0,
                            # the RHS is 0 when w is not in w_constraints
                            pass  # This should be fine given the construction
                    
                    # If feasible, perform the merge
                    U_cur[:, a] = U_merge
                    V_cur[:, a] = V_merge
                    W_cur[:, a] = w_merge
                    # Remove column b
                    active.pop(idx_b)
                    merged_this_round = True
                    merged_any = True
                    break
                if merged_this_round:
                    break
            
            if not merged_this_round:
                break
        
        if merged_any:
            new_rank = len(active)
            if new_rank < best_rank:
                # Extract the active columns
                U_new = U_cur[:, active]
                V_new = V_cur[:, active]
                W_new = W_cur[:, active]
                
                # Validate
                if validate(U_new, V_new, W_new, new_rank):
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found_lower = True
                    break  # Try to find even lower rank
        
        if found_lower:
            break
    
    # Return the best result found
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
