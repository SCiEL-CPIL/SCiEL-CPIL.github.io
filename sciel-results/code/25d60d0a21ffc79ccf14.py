"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses Strassen-based block decomposition with proper rank reduction via
    identifying and merging terms with identical V and W patterns.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    def strassen_based_decomposition():
        """
        Build a decomposition using Strassen on 2x2x2 blocks, then merge
        terms that share identical V and W column patterns.
        """
        # Strassen 2x2x2 terms: each term has U, V, W contributions
        # Local indices 0-3 for 2x2 matrices: index = row*2 + col
        strassen_terms = [
            # term 0: M1 = (A11+A22)(B11+B22)
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            # term 1: M2 = (A21+A22)B11
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
            # term 2: M3 = A11(B21-B22)
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
            # term 4: M4 = A22(B11-B21)
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
            # term 5: M5 = (A11+A12)B22
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
            # term 6: M6 = (A21-A11)(B11+B12)
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            # term 7: M7 = (A12-A22)(B21+B22)
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
        ]
        
        # For each (r_block, sub, s_idx), generate the global U, V, W vectors
        # r_block: 0,1,2 (3 blocks of 2 columns in output)
        # sub: 0,1 (2 blocks of 2 columns in A, 2 rows in B)
        terms = []
        for r_block in range(3):
            for sub in range(2):
                a_col_offset = sub * 2  # offset in A's columns (and B's rows)
                b_col_offset = r_block * 2  # offset in B's columns (and output's columns)
                
                for s_idx in range(7):
                    term = strassen_terms[s_idx]
                    
                    # Build U vector (length n*m = 8)
                    U_vec = [0.0] * (n * m)
                    for local_u, coeff in term['U']:
                        a = local_u // 2  # local row in 2x2
                        l = local_u % 2   # local col in 2x2
                        global_row = a  # global row in A (0 or 1)
                        global_col = a_col_offset + l
                        global_u = global_row * m + global_col
                        U_vec[global_u] += coeff
                    
                    # Build V vector (length m*p = 24)
                    V_vec = [0.0] * (m * p)
                    for local_v, coeff in term['V']:
                        l = local_v // 2   # local row in 2x2 (B's row)
                        c = local_v % 2    # local col in 2x2 (B's col)
                        global_row = a_col_offset + l  # B's row
                        global_col = b_col_offset + c  # B's col
                        global_v = global_row * p + global_col
                        V_vec[global_v] += coeff
                    
                    # Build W vector (length p*n = 12)
                    W_vec = [0.0] * (p * n)
                    for local_w, coeff in term['W']:
                        a = local_w // 2   # local row in 2x2 (output's row)
                        c = local_w % 2    # local col in 2x2 (output's col)
                        global_row = b_col_offset + c  # output's col
                        global_col = a  # output's row
                        global_w = global_row * n + global_col
                        W_vec[global_w] += coeff
                    
                    terms.append((U_vec, V_vec, W_vec))
        
        # Now merge terms that have identical V and W vectors
        # Group by (V_vec, W_vec)
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (U_vec, V_vec, W_vec) in enumerate(terms):
            # Use rounded values as key to handle floating point
            V_key = tuple(round(v, 10) for v in V_vec)
            W_key = tuple(round(v, 10) for v in W_vec)
            groups[(V_key, W_key)].append((idx, U_vec))
        
        # Build merged decomposition
        merged_U = []
        merged_V = []
        merged_W = []
        for (V_key, W_key), group in groups.items():
            # Sum all U vectors in this group
            U_merged = [0.0] * (n * m)
            for idx, U_vec in group:
                for i in range(len(U_vec)):
                    U_merged[i] += U_vec[i]
            
            # Recover V and W from key
            V_merged = [v for v in V_key]
            W_merged = [w for w in W_key]
            
            merged_U.append(U_merged)
            merged_V.append(V_merged)
            merged_W.append(W_merged)
        
        rank = len(merged_U)
        
        # Convert to list-of-lists format
        U = [list(row) for row in merged_U]
        V = [list(row) for row in merged_V]
        W = [list(row) for row in merged_W]
        
        return U, V, W, rank

    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def try_linear_reduction(U, V, W, r):
        """
        Try to reduce rank by checking if any term's tensor is a linear combination
        of others. Uses a more robust approach: solve for coefficients that express
        term t as sum of c_s * term_s, then if successful, we can drop term t
        only if we can also adjust the other terms' U vectors.
        
        Actually, the correct approach: if U_t ⊗ V_t ⊗ W_t = Σ c_s (U_s ⊗ V_s ⊗ W_s),
        this is a constraint on the full tensor. We can't simply drop term t unless
        V_t = V_s and W_t = W_s for all s with c_s ≠ 0.
        
        Better approach: use the null space of the tensor matricization.
        """
        # For now, try a simpler approach: check for duplicate (V,W) pairs again
        # and also check if any U column is zero
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        # Remove any terms where U, V, or W is all zeros
        keep = []
        for s in range(r):
            if all(abs(U[i][s]) < 1e-12 for i in range(n*m)) or \
               all(abs(V[i][s]) < 1e-12 for i in range(m*p)) or \
               all(abs(W[i][s]) < 1e-12 for i in range(p*n)):
                continue
            keep.append(s)
        
        if len(keep) < r:
            new_r = len(keep)
            new_U = [[U[i][c] for c in keep] for i in range(n*m)]
            new_V = [[V[i][c] for c in keep] for i in range(m*p)]
            new_W = [[W[i][c] for c in keep] for i in range(p*n)]
            if verify(new_U, new_V, new_W, new_r):
                return new_U, new_V, new_W, new_r
        
        return U, V, W, r

    def randomized_rank_reduction(U, V, W, r, max_iters=200):
        """
        Use randomized linear algebra to find the rank of the tensor.
        The tensor T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
        We can check if the decomposition is minimal by trying to find
        a lower-rank representation.
        
        Approach: use the fact that for a rank-r tensor, the matricizations
        have rank at most r. We can check if any column is redundant.
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        current_r = r
        
        # Try to eliminate each term by solving a linear system
        # where we express term t's contribution as a combination of others
        # This is valid if we can find c_s such that for all (i,j,k):
        # U[i,t]*V[j,t]*W[k,t] = sum_s c_s * U[i,s]*V[j,s]*W[k,s]
        # AND we can update the U vectors: U[i,s] -> U[i,s] + c_s*U[i,t] won't work
        # because that changes the tensor.
        
        # Correct approach: we need to find a different decomposition.
        # Let's try: for each pair of terms, check if their (V,W) are proportional
        # If V_t = alpha * V_s and W_t = beta * W_s, then:
        # U_t ⊗ V_t ⊗ W_t = alpha*beta * U_t ⊗ V_s ⊗ W_s
        # So we can merge: U_s_new = U_s + alpha*beta*U_t / (alpha*beta) ... no
        # Actually: U_t ⊗ (alpha V_s) ⊗ (beta W_s) = alpha*beta * U_t ⊗ V_s ⊗ W_s
        # So term t contributes alpha*beta * U_t ⊗ V_s ⊗ W_s
        # We can add this to term s: U_s_new = U_s + alpha*beta * U_t
        # Then drop term t.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_keep = list(range(current_r))
            eliminated = set()
            
            for t in range(current_r):
                if t in eliminated:
                    continue
                if time.time() > deadline:
                    break
                
                # Check if V_t and W_t are proportional to some V_s and W_s
                for s in range(current_r):
                    if s == t or s in eliminated:
                        continue
                    
                    # Check if V_t = alpha * V_s
                    alpha = None
                    valid_alpha = True
                    for i in range(m*p):
                        if abs(V[i][s]) > 1e-10:
                            if alpha is None:
                                alpha = V[i][t] / V[i][s]
                            elif abs(V[i][t] - alpha * V[i][s]) > 1e-9:
                                valid_alpha = False
                                break
                        elif abs(V[i][t]) > 1e-9:
                            valid_alpha = False
                            break
                    
                    if not valid_alpha or alpha is None:
                        continue
                    
                    # Check if W_t = beta * W_s
                    beta = None
                    valid_beta = True
                    for i in range(p*n):
                        if abs(W[i][s]) > 1e-10:
                            if beta is None:
                                beta = W[i][t] / W[i][s]
                            elif abs(W[i][t] - beta * W[i][s]) > 1e-9:
                                valid_beta = False
                                break
                        elif abs(W[i][t]) > 1e-9:
                            valid_beta = False
                            break
                    
                    if not valid_beta or beta is None:
                        continue
                    
                    # Merge: U_s += alpha * beta * U_t, then drop t
                    scale = alpha * beta
                    for i in range(n*m):
                        U[i][s] += scale * U[i][t]
                    
                    eliminated.add(t)
                    changed = True
                    break  # Restart the loop after a merge
            
            if changed:
                # Rebuild with eliminated terms removed
                keep = [i for i in range(current_r) if i not in eliminated]
                current_r = len(keep)
                U = [[U[i][c] for c in keep] for i in range(n*m)]
                V = [[V[i][c] for c in keep] for i in range(m*p)]
                W = [[W[i][c] for c in keep] for i in range(p*n)]
        
        if current_r < r:
            if verify(U, V, W, current_r):
                return U, V, W, current_r
        
        return U, V, W, r

    # Start with the Strassen-based decomposition
    U, V, W, r = strassen_based_decomposition()
    
    # Verify the base decomposition
    if not verify(U, V, W, r):
        # Fall back to schoolbook
        r = n * m * p
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        W = [[0.0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    else:
        # Try to reduce rank
        U, V, W, r = randomized_rank_reduction(U, V, W, r)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
