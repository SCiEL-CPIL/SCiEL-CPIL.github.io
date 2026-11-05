import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialize with Schoolbook Decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1
                
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(rank):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    current_rank = rank_school
    
    # 2. Randomized Local Search for Rank Reduction
    # Strategy: Try to replace two terms (r1, r2) with a single new term (r_new)
    # such that the total tensor sum remains unchanged.
    # This requires solving a system of equations.
    # T_r1 + T_r2 = T_new
    # u1 \otimes v1 \otimes w1 + u2 \otimes v2 \otimes w2 = u_new \otimes v_new \otimes w_new
    # This is generally impossible for arbitrary u,v,w.
    # However, if u1=u2 and v1=v2, then w_new = w1+w2.
    # So we look for pairs with proportional U and V vectors.
    
    # To make it a "search", we will randomly perturb the vectors slightly and see 
    # if we can find a near-dependence that allows exact integer/half-integer solution?
    # No, must be exact.
    
    # Let's try a different search: 
    # We know the Strassen 4x4x4 block is optimal (49).
    # The rest is schoolbook.
    # We will attempt to search for a better decomposition of the "remainder" 
    # (the parts involving index 4 in j or k) by mixing them with the Strassen block?
    # This is complex.
    
    # Instead, let's implement a generic "Term Merging" search:
    # Iterate through all pairs of terms.
    # Check if U[:, r1] is a scalar multiple of U[:, r2] AND V[:, r1] is the same scalar multiple of V[:, r2].
    # If so, W[:, r_new] = W[:, r1] + scalar * W[:, r2].
    # Remove r1, r2, add r_new. Rank decreases by 1.
    
    def merge_terms(U, V, W, current_rank):
        merged = False
        # We check pairs. O(R^2)
        for r1 in range(current_rank):
            if time.time() > deadline:
                return U, V, W, current_rank, False
            for r2 in range(r1 + 1, current_rank):
                # Check if U[:, r1] and U[:, r2] are proportional
                # and V[:, r1] and V[:, r2] are proportional with same factor
                # Use cross product or simple ratio check
                u1 = U[:, r1]
                u2 = U[:, r2]
                v1 = V[:, r1]
                v2 = V[:, r2]
                
                # Find non-zero indices to determine ratio
                idxs = np.nonzero(u1)[0]
                if len(idxs) == 0: continue
                
                ratio = None
                valid = True
                for idx in idxs:
                    if u2[idx] == 0:
                        valid = False
                        break
                    r = u2[idx] / u1[idx]
                    if ratio is None:
                        ratio = r
                    else:
                        if abs(r - ratio) > 1e-9:
                            valid = False
                            break
                if not valid or ratio is None:
                    continue
                    
                # Check V
                idxs_v = np.nonzero(v1)[0]
                if len(idxs_v) == 0: continue
                valid_v = True
                for idx in idxs_v:
                    if v2[idx] == 0:
                        valid_v = False
                        break
                    r_v = v2[idx] / v1[idx]
                    if abs(r_v - ratio) > 1e-9:
                        valid_v = False
                        break
                if not valid_v:
                    continue
                
                # We found a mergeable pair!
                # New W = W[r1] + ratio * W[r2]
                w_new = W[:, r1] + ratio * W[:, r2]
                
                # Construct new matrices
                # Remove r2, update r1
                U_new = np.delete(U, r2, axis=1)
                V_new = np.delete(V, r2, axis=1)
                W_new = np.delete(W, r2, axis=1)
                
                # Update W_new at index r1 (or r1-1 if r2 < r1, but r2 > r1 so r1 stays)
                W_new[:, r1] = w_new
                
                # Verify
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new, current_rank - 1, True
                else:
                    # Should not fail if math is correct, but safe check
                    pass
                    
        return U, V, W, current_rank, False

    # Run merging search
    improved = True
    while improved:
        if time.time() > deadline:
            break
        U, V, W, current_rank, improved = merge_terms(U, V, W, current_rank)

    # If no merges found, we are at a local minimum for this heuristic.
    # For schoolbook, no merges exist (all basis vectors are orthogonal).
    # So this will return 100.
    
    # To beat 100, we MUST use the Strassen block.
    # I will hardcode the Strassen block construction as the "initial guess" 
    # for the search, rather than schoolbook.
    # This is a different strategy: "Start from a known good algebraic structure 
    # and try to improve via local search."
    
    # Re-initialize with Strassen Block
    def build_strassen_block():
        # 4x4x4 Strassen
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        def tensor_product_vec(v1, v2):
            result = [0] * 16
            for a1 in range(2):
                for b1 in range(2):
                    for a2 in range(2):
                        for b2 in range(2):
                            idx = (a1*2+a2)*4 + (b1*2+b2)
                            result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
            return result
        
        strassen_4x4x4 = []
        for t1 in strassen_2x2x2:
            for t2 in strassen_2x2x2:
                u = tensor_product_vec(t1[0], t2[0])
                v = tensor_product_vec(t1[1], t2[1])
                w = tensor_product_vec(t1[2], t2[2])
                strassen_4x4x4.append((u, v, w))
        
        rank = 49 + 16 + 16 + 4
        U_s = np.zeros((n * m, rank), dtype=float)
        V_s = np.zeros((m * p, rank), dtype=float)
        W_s = np.zeros((p * n, rank), dtype=float)
        
        # Add Strassen 4x4x4 terms
        for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
            for i in range(4):
                for j in range(4):
                    U_s[i*5+j, r_idx] = u[i*4+j]
            for j in range(4):
                for k in range(4):
                    V_s[j*5+k, r_idx] = v[j*4+k]
            for k in range(4):
                for i in range(4):
                    W_s[k*4+i, r_idx] = w[k*4+i]
        
        # Add 4x1x4 block (j<4, k==4): schoolbook
        r_offset = 49
        for i in range(4):
            for j in range(4):
                U_s[i*5+j, r_offset] = 1
                V_s[j*5+4, r_offset] = 1
                W_s[4*4+i, r_offset] = 1
                r_offset += 1
        
        # Add 1x4x4 block (j==4, k<4): schoolbook
        for i in range(4):
            for k in range(4):
                U_s[i*5+4, r_offset] = 1
                V_s[4*5+k, r_offset] = 1
                W_s[k*4+i, r_offset] = 1
                r_offset += 1
        
        # Add 1x1x4 block (j==4, k==4): schoolbook
        for i in range(4):
            U_s[i*5+4, r_offset] = 1
            V_s[4*5+4, r_offset] = 1
            W_s[4*4+i, r_offset] = 1
            r_offset += 1
            
        return U_s, V_s, W_s, rank

    U, V, W, current_rank = build_strassen_block()
    
    if not verify(U, V, W):
        # Fallback to schoolbook if Strassen construction has a bug
        U, V, W = np.zeros((n*m, n*m*p)), np.zeros((m*p, n*m*p)), np.zeros((p*n, n*m*p))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        current_rank = n*m*p

    # Now run the local search (merging) on the Strassen-based solution
    improved = True
    while improved:
        if time.time() > deadline:
            break
        U, V, W, current_rank, improved = merge_terms(U, V, W, current_rank)

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
