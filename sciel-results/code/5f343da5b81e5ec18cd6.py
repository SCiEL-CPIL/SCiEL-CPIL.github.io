import numpy as np
import time
from itertools import product, combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8

    # --- Schoolbook fallback (rank 64) ---
    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # --- Rank-56 Strassen block decomposition (known good) ---
    def make_strassen_56():
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7))
        # Standard Strassen for 2x2*2x2
        U_22[0,0]=1; U_22[3,0]=1
        V_22[2,0]=1; V_22[3,0]=1
        W_22[0,0]=1; W_22[3,0]=-1
        U_22[2,1]=1; U_22[3,1]=1
        V_22[0,1]=1
        W_22[2,1]=1; W_22[0,1]=-1
        U_22[0,2]=1
        V_22[0,2]=1; V_22[2,2]=-1
        W_22[1,2]=1; W_22[3,2]=1
        U_22[3,3]=1
        V_22[3,3]=1; V_22[1,3]=-1
        W_22[1,3]=-1; W_22[2,3]=-1
        U_22[0,4]=1; U_22[1,4]=1
        V_22[3,4]=1
        W_22[1,4]=1; W_22[0,4]=1
        U_22[2,5]=1; U_22[0,5]=-1
        V_22[0,5]=1; V_22[1,5]=1
        W_22[2,5]=1; W_22[3,5]=-1
        U_22[1,6]=1; U_22[3,6]=-1
        V_22[2,6]=1; V_22[3,6]=1
        W_22[0,6]=1; W_22[3,6]=1

        U_g = np.zeros((8, 56))
        V_g = np.zeros((32, 56))
        W_g = np.zeros((16, 56))

        def add_block(u_map, v_map, w_map, sc):
            for c in range(7):
                for l in range(4):
                    if U_22[l,c] != 0:
                        U_g[u_map[l], sc+c] += U_22[l,c]
                    if V_22[l,c] != 0:
                        V_g[v_map[l], sc+c] += V_22[l,c]
                    if W_22[l,c] != 0:
                        W_g[w_map[l], sc+c] += W_22[l,c]

        # 8 blocks: 2 row-groups x 4 col-groups
        add_block([0,1,4,5],[0,1,8,9],[0,1,2,3],0)
        add_block([0,1,4,5],[2,3,10,11],[4,5,6,7],7)
        add_block([0,1,4,5],[4,5,12,13],[8,9,10,11],14)
        add_block([0,1,4,5],[6,7,14,15],[12,13,14,15],21)
        add_block([2,3,6,7],[16,17,24,25],[0,1,2,3],28)
        add_block([2,3,6,7],[18,19,26,27],[4,5,6,7],35)
        add_block([2,3,6,7],[20,21,28,29],[8,9,10,11],42)
        add_block([2,3,6,7],[22,23,30,31],[12,13,14,15],49)
        return U_g, V_g, W_g

    # --- Search for rank-13 decomposition of 2x2*2x4 ---
    # Tensor: A(2x2) * B(2x4) -> C(2x4)
    # Indices: A: 0,1,2,3; B: 0,1,2,3,4,5,6,7; C: 0,1,2,3,4,5,6,7
    # T[a,b,c] = 1 iff c = a*2 + (b // 2) ... wait
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # A index: i*2+j -> 0,1,2,3
    # B index: j*4+k -> 0..7
    # C index: i*4+k -> 0..7
    # T[a_idx, b_idx, c_idx] = 1 iff (i,j) from a_idx, (j',k) from b_idx, (i',k') from c_idx
    #   with i=i', j=j', k=k'
    
    def build_224_tensor():
        T = np.zeros((4, 8, 8))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    a_idx = i * 2 + j
                    b_idx = j * 4 + k
                    c_idx = i * 4 + k
                    T[a_idx, b_idx, c_idx] = 1
        return T

    def verify_224(U, V, W, T_target):
        r = U.shape[1]
        T_rec = np.zeros((4, 8, 8))
        for c in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return np.allclose(T_rec, T_target, atol=1e-9)

    def search_224_rank13():
        """Search for rank-13 decomposition of 2x2*2x4."""
        T_target = build_224_tensor()
        
        # Strategy: start from rank-14 Strassen-like and try to merge/eliminate terms
        # Rank-14: two independent Strassen 2x2*2x2 blocks
        U14 = np.zeros((4, 14))
        V14 = np.zeros((8, 14))
        W14 = np.zeros((8, 14))
        
        # Strassen for first 2x2*2x2 (columns 0,1 of B -> columns 0,1 of C)
        U14[0,0]=1; U14[3,0]=1
        V14[2,0]=1; V14[3,0]=1  # B[1,0],B[1,1]
        W14[0,0]=1; W14[1,0]=-1  # C[0,0],C[0,1]
        U14[2,1]=1; U14[3,1]=1
        V14[0,1]=1  # B[0,0]
        W14[2,1]=1; W14[0,1]=-1  # C[1,0],C[0,0]
        U14[0,2]=1
        V14[0,2]=1; V14[2,2]=-1  # B[0,0]-B[1,0]
        W14[1,2]=1; W14[3,2]=1  # C[0,1],C[1,1]
        U14[3,3]=1
        V14[3,3]=1; V14[1,3]=-1  # B[1,1]-B[0,1]
        W14[1,3]=-1; W14[3,3]=-1  # C[0,1],C[1,1]
        U14[0,4]=1; U14[1,4]=1
        V14[3,4]=1  # B[1,1]
        W14[1,4]=1; W14[0,4]=1  # C[0,1],C[0,0]
        U14[2,5]=1; U14[0,5]=-1
        V14[0,5]=1; V14[1,5]=1  # B[0,0]+B[0,1]
        W14[2,5]=1; W14[3,5]=-1  # C[1,0],C[1,1]
        U14[1,6]=1; U14[3,6]=-1
        V14[2,6]=1; V14[3,6]=1  # B[1,0]+B[1,1]
        W14[0,6]=1; W14[1,6]=-1  # C[0,0],C[0,1]
        
        # Second block (columns 2,3 of B -> columns 2,3 of C)
        U14[0,7]=1; U14[3,7]=1
        V14[6,7]=1; V14[7,7]=1
        W14[4,7]=1; W14[5,7]=-1
        U14[2,8]=1; U14[3,8]=1
        V14[4,8]=1
        W14[6,8]=1; W14[4,8]=-1
        U14[0,9]=1
        V14[4,9]=1; V14[6,9]=-1
        W14[5,9]=1; W14[7,9]=1
        U14[3,10]=1
        V14[7,10]=1; V14[5,10]=-1
        W14[5,10]=-1; W14[7,10]=-1
        U14[0,11]=1; U14[1,11]=1
        V14[7,11]=1
        W14[5,11]=1; W14[4,11]=1
        U14[2,12]=1; U14[0,12]=-1
        V14[4,12]=1; V14[5,12]=1
        W14[6,12]=1; W14[7,12]=-1
        U14[1,13]=1; U14[3,13]=-1
        V14[6,13]=1; V14[7,13]=1
        W14[4,13]=1; W14[5,13]=-1
        
        if not verify_224(U14, V14, W14, T_target):
            return None
        
        # Try to find rank-13 by eliminating one term via linear combination
        # For each pair of terms (i,j), check if term i can be expressed as
        # linear combination of the other 13 terms (restricted to make it rank 13)
        # Actually: check if removing term i and adjusting others still works.
        # This is: is term i in the span of the other 13 terms?
        
        # Build the tensor as a matrix: each term is a vector in R^{4*8*8}
        # Flatten each term
        def flatten_term(U, V, W, c):
            vec = np.zeros((4, 8, 8))
            for a in range(4):
                for b in range(8):
                    for cc in range(8):
                        vec[a,b,cc] = U[a,c] * V[b,c] * W[cc,c]
            return vec.flatten()
        
        # Check if we can eliminate any single term
        for elim in range(14):
            # Try: can we adjust the remaining 13 terms to compensate?
            # This requires solving a linear system - too expensive for 13 unknowns per elimination
            # Instead: check if the eliminated term's contribution is already covered
            # by a linear combination of the other terms' contributions
            pass
        
        # Alternative: try specific structural modifications
        # Merge terms that share structure
        
        # Try: can terms 0 and 7 be merged? (they both have U pattern [1,0,0,1])
        # Term 0: U=[1,0,0,1], V=[0,0,1,1,0,0,0,0], W=[1,-1,0,0,0,0,0,0]
        # Term 7: U=[1,0,0,1], V=[0,0,0,0,0,0,1,1], W=[0,0,0,0,1,-1,0,0]
        # If we add them: U=[2,0,0,2], V=[0,0,1,1,0,0,1,1], W=[1,-1,0,0,1,-1,0,0]
        # This doesn't help directly.
        
        # Try a different approach: use the fact that for 2x2*2x4,
        # we can use 7 terms for the "main" 2x2*2x2 and 6 terms for the extension
        # if we can share some structure.
        
        # Actually, let's try a direct rank-13 construction:
        # Use 7 Strassen terms for columns 0-1, and try to handle columns 2-3
        # with only 6 terms by sharing A-precomputations.
        
        # Key insight: For columns 2-3, we need:
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
        
        # This is exactly a 2x2*2x2 multiplication (with B being the 2x2 submatrix
        # of columns 2-3). So it needs 7 terms independently.
        # Total = 14. Can't reduce below 14 with this structure.
        
        # Unless... we use a different decomposition that doesn't split by columns.
        # E.g., use all 4 columns of B together in each term.
        
        # Let's try: 13 terms where each term uses a linear combination of all 8 B entries
        # and all 8 C entries. This is a much larger search space.
        
        # Given time constraints, let's try a randomized local search:
        # Start from rank-14, try to eliminate one term by adjusting all others.
        
        np.random.seed(42)
        U14f = U14.copy()
        V14f = V14.copy()
        W14f = W14.copy()
        
        for elim in range(14):
            if time.time() > deadline - 30:
                break
            # Zero out term 'elim'
            U14f[:, elim] = 0
            V14f[:, elim] = 0
            W14f[:, elim] = 0
            
            # Try to fix by adding small perturbations to other terms
            # This is a nonlinear problem, so use gradient-free approach
            # Actually, let's try: for each other term j, can we add a multiple
            # of term j's pattern to compensate?
            
            # The residual after eliminating term 'elim' is:
            # R = -flatten_term(U14, V14, W14, elim)
            # We need to find coefficients c_j such that sum_j c_j * flatten_term(j) ≈ R
            # This is a linear system!
            
            R = -flatten_term(U14, V14, W14, elim)
            
            # Build matrix of other terms
            M = np.zeros((13, 4*8*8))
            for idx, j in enumerate(range(14)):
                if j == elim:
                    continue
                M[idx, :] = flatten_term(U14, V14, W14, j)
            
            # Solve M^T c = R (least squares)
            try:
                c, residuals, rank, sv = np.linalg.lstsq(M.T, R, rcond=None)
            except:
                continue
            
            # Check if the solution is small enough to be "absorbed"
            # i.e., the adjustment to each term is bounded
            if np.max(np.abs(c)) < 100:
                # Apply the adjustments
                U_new = U14.copy()
                V_new = V14.copy()
                W_new = W14.copy()
                
                for idx, j in enumerate(range(14)):
                    if j == elim:
                        continue
                    # c[idx] is the coefficient to add to term j
                    # But we need to add c[idx] * (original term j) to the sum
                    # This means we modify term j by adding c[idx] * term_j
                    # But that changes the rank structure...
                    # Actually, we just need the sum of all terms to equal T.
                    # If we add c[idx]*term_j to term j, the total sum becomes
                    # sum_j (1+c[idx])*term_j for j≠elim, which equals
                    # sum_all term_j + sum_j c[idx]*term_j = T + R = T - R + R = T
                    # Wait, that's not right. Let me reconsider.
                    
                    # Original: sum_{j=0}^{13} term_j = T
                    # After eliminating term_elim: sum_{j≠elim} term_j = T - term_elim
                    # We want: sum_{j≠elim} (term_j + delta_j) = T
                    # So: sum_{j≠elim} delta_j = term_elim
                    # delta_j = c[idx] * term_j means sum c[idx]*term_j should = term_elim
                    # But we solved M^T c = R = -term_elim, so sum c[idx]*term_j = -term_elim
                    # That's the wrong sign. We need sum c[idx]*term_j = term_elim.
                    # So we should solve M^T c = term_elim, not -term_elim.
                    
                    pass
                
                # Let me redo: solve M^T c = flatten_term(elim)
                R2 = flatten_term(U14, V14, W14, elim)
                try:
                    c2, _, _, _ = np.linalg.lstsq(M.T, R2, rcond=None)
                except:
                    continue
                
                if np.max(np.abs(c2)) < 50:
                    # Check if this gives a valid decomposition
                    # New term j (j≠elim) becomes (1+c2[idx]) * original_term_j
                    # But we can't just scale a term - we need to update U,V,W
                    # The issue: term_j = U_j ⊗ V_j ⊗ W_j
                    # (1+c)*term_j ≠ (1+c)*U_j ⊗ (1+c)*V_j ⊗ (1+c)*W_j
                    # Scaling a rank-1 tensor by c means multiplying one factor by c.
                    # So new_U_j = (1+c[idx]) * U_j, new_V_j = V_j, new_W_j = W_j
                    # This works!
                    
                    U_new = U14.copy()
                    V_new = V14.copy()
                    W_new = W14.copy()
                    U_new[:, elim] = 0
                    V_new[:, elim] = 0
                    W_new[:, elim] = 0
                    
                    for idx, j in enumerate(range(14)):
                        if j == elim:
                            continue
                        scale = 1.0 + c2[idx]
                        if abs(scale) > 1e-10:
                            U_new[:, j] = scale * U14[:, j]
                    
                    # Verify
                    if verify_224(U_new, V_new, W_new, T_target):
                        # Check if entries are "nice" (close to integers or half-integers)
                        all_entries = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
                        all_entries = all_entries[np.abs(all_entries) > 1e-10]
                        if len(all_entries) > 0:
                            # Check if close to integers or half-integers
                            closest = np.round(all_entries * 2) / 2
                            if np.max(np.abs(all_entries - closest)) < 1e-6:
                                # Great! We have a rank-13 with nice entries
                                return U_new, V_new, W_new
                        else:
                            return U_new, V_new, W_new
        
        return None

    # --- Compose 2x2x4 decomposition with dot products to get 2x4x8 ---
    def compose_224_to_248(U224, V224, W224):
        """
        Given a rank-r decomposition of 2x2*2x4,
        construct a rank-2r decomposition of 2x4*4x8.
        
        A (2x4) = [A_left | A_right] where A_left, A_right are 2x2
        B (4x8) = [[B11, B12], [B21, B22]] where Bij are 2x4
        
        C = A*B = [A_left*B11 + A_right*B21, A_left*B12 + A_right*B22]
        
        Wait, this isn't quite right. Let me re-derive.
        
        A is 2x4: A[i,j] for i=0,1; j=0,1,2,3
        B is 4x8: B[j,k] for j=0,1,2,3; k=0..7
        C[i,k] = sum_j A[i,j]*B[j,k]
        
        Split j into j=0,1 (first half) and j=2,3 (second half):
        C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
        
        Let A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
        Let B1 = B[0:2,:] (2x8), B2 = B[2:4,:] (2x8)
        
        C = A1*B1 + A2*B2
        
        Now B1 is 2x8, B2 is 2x8.
        Split B1 into B1_left (2x4, cols 0-3) and B1_right (2x4, cols 4-7)
        Similarly for B2.
        
        C_left (2x4) = A1*B1_left + A2*B2_left
        C_right (2x4) = A1*B1_right + A2*B2_right
        
        Each of A1*B1_left, A2*B2_left is a 2x2*2x4 product.
        If we have a rank-r decomposition for 2x2*2x4, then:
        A1*B1_left uses r terms
        A2*B2_left uses r terms
        Total for C_left: 2r terms
        Total for C_right: 2r terms
        Grand total: 4r terms
        
        So rank = 4 * r(2x2x4).
        If r=13, total rank = 52. Still worse than 56? No, 52 < 56!
        
        Wait, but we can do better. The A1*B1_left and A2*B2_left share the
        same "B structure" if B1_left and B2_left have the same pattern... no they don't.
        
        Actually, let me reconsider. The decomposition of 2x2*2x4 gives us:
        For any 2x2 matrix X and 2x4 matrix Y:
        X*Y = sum_{t=1}^{r} (U_t^T * x) * (V_t^T * y) * W_t
        
        where x is the 4-vector of X entries, y is the 8-vector of Y entries,
        and the result is an 8-vector of (X*Y) entries.
        
        For C_left = A1*B1_left + A2*B2_left:
        A1*B1_left = sum_t (U_t^T * a1) * (V_t^T * b1l) * W_t
        A2*B2_left = sum_t (U_t^T * a2) * (V_t^T * b2l) * W_t
        
        These are independent sums (different a, b vectors), so total 2r terms.
        
        Similarly C_right needs 2r terms.
        Total: 4r terms.
        
        With r=14: 56 (matches current)
        With r=13: 52 (improvement!)
        """
        r = U224.shape[1]
        total_rank = 4 * r
        
        U_global = np.zeros((8, total_rank))
        V_global = np.zeros((32, total_rank))
        W_global = np.zeros((16, total_rank))
        
        # A indices: A[i,j] -> U index i*4+j
        # A1 = A[:,0:2]: indices 0,1,4,5
        # A2 = A[:,2:4]: indices 2,3,6,7
        
        # B indices: B[j,k] -> V index j*8+k
        # B1_left = B[0:2, 0:4]: indices 0,1,2,3, 8,9,10,11
        # B1_right = B[0:2, 4:7]: indices 4,5,6,7, 12,13,14,15
        # B2_left = B[2:4, 0:4]: indices 16,17,18,19, 24,25,26,27
        # B2_right = B[2:4, 4:7]: indices 20,21,22,23, 28,29,30,31
        
        # C indices: C[i,k] -> W index k*2+i
        # C_left = C[:,0:4]: indices 0,1,2,3, 4,5,6,7
        # C_right = C[:,4:7]: indices 8,9,10,11, 12,13,14,15
        
        # For each sub-product (A_block * B_block -> C_block):
        # U_block maps A_block entries (4 entries) to U_global rows
        # V_block maps B_block entries (8 entries) to V_global rows  
        # W_block maps C_block entries (8 entries) to W_global rows
        
        def add_subproduct(u_rows, v_rows, w_rows, start_col):
            for t in range(r):
                for a in range(4):
                    if U224[a, t] != 0:
                        U_global[u_rows[a], start_col + t] += U224[a, t]
                for b in range(8):
                    if V224[b, t] != 0:
                        V_global[v_rows[b], start_col + t] += V224[b, t]
                for c in range(8):
                    if W224[c, t] != 0:
                        W_global[w_rows[c], start_col + t] += W224[c, t]
        
        # Sub-product 1: A1 * B1_left -> C_left
        # A1 entries: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
        # B1_left entries: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=8, B[1,1]=9, B[1,2]=10, B[1,3]=11
        # C_left entries: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
        
        # The 2x2*2x4 decomposition expects:
        # A indices: 0,1,2,3 (row-major 2x2)
        # B indices: 0..7 (row-major 2x4)
        # C indices: 0..7 (row-major 2x4)
        
        # Mapping for A1: A_global[0]->A224[0], A_global[1]->A224[1], A_global[4]->A224[2], A_global[5]->A224[3]
        # Mapping for B1_left: B_global[0]->B224[0], B_global[1]->B224[1], ..., B_global[3]->B224[3], B_global[8]->B224[4], ...
        # Mapping for C_left: C_global[0]->C224[0], ..., C_global[7]->C224[7]
        
        add_subproduct([0,1,4,5], [0,1,2,3,8,9,10,11], [0,1,2,3,4,5,6,7], 0)
        
        # Sub-product 2: A1 * B1_right -> C_right
        add_subproduct([0,1,4,5], [4,5,6,7,12,13,14,15], [8,9,10,11,12,13,14,15], r)
        
        # Sub-product 3: A2 * B2_left -> C_left
        # A2 entries: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
        add_subproduct([2,3,6,7], [16,17,18,19,24,25,26,27], [0,1,2,3,4,5,6,7], 2*r)
        
        # Sub-product 4: A2 * B2_right -> C_right
        add_subproduct([2,3,6,7], [20,21,22,23,28,29,30,31], [8,9,10,11,12,13,14,15], 3*r)
        
        return U_global, V_global, W_global

    # --- Verify full 2x4*4x8 decomposition ---
    def verify_full(U, V, W):
        r = U.shape[1]
        T_rec = np.zeros((8, 32, 16))
        for c in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        T_orig = np.zeros((8, 32, 16))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    T_orig[i*4+j, j*8+k, k*2+i] = 1
        
        return np.allclose(T_rec, T_orig, atol=1e-9)

    # --- Main logic ---
    # Try to find rank-13 for 2x2x4
    U224_13 = None
    if time.time() < deadline - 40:
        U224_13 = search_224_rank13()
    
    best_rank = 64
    best_result = None
    
    if U224_13 is not None:
        U_g, V_g, W_g = compose_224_to_248(*U224_13)
        if verify_full(U_g, V_g, W_g):
            best_rank = U_g.shape[1]
            best_result = (U_g, V_g, W_g)
    
    # Fallback: rank-56 Strassen
    if best_rank > 56 or best_result is None:
        U56, V56, W56 = make_strassen_56()
        if verify_full(U56, V56, W56):
            best_rank = 56
            best_result = (U56, V56, W56)
    
    # Fallback: schoolbook
    if best_result is None:
        U_s, V_s, W_s = make_schoolbook()
        best_result = (U_s, V_s, W_s)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
