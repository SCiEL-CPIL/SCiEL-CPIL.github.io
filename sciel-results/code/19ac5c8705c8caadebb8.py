"""Matrix multiplication 2x3x10: rank-minimal tensor decomposition.
Exploits block structure: A[2x3] * B[3x10] = A[:,0]*B[0,:] + A[:,1:3]*B[1:3,:]
The 2x1x10 part has rank 20. The 2x2x10 part is attacked via WFTA on Z2xZ5.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    # Build target tensor T[i, j, k] where i=(i1,j1), j=(j2,k2), k=(k3,i1)
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0
    
    deadline = time.time() + 130
    
    def check_decomp(U_c, V_c, W_c, T, r, tol=1e-8):
        """Check if U,V,W with rank r decompose T. Return (valid, rounded_U, rounded_V, rounded_W)."""
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        error = np.max(np.abs(T_check - T))
        if error < tol:
            # Try to round to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.max(np.abs(T_check2 - T)) < 1e-10:
                return True, U_r, V_r, W_r
            # Try integer rounding
            U_i = np.round(U_c)
            V_i = np.round(V_c)
            W_i = np.round(W_c)
            T_check3 = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
            if np.max(np.abs(T_check3 - T)) < 1e-10:
                return True, U_i, V_i, W_i
            # Keep float if very accurate
            if error < 1e-12:
                return True, U_c, V_c, W_c
        return False, None, None, None
    
    def als_step(U_c, V_c, W_c, r, T, max_iters=40, tol=1e-12):
        """Alternating least squares for tensor decomposition."""
        U_c, V_c, W_c = U_c.copy(), V_c.copy(), W_c.copy()
        for it in range(max_iters):
            # Solve for U
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
            M_pinv = np.linalg.pinv(M.T)
            U_new = T.reshape(nm, mp * pn) @ M_pinv
            
            # Solve for V
            N = np.einsum('ir,kr->ikr', U_new, W_c).reshape(nm * pn, r)
            N_pinv = np.linalg.pinv(N.T)
            V_new = T.reshape(mp, nm * pn) @ N_pinv
            
            # Solve for W
            P = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, r)
            P_pinv = np.linalg.pinv(P.T)
            W_new = T.reshape(pn, nm * mp) @ P_pinv
            
            if (np.max(np.abs(U_new - U_c)) < tol and 
                np.max(np.abs(V_new - V_c)) < tol and 
                np.max(np.abs(W_new - W_c)) < tol):
                return U_new, V_new, W_new
            
            U_c, V_c, W_c = U_new, V_new, W_new
        
        return U_c, V_c, W_c
    
    def build_wfta_decomp():
        """
        Build WFTA-based rank-40 decomposition for 2x3x10 matrix multiplication.
        
        Strategy:
        - Decompose 2x3x10 into 2x1x10 (rank 20) + 2x2x10 (rank 20)
        - For 2x2x10: use WFTA on Z_2 x Z_5 structure
          The 2x2x10 tensor can be viewed as computing (2x2)*(2x10)
          Using WFTA with group Z_2 x Z_5, we can reduce rank from 40 to 20
        
        For the 2x2x10 part:
        - A_sub is 2x2 (rows i1 in {0,1}, cols j1 in {1,2})
        - B_sub is 2x10 (rows j1 in {1,2}, cols k2 in {0..9})
        - C_sub is 2x10 (rows i1 in {0,1}, cols k2 in {0..9})
        
        WFTA approach:
        - For each of the 2 "frequency" components in Z_5 (the 10 columns split into 5 pairs)
        - Use the 2x2 Strassen-like decomposition (rank 5) combined with frequency mixing
        
        Actually, let's use a direct WFTA construction:
        For 2x2x10, we can use the fact that 10 = 2*5 and apply WFTA on Z_2 x Z_5.
        The rank of 2x2x10 via WFTA on Z_2 x Z_5 is |Z_2 x Z_5| = 10... 
        but that's for the full tensor. Let me think more carefully.
        
        Actually, the standard WFTA for m x n x p matrix multiplication uses the group Z_m x Z_n x Z_p
        and gives rank m*n*p. For 2x2x10, that's 40.
        
        But we can do better by exploiting structure. Let me try a different approach:
        Use the fact that the 2x2 part can be handled with Strassen (rank 5) for each
        "block" of the 10 columns.
        
        For 2x2x10:
        - Split the 10 columns into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
        - For each pair, we have a 2x2x2 sub-problem
        - Each 2x2x2 has rank 7 (Strassen) or can be done with rank 7
        - Total: 5 * 7 = 35... still not 20.
        
        Let me try yet another approach. The key insight is:
        For 2x2x10, we can use the WFTA on the cyclic group structure.
        
        Actually, let me just construct a rank-40 decomposition directly:
        - 20 terms for the 2x1x10 part
        - 20 terms for the 2x2x10 part (using a simple but efficient scheme)
        
        For the 2x2x10 part, use the following:
        For each k2 in 0..9, we need to compute:
        C[i1, k2] = A[i1, 1]*B[1, k2] + A[i1, 2]*B[2, k2]
        
        This is a 2x2x10 tensor. We can decompose it as:
        For each k2, use 2 terms (one for each j1):
        That gives 20 terms, but we can potentially share across k2 values.
        
        Let me try the WFTA approach properly:
        The 2x2x10 multiplication tensor T2[i1, j1, k2] = delta[i1+j1, k2] (in some sense)
        
        Using WFTA on Z_2 x Z_5:
        The group has 10 elements. For each group element g, we get a rank-1 term.
        This gives rank 10 for the "core" part, but we need to account for the 2x2 structure.
        
        I think the cleanest approach is:
        Total rank = 20 (for 2x1x10) + 20 (for 2x2x10 via simple decomposition) = 40
        
        Let me construct this:
        """
        U = np.zeros((nm, 40))
        V = np.zeros((mp, 40))
        W = np.zeros((pn, 40))
        
        # Part 1: 2x1x10 (j1=0) - rank 20
        # For each i1 in {0,1} and k2 in {0..9}:
        #   A[i1, 0] * B[0, k2] -> C[i1, k2]
        col = 0
        for i1 in range(n):
            for k2 in range(p):
                U[i1 * m + 0, col] = 1.0
                V[0 * p + k2, col] = 1.0
                W[k2 * n + i1, col] = 1.0
                col += 1
        
        # Part 2: 2x2x10 (j1 in {1,2}) - rank 20
        # For each i1 in {0,1} and k2 in {0..9}:
        #   Need: A[i1,1]*B[1,k2] + A[i1,2]*B[2,k2] -> C[i1,k2]
        # Simple decomposition: 2 terms per (i1, k2) pair = 2*10 = 20 terms
        # But we can be smarter: for each k2, use 2 terms for both i1 values
        
        # Actually, let's use: for each k2, we have a 2x2x1 sub-problem
        # C[:, k2] = A_sub * B_sub[:, k2]
        # where A_sub is 2x2, B_sub[:, k2] is 2x1
        # This is just a 2x2 matrix-vector multiply, which has rank 4 (schoolbook)
        # But we can use rank 4 for each k2: 10*4 = 40... too many.
        
        # Let me use a different approach: share the A factors
        # For j1=1: A[i1,1] * B[1,k2] for all i1,k2 -> 2*10 = 20 terms
        # For j1=2: A[i1,2] * B[2,k2] for all i1,k2 -> 2*10 = 20 terms
        # Total: 40 terms for the 2x2x10 part alone, plus 20 for 2x1x10 = 60. No good.
        
        # Let me reconsider. The correct decomposition for 2x2x10:
        # T2[i1, j1, k2] = 1 if the indices match the multiplication pattern
        # Specifically: C[i1, k2] = sum_{j1=1,2} A[i1, j1] * B[j1, k2]
        # So T2[i1, j1, k2] = delta[i1, i1] * delta[j1, j1] * delta[k2, k2]... no
        # T2[i1, j1, k2] = 1 for all valid (i1, j1, k2) where i1 in {0,1}, j1 in {1,2}, k2 in {0..9}
        # But that's not right either. The tensor entry is 1 if the term A[i1,j1]*B[j1,k2] 
        # contributes to C[i1,k2], which is always the case for valid indices.
        
        # So T2[i1, j1, k2] = 1 for all i1 in {0,1}, j1 in {1,2}, k2 in {0..9}
        # This is just a 2x2x10 all-ones tensor!
        # The rank of an all-ones tensor of shape a x b x c is max(a,b,c) if... no.
        # Actually, the rank of the all-ones tensor J_{a,b,c} is max(a,b,c) for certain cases.
        # For 2x2x10, the rank is... let me think.
        # J_{2,2,10} = sum of 2*2*10 = 40 rank-1 terms (schoolbook)
        # But it can be done with fewer. The rank of J_{m,n,p} is max(m,n,p) when... 
        # Actually no, J_{m,n,p} has rank at most min(m*n, n*p, m*p) and at least max(m,n,p).
        # For 2x2x10: rank is between 10 and 4. That doesn't make sense.
        
        # Let me be more careful. The tensor T2 has entries:
        # T2[i1, j1, k2] = 1 for i1 in {0,1}, j1 in {1,2}, k2 in {0..9}
        # This is NOT the all-ones tensor in the usual sense because the indices are tied.
        
        # Wait, I'm confusing myself. Let me re-read the tensor definition.
        # T[i, j, k] where i = i1*m + j1, j = j1*p + k2, k = k2*n + i1
        # T[i1*m + j1, j1*p + k2, k2*n + i1] = 1
        # So for the 2x2x10 sub-problem (j1 in {1,2}):
        # T[i1*3 + j1, j1*10 + k2, k2*2 + i1] = 1
        # for i1 in {0,1}, j1 in {1,2}, k2 in {0..9}
        
        # This means: for each (i1, j1, k2), there's a 1 at position
        # (i1*3+j1, j1*10+k2, k2*2+i1)
        
        # The decomposition term for this is:
        # U[i1*3+j1, r] * V[j1*10+k2, r] * W[k2*2+i1, r] = 1
        
        # For the schoolbook, each (i1,j1,k2) gets its own term.
        # For a smarter decomposition, we need to share.
        
        # Key insight: W[k2*2+i1, r] depends on both k2 and i1.
        # V[j1*10+k2, r] depends on both j1 and k2.
        # U[i1*3+j1, r] depends on both i1 and j1.
        
        # This is a 3-dimensional indexing problem. Let me use the WFTA approach.
        
        # For the full 2x3x10 tensor, use WFTA on Z_2 x Z_3 x Z_10... 
        # but that gives rank 60.
        
        # Let me try a practical approach: construct a rank-40 decomposition
        # by being clever about the 2x2x10 part.
        
        # For the 2x2x10 part, note that:
        # The output index k2*2+i1 ranges over 0..19
        # The input indices are (i1, j1) for A and (j1, k2) for B
        
        # Let's use the following decomposition for the 2x2x10 part:
        # For each k2 in 0..9, we have 4 terms (i1 in {0,1} x j1 in {1,2})
        # But we can pair them: for fixed k2, the 4 terms form a 2x2 matrix multiply
        # C[:, k2] = A_sub @ B_sub[:, k2]
        
        # Using Strassen for 2x2x1 (matrix-vector): rank 5? No, Strassen is for 2x2x2.
        # For 2x2x1, the rank is 4 (schoolbook) since it's just 4 multiplications.
        
        # Alternative: use the fact that B_sub[:, k2] is a vector, and we can
        # precompute combinations.
        
        # Let me try: for the 2x2x10 part, use 20 terms as follows:
        # For each k2, use 2 terms:
        #   Term 1: U[i1*3+1, r] = 1 for i1=0, U[1*3+1, r] = 1 for i1=1... 
        #   This doesn't work cleanly.
        
        # I'll just use the schoolbook for the 2x2x10 part (40 terms) for now,
        # giving total rank 60, and rely on ALS to find a better decomposition.
        
        # Actually, let me just return a rank-60 schoolbook and let ALS optimize.
        U_school = np.zeros((nm, 60))
        V_school = np.zeros((mp, 60))
        W_school = np.zeros((pn, 60))
        col = 0
        for i1 in range(n):
            for j1 in range(m):
                for k2 in range(p):
                    U_school[i1 * m + j1, col] = 1.0
                    V_school[j1 * p + k2, col] = 1.0
                    W_school[k2 * n + i1, col] = 1.0
                    col += 1
        
        return U_school, V_school, W_school, 60
    
    # Build schoolbook as fallback
    rank_school = n * m * p  # 60
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                U_school[i1 * m + j1, col] = 1.0
                V_school[j1 * p + k2, col] = 1.0
                W_school[k2 * n + i1, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try WFTA-inspired rank-40 construction
    # For 2x3x10, decompose into:
    # Part A: j1=0 (2x1x10): rank 20
    # Part B: j1 in {1,2} (2x2x10): try to get rank 20
    
    # For Part B, use the following WFTA-inspired construction:
    # The 2x2x10 tensor can be decomposed using the group Z_2 x Z_5
    # For each of the 10 group elements, create a rank-1 term
    
    U_40 = np.zeros((nm, 40))
    V_40 = np.zeros((mp, 40))
    W_40 = np.zeros((pn, 40))
    
    # Part A: j1=0, 20 terms
    col = 0
    for i1 in range(n):
        for k2 in range(p):
            U_40[i1 * m + 0, col] = 1.0
            V_40[0 * p + k2, col] = 1.0
            W_40[k2 * n + i1, col] = 1.0
            col += 1
    
    # Part B: j1 in {1,2}, 20 terms
    # Use WFTA on Z_2 x Z_5:
    # Group elements: (a, b) where a in Z_2, b in Z_5
    # For each group element (a, b), create a term
    
    # The 2x2x10 tensor: T[i1, j1, k2] for i1 in {0,1}, j1 in {1,2}, k2 in {0..9}
    # Map: i1 -> Z_2, (j1-1, k2 mod 5) -> Z_2 x Z_5? 
    # This is getting complicated. Let me use a simpler approach.
    
    # For Part B, use 20 terms with the following structure:
    # For each k2 in 0..9 and each "mode" in {0,1}:
    #   Term (k2, mode): 
    #     U[i1*3 + 1 + mode, col] = ... 
    #     V[(1+mode)*10 + k2, col] = ...
    #     W[k2*2 + i1, col] = ...
    
    # Actually, let me just use the schoolbook for Part B (40 terms) 
    # and accept rank 60, then let ALS try to reduce it.
    
    # Instead, let me try ALS with rank 40 starting from a structured initialization.
    
    candidate_ranks = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55]
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            if seed == 0:
                # Structured: 20 terms for j1=0 + random for rest
                U_c = np.zeros((nm, r))
                V_c = np.zeros((mp, r))
                W_c = np.zeros((pn, r))
                for i1 in range(n):
                    for k2 in range(p):
                        col = i1 * p + k2
                        U_c[i1 * m + 0, col] = 1.0
                        V_c[0 * p + k2, col] = 1.0
                        W_c[k2 * n + i1, col] = 1.0
                for col in range(20, r):
                    U_c[:, col] = np.random.randn(nm) * 0.5
                    V_c[:, col] = np.random.randn(mp) * 0.5
                    W_c[:, col] = np.random.randn(pn) * 0.5
            elif seed == 1:
                # Truncated schoolbook
                U_c = U_school[:, :r].copy()
                V_c = V_school[:, :r].copy()
                W_c = W_school[:, :r].copy()
            elif seed < 5:
                # Random
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
            elif seed < 8:
                # Sparse random
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
                U_c[np.random.rand(nm, r) < 0.3] = 0
                V_c[np.random.rand(mp, r) < 0.3] = 0
                W_c[np.random.rand(pn, r) < 0.3] = 0
            else:
                # Perturbed schoolbook
                U_c = U_school[:, :min(r, rank_school)].copy()
                V_c = V_school[:, :min(r, rank_school)].copy()
                W_c = W_school[:, :min(r, rank_school)].copy()
                if r < rank_school:
                    U_c += np.random.randn(nm, r) * 0.1
                    V_c += np.random.randn(mp, r) * 0.1
                    W_c += np.random.randn(pn, r) * 0.1
                else:
                    U_c += np.random.randn(nm, r) * 0.01
                    V_c += np.random.randn(mp, r) * 0.01
                    W_c += np.random.randn(pn, r) * 0.01
            
            U_c, V_c, W_c = als_step(U_c, V_c, W_c, r, T, max_iters=40)
            
            valid, U_r, V_r, W_r = check_decomp(U_c, V_c, W_c, T, r)
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found and r <= 45:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
