"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices for the full problem.
    # However, to keep the code structurally distinct and generalizable,
    # we can use a recursive helper that generates the indices for sub-problems.
    # 
    # The problem is C = A * B, where A is n x m, B is m x p.
    # Tensor T[a_idx, b_idx, c_idx] = 1 if a_idx=(i,j), b_idx=(j,k), c_idx=(k,i)
    #
    # Strategy: Split P into two halves if possible, or N, or M.
    # Since P=4, we can split B into two 2x2 blocks.
    # A is 2x2.
    # So we have two subproblems: C_left = A * B_left (2x2 * 2x2 -> 2x2)
    # and C_right = A * B_right (2x2 * 2x2 -> 2x2).
    #
    # We define a recursive function that returns the rank and the list of
    # (u_idx, v_idx, w_idx, coeff) tuples for a sub-problem of size n_sub x m_sub x p_sub.
    # The indices are local to the sub-problem.
    
    def get_strassen_decomposition(n_sub, m_sub, p_sub):
        """
        Returns a list of tuples (local_u, local_v, local_w, coeff)
        for the tensor decomposition of n_sub x m_sub by m_sub x p_sub multiplication.
        """
        # Base case: 1x1 x 1x1
        if n_sub == 1 and m_sub == 1 and p_sub == 1:
            return [(0, 0, 0, 1.0)]
        
        # If dimensions are small and not all 1s, we might need a different strategy or it's invalid for Strassen.
        # Standard Strassen works for equal powers of 2.
        # Here we have 2x2x2 effectively.
        
        # Check if we can split P
        if p_sub % 2 == 0 and n_sub >= 1 and m_sub >= 1:
            p1 = p_sub // 2
            p2 = p_sub - p1
            # B is m_sub x p_sub. Split into B1 (m_sub x p1) and B2 (m_sub x p2)
            # A is n_sub x m_sub.
            # C is n_sub x p_sub. Split into C1 (n_sub x p1) and C2 (n_sub x p2)
            
            # Recursive calls for the sub-problems
            # Note: The "inner" dimension for A is still m_sub.
            # Subproblem 1: n_sub x m_sub * m_sub x p1
            # Subproblem 2: n_sub x m_sub * m_sub x p2
            
            # However, standard Strassen splits the inner dimension or the square dimensions.
            # If we split P, A is not split. This is valid: C = A * [B1 B2] = [A*B1 A*B2].
            # This reduces the problem to two independent multiplications.
            # This doesn't reduce rank below sum of ranks if no sharing.
            # To get Strassen benefits, we usually split the "square" part.
            
            # Let's try splitting the square part if n_sub == m_sub == p_sub
            if n_sub == m_sub == p_sub:
                h = n_sub // 2
                # Split A into [[A11 A12] [A21 A22]], B into [[B11 B12] [B21 B22]], C into [[C11 C12] [C21 C22]]
                # Strassen formulas:
                # M1 = (A11 + A22) * (B11 + B22)
                # M2 = (A21 + A22) * B11
                # M3 = A11 * (B12 - B22)
                # M4 = A22 * (B21 - B11)
                # M5 = (A11 + A12) * B22
                # M6 = (A21 - A11) * (B11 + B12)
                # M7 = (A12 - A22) * (B21 + B22)
                
                # We need to map these to local indices.
                # This is complex to do recursively with index mapping.
                # Given the small size (2x2x2), it is safer to hardcode the base case for 2x2x2
                # and handle the P-split for 2x2x4.
                
                if n_sub == 2:
                    # Local indices for 2x2x2
                    # U idx: 0:A11, 1:A12, 2:A21, 3:A22
                    # V idx: 0:B11, 1:B12, 2:B21, 3:B22
                    # W idx: 0:C11, 1:C12, 2:C21, 3:C22 (Row-major? No, trace convention)
                    # Trace convention W[k*n+i, r]. For n=2, k is col, i is row.
                    # C[i,k] -> W[k*2+i].
                    # C[0,0] -> W[0]
                    # C[0,1] -> W[2]
                    # C[1,0] -> W[1]
                    # C[1,1] -> W[3]
                    
                    terms = []
                    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
                    # A11=0, A22=3, B11=0, B22=3
                    # W0 (C00) += 1, W3 (C11) += 1
                    terms.append((0, 0, 0, 1.0))  # A11 * B11 -> C00
                    terms.append((3, 3, 3, 1.0))  # A22 * B22 -> C11
                    terms.append((0, 3, 0, 1.0))  # A11 * B22 -> C00
                    terms.append((3, 0, 3, 1.0))  # A22 * B11 -> C11
                    
                    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
                    # A21=2, A22=3, B11=0
                    # W1 (C10) += 1, W3 (C11) -= 1
                    terms.append((2, 0, 1, 1.0))  # A21 * B11 -> C10
                    terms.append((3, 0, 3, -1.0)) # A22 * B11 -> C11
                    # Wait, M2 is (A21+A22)*B11. 
                    # Expansion: A21*B11 + A22*B11.
                    # C21 gets A21*B11. C22 gets A22*B11? 
                    # Formula: C21 = M2 + M4 - M1? No.
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M6
                    # C22 = M1 - M2 + M3 + M6
                    
                    # Let's re-verify standard Strassen indices for C[i,j] where i is row of A, j is col of B.
                    # C[0,0] = M1 + M4 - M5 + M7
                    # C[0,1] = M3 + M5
                    # C[1,0] = M2 + M6
                    # C[1,1] = M1 - M2 + M3 + M6
                    
                    # Mapping to W indices (k*n+i):
                    # C[0,0] -> W[0]
                    # C[0,1] -> W[2]
                    # C[1,0] -> W[1]
                    # C[1,1] -> W[3]
                    
                    # M1 = (A11+A22)(B11+B22)
                    # Terms:
                    # A11*B11: U0, V0, W0 (C00), W3 (C11)
                    # A11*B22: U0, V3, W0 (C00), W3 (C11)
                    # A22*B11: U3, V0, W0 (C00), W3 (C11)
                    # A22*B22: U3, V3, W0 (C00), W3 (C11)
                    
                    # Let's just list the 7 intermediate products and their contributions.
                    # M1: (A11+A22)(B11+B22)
                    # M2: (A21+A22)B11
                    # M3: A11(B12-B22)
                    # M4: A22(B21-B11)
                    # M5: (A11+A12)B22
                    # M6: (A21-A11)(B11+B12)
                    # M7: (A12-A22)(B21+B22)
                    
                    # We create rank-1 terms for each expansion.
                    # Instead of expanding into scalar multiplications, we can just return the 7 terms if we treat the factors as linear combos.
                    # But the interface requires U, V, W matrices.
                    # So we must expand into the scalar basis.
                    
                    # Let's hardcode the 7 Strassen terms for 2x2x2 directly in the return list.
                    # Each term is a linear combination of U, V, W indices.
                    # Actually, the easiest way is to just use the 7 intermediate scalars as the "rank" dimension locally,
                    # but the global interface requires U, V, W to have rank R.
                    # So for the 2x2x2 subproblem, we generate 7 local rank slots.
                    # Then we map these to global indices.
                    
                    return [
                        # M1 = (A11+A22)(B11+B22)
                        # Local U: [1, 0, 0, 1] (A11+A22)
                        # Local V: [1, 0, 0, 1] (B11+B22)
                        # Local W: C00 gets 1, C11 gets 1. W[0]=1, W[3]=1.
                        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
                        
                        # M2 = (A21+A22)B11
                        # Local U: [0, 0, 1, 1]
                        # Local V: [1, 0, 0, 0]
                        # Local W: C10 gets 1, C11 gets -1. W[1]=1, W[3]=-1.
                        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 1, 0, -1]},
                        
                        # M3 = A11(B12-B22)
                        # Local U: [1, 0, 0, 0]
                        # Local V: [0, 1, 0, -1]
                        # Local W: C01 gets 1, C11 gets 1. W[2]=1, W[3]=1.
                        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 0, 1, 1]},
                        
                        # M4 = A22(B21-B11)
                        # Local U: [0, 0, 0, 1]
                        # Local V: [0, 0, 1, -1]
                        # Local W: C00 gets 1, C10 gets 1. W[0]=1, W[1]=1.
                        {'u': [0, 0, 0, 1], 'v': [0, 0, 1, -1], 'w': [1, 1, 0, 0]},
                        
                        # M5 = (A11+A12)B22
                        # Local U: [1, 1, 0, 0]
                        # Local V: [0, 0, 0, 1]
                        # Local W: C00 gets -1, C01 gets 1. W[0]=-1, W[2]=1.
                        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 0, 1, 0]},
                        
                        # M6 = (A21-A11)(B11+B12)
                        # Local U: [-1, 0, 1, 0]
                        # Local V: [1, 1, 0, 0]
                        # Local W: C11 gets 1. W[3]=1.
                        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
                        
                        # M7 = (A12-A22)(B21+B22)
                        # Local U: [0, 1, 0, -1]
                        # Local V: [0, 0, 1, 1]
                        # Local W: C00 gets 1. W[0]=1.
                        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
                    ]
                else:
                    # Fallback for non-2x2x2 square
                    raise NotImplementedError("Only 2x2x2 Strassen implemented")
            
            # If we can't split square, but P is even, we split P
            # This is just concatenation of two sub-problems.
            terms = []
            offset_v = 0
            offset_w = 0
            # Subproblem 1: n_sub x m_sub * m_sub x p1
            terms1 = get_strassen_decomposition(n_sub, m_sub, p1)
            for t in terms1:
                # Map local indices to global sub-problem indices
                # V index is local_v + offset_v
                # W index is local_w + offset_w
                terms.append({
                    'u': t['u'],
                    'v': t['v'],
                    'w': t['w'],
                    'v_offset': offset_v,
                    'w_offset': offset_w
                })
            
            # Subproblem 2: n_sub x m_sub * m_sub x p2
            offset_v += p1
            offset_w += p1 # Wait, W size is p*n. If we split P, the W indices for the second block are shifted by p1*n?
            # W index is k*n + i. If k is in the second half, k = k_local + p1.
            # So W_global = (k_local + p1)*n + i = k_local*n + i + p1*n.
            # So offset_w should be p1 * n_sub.
            offset_w = p1 * n_sub
            
            terms2 = get_strassen_decomposition(n_sub, m_sub, p2)
            for t in terms2:
                terms.append({
                    'u': t['u'],
                    'v': t['v'],
                    'w': t['w'],
                    'v_offset': offset_v,
                    'w_offset': offset_w
                })
            
            return terms
            
        # If we cannot split P, try splitting N?
        # For our specific case 2x2x4, the above P-split will trigger.
        # It will call get_strassen_decomposition(2, 2, 2) twice.
        # That will return the 7 terms each.
        # Total 14 terms.
        
        return None # Should not reach here for this specific problem

    # Get the decomposition for the full problem
    # n=2, m=2, p=4
    full_terms = get_strassen_decomposition(n, m, p)
    
    # Determine the rank (number of terms)
    rank = len(full_terms)
    
    # Initialize global U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Populate U, V, W
    for r, term in enumerate(full_terms):
        # term has 'u', 'v', 'w' vectors and potentially 'v_offset', 'w_offset'
        # 'u' is local to A (size n*m)
        # 'v' is local to B block (size m*p_block)
        # 'w' is local to C block (size p_block*n)
        
        u_vec = term['u']
        v_vec = term['v']
        w_vec = term['w']
        
        v_off = term.get('v_offset', 0)
        w_off = term.get('w_offset', 0)
        
        # Fill U
        for i in range(len(u_vec)):
            U[i, r] = u_vec[i]
            
        # Fill V
        for i in range(len(v_vec)):
            V[i + v_off, r] = v_vec[i]
            
        # Fill W
        for i in range(len(w_vec)):
            W[i + w_off, r] = w_vec[i]
            
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
