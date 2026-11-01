import numpy as np
import time
import itertools

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n+i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank14_strassen():
    """Build a rank-14 decomposition using Strassen for each 2x2 block.
    
    For 2x2*2x4, we split into two 2x2*2x2 subproblems.
    U: (4, rank), V: (8, rank), W: (8, rank)
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Standard Strassen 2x2*2x2 patterns
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_patterns = [
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
        ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
        ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
        ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
        ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (1, 3), (1, 1)),  # M7
    ]
    
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    # First 2x2 block (columns 0-6)
    for r in range(7):
        a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
        for idx, a_idx in enumerate(a_sup):
            U[a_idx][r] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V[b_idx][r] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W[c_idx][r] = c_sgn[idx]
    
    # Second 2x2 block (columns 7-13)
    for r in range(7):
        a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
        for idx, a_idx in enumerate(a_sup):
            U[a_idx][r + 7] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V[4 + b_idx][r + 7] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W[4 + c_idx][r + 7] = c_sgn[idx]
    
    return U, V, W

def _algebraic_rank_reduction(n, m, p, deadline):
    """
    Structural algebraic search for rank 13.
    
    Strategy:
    1. Start with the known rank-14 solution.
    2. The set of all rank-14 decompositions forms an affine variety.
    3. We look for a "singularity" in this variety where the rank drops to 13.
    4. This happens if there exists a linear dependence among the rank-1 terms
       that allows us to express one term as a linear combination of others
       in the tensor product space, effectively reducing the count.
    
    Specifically, we attempt to find coefficients c_r such that:
    sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    but the vectors U_r, V_r, W_r are linearly independent in a way that
    allows merging.
    
    A more direct structural attack:
    We know rank(2x2x2) = 7. The tensor for 2x2x4 is essentially two 2x2x2 tensors
    plus potential cross-terms.
    If the rank is 13, it implies a "shared" structure between the two blocks
    that saves 1 rank compared to 7+7=14.
    
    We model this by introducing a shared rank-1 term that participates in both blocks.
    Let the decomposition be:
    T = sum_{r=1}^{6} (U1_r ⊗ V1_r ⊗ W1_r)  # Block 1 core
      + sum_{r=1}^{6} (U2_r ⊗ V2_r ⊗ W2_r)  # Block 2 core
      + (U_shared ⊗ V_shared ⊗ W_shared)      # Shared term
    
    Total rank = 6 + 6 + 1 = 13.
    
    We fix the 6 core terms of Block 1 to be standard Strassen (minus one term, say M7).
    We fix the 6 core terms of Block 2 to be standard Strassen (minus one term, say M7).
    Then we solve for the Shared term such that:
    T_total = T_strassen_block1_full + T_strassen_block2_full
             - T_strassen_M1_block1 - T_strassen_M2_block2
             + T_shared
    
    This implies:
    T_shared = T_strassen_M1_block1 + T_strassen_M2_block2
    
    For T_shared to be a rank-1 tensor, the sum of these two specific Strassen terms
    (mapped to the 2x2x4 space) must itself be rank-1.
    
    Let's check if M1 and M2 from Strassen can be aligned to form a rank-1 tensor
    in the 2x2x4 space.
    
    M1 in Block 1 (indices 0..3 for B, 0..3 for C):
    A: (0,3) -> (1,1)
    B: (0,3) -> (1,1)
    C: (0,1,2,3) -> (1,-1,1,1)
    
    M2 in Block 2 (indices 4..7 for B, 4..7 for C):
    A: (2,3) -> (1,1)
    B: (4) -> (1)  # B index 0 of block 2 is global index 4
    C: (6,7) -> (1,1) # C indices 2,3 of block 2 are global 6,7
    
    Let's construct the tensor T_candidate = T_M1_block1 + T_M2_block2.
    If this tensor is rank-1, we have a rank-13 solution.
    """
    
    # Construct T_M1_block1
    # A indices: 0, 3. B indices: 0, 3. C indices: 0, 1, 2, 3.
    # T[i,j,k] = U[i] V[j] W[k]
    
    T = np.zeros((n, m, p))
    
    # Add M1 contribution for Block 1 (k in 0..3)
    # A = [[1,0],[0,1]] (indices 0,3) -> U_vec = [1, 0, 0, 1]
    # B = [[1,0],[0,1]] (indices 0,3) -> V_vec = [1, 0, 0, 1]
    # C = [[1,-1],[1,1]] (indices 0,1,2,3) -> W_vec = [1, -1, 1, 1]
    
    U_m1 = np.array([1, 0, 0, 1])
    V_m1 = np.array([1, 0, 0, 1])
    W_m1 = np.array([1, -1, 1, 1])
    
    for i in range(n):
        for j in range(m):
            for k in range(4):
                T[i, j, k] += U_m1[i*m+j] * V_m1[j*p+k] * W_m1[k*n+i]
    
    # Add M2 contribution for Block 2 (k in 4..7)
    # A = [[0,0],[1,1]] (indices 2,3) -> U_vec = [0, 0, 1, 1]
    # B = [[1,0],[0,0]] (index 0 of block 2 is global 4) -> V_vec = [0,0,0,0, 1,0,0,0]
    # C = [[0,0],[1,1]] (indices 2,3 of block 2 are global 6,7) -> W_vec = [0,0,0,0, 0,0,1,1]
    
    U_m2 = np.array([0, 0, 1, 1])
    V_m2 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    W_m2 = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                T[i, j, k] += U_m2[i*m+j] * V_m2[j*p+k] * W_m2[k*n+i]
                
    # Check if T is rank 1
    # Reshape T to (n*m, p) or similar? No, check if T[i,j,k] = u_i v_j w_k
    # A tensor is rank 1 if it can be written as outer product of 3 vectors.
    # We can check this by flattening and checking SVD, or by checking if
    # T[i,j,k] * T[i',j',k'] == T[i,j',k'] * T[i',j,k] for all indices.
    
    is_rank1 = True
    # Pick a pivot non-zero element
    pivot_found = False
    pi, pj, pk = 0, 0, 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k]) > 1e-9:
                    pi, pj, pk = i, j, k
                    pivot_found = True
                    break
            if pivot_found: break
        if pivot_found: break
    
    if not pivot_found:
        return None # Zero tensor, not useful
    
    pivot_val = T[pi, pj, pk]
    
    # Define u_i = T[i, pj, pk] / pivot_val
    # Define v_j = T[pi, j, pk] / pivot_val
    # Define w_k = T[pi, pj, k] / pivot_val
    # Check if T[i,j,k] == u_i * v_j * w_k * pivot_val
    
    u = np.zeros(n)
    v = np.zeros(m)
    w = np.zeros(p)
    
    u[pi] = 1.0
    v[pj] = 1.0
    w[pk] = 1.0
    
    for i in range(n):
        u[i] = T[i, pj, pk] / pivot_val
    for j in range(m):
        v[j] = T[pi, j, pk] / pivot_val
    for k in range(p):
        w[k] = T[pi, pj, k] / pivot_val
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = u[i] * v[j] * w[k] * pivot_val
                if abs(T[i,j,k] - expected) > 1e-9:
                    is_rank1 = False
                    break
            if not is_rank1: break
        if not is_rank1: break
        
    if is_rank1:
        # We found a rank-1 shared term!
        # Construct the full rank-13 decomposition.
        # U, V, W will have 13 columns.
        # Cols 0-5: Strassen M1-M6 for Block 1
        # Cols 6-11: Strassen M1-M6 for Block 2
        # Col 12: Shared term (u, v, w)
        
        strassen_patterns = [
            ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
            ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
            ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
            ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
            ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
            ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ]
        
        U = [[0] * 13 for _ in range(n*m)]
        V = [[0] * 13 for _ in range(m*p)]
        W = [[0] * 13 for _ in range(p*n)]
        
        # Fill Block 1 (cols 0-5)
        for r in range(6):
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
            for idx, a_idx in enumerate(a_sup):
                U[a_idx][r] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V[b_idx][r] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W[c_idx][r] = c_sgn[idx]
                
        # Fill Block 2 (cols 6-11)
        for r in range(6):
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
            for idx, a_idx in enumerate(a_sup):
                U[a_idx][r + 6] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V[4 + b_idx][r + 6] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W[4 + c_idx][r + 6] = c_sgn[idx]
                
        # Fill Shared Term (col 12)
        # u, v, w are the vectors we found.
        # Note: The shared term was T_M1_block1 + T_M2_block2.
        # In the decomposition T = Sum(Core1) + Sum(Core2) + Shared,
        # we removed M1 from Core1 and M2 from Core2.
        # So we add the Shared term which accounts for both.
        
        for i in range(n*m):
            U[i][12] = u[i//m][i%m] if False else u[i//m] * (1 if i%m==0 else 1) # Wait, u is defined on i
            # u[i] corresponds to A[i]
            U[i][12] = u[i//m] if (i%m)==0 else 0 # No, u is vector of length n.
            # My u vector is length n. U is length n*m.
            # In the rank-1 tensor T[i,j,k] = u_i v_j w_k.
            # U[i*m+j, r] should be u_i.
            # V[j*p+k, r] should be v_j.
            # W[k*n+i, r] should be w_k.
            
        # Let's re-assign correctly
        for i in range(n*m):
            row_i = i // m
            U[row_i * m + (i % m)][12] = u[row_i] # This is wrong indexing logic above, let's do it cleanly
            
        U = [[0] * 13 for _ in range(n*m)]
        V = [[0] * 13 for _ in range(m*p)]
        W = [[0] * 13 for _ in range(p*n)]
        
        # Refill Strassen parts
        for r in range(6):
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
            for idx, a_idx in enumerate(a_sup):
                U[a_idx][r] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V[b_idx][r] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W[c_idx][r] = c_sgn[idx]
                
        for r in range(6):
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
            for idx, a_idx in enumerate(a_sup):
                U[a_idx][r + 6] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V[4 + b_idx][r + 6] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W[4 + c_idx][r + 6] = c_sgn[idx]
                
        # Assign Shared Term
        for i in range(n):
            for j in range(m):
                U[i*m+j][12] = u[i]
        for j in range(m):
            for k in range(p):
                V[j*p+k][12] = v[j]
        for k in range(p):
            for i in range(n):
                W[k*n+i][12] = w[k]
                
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    return None

def _search_rank13_via_subspace(n, m, p, deadline):
    """
    Generalized search:
    Try to remove term M_r from Block 1 and M_s from Block 2.
    Check if T_Mr_B1 + T_Ms_B2 is rank-1.
    """
    strassen_patterns = [
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
        ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
        ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
        ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
        ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (1, 3), (1, 1)),  # M7
    ]
    
    # We will try to merge any pair (r1, r2) where r1 is from block 1 and r2 is from block 2.
    # There are 7*7 = 49 pairs.
    
    for r1 in range(7):
        for r2 in range(7):
            if time.time() > deadline:
                return None
                
            T = np.zeros((n, m, p))
            
            # Add term r1 from Block 1
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r1]
            U_vec = np.zeros(n*m)
            V_vec = np.zeros(m*p)
            W_vec = np.zeros(p*n)
            for idx, a_idx in enumerate(a_sup):
                U_vec[a_idx] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V_vec[b_idx] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W_vec[c_idx] = c_sgn[idx]
                
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_vec[i*m+j] * V_vec[j*p+k] * W_vec[k*n+i]
                        
            # Add term r2 from Block 2
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r2]
            U_vec2 = np.zeros(n*m)
            V_vec2 = np.zeros(m*p)
            W_vec2 = np.zeros(p*n)
            for idx, a_idx in enumerate(a_sup):
                U_vec2[a_idx] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V_vec2[4 + b_idx] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W_vec2[4 + c_idx] = c_sgn[idx]
                
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_vec2[i*m+j] * V_vec2[j*p+k] * W_vec2[k*n+i]
                        
            # Check if T is rank 1
            # Find pivot
            pivot_found = False
            pi, pj, pk = 0, 0, 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T[i,j,k]) > 1e-9:
                            pi, pj, pk = i, j, k
                            pivot_found = True
                            break
                    if pivot_found: break
                if pivot_found: break
                
            if not pivot_found:
                continue
                
            pivot_val = T[pi, pj, pk]
            
            u = np.zeros(n)
            v = np.zeros(m)
            w = np.zeros(p)
            
            for i in range(n):
                u[i] = T[i, pj, pk] / pivot_val
            for j in range(m):
                v[j] = T[pi, j, pk] / pivot_val
            for k in range(p):
                w[k] = T[pi, pj, k] / pivot_val
                
            is_rank1 = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = u[i] * v[j] * w[k] * pivot_val
                        if abs(T[i,j,k] - expected) > 1e-9:
                            is_rank1 = False
                            break
                    if not is_rank1: break
                if not is_rank1: break
                
            if is_rank1:
                # Construct solution
                # Cols 0-6: Block 1 Strassen, EXCEPT r1
                # Cols 7-13: Block 2 Strassen, EXCEPT r2
                # Col 14? No, we removed 2 and added 1, so total rank = 7 + 7 - 2 + 1 = 13.
                
                U = [[0] * 13 for _ in range(n*m)]
                V = [[0] * 13 for _ in range(m*p)]
                W = [[0] * 13 for _ in range(p*n)]
                
                col = 0
                # Fill Block 1
                for r in range(7):
                    if r == r1: continue
                    a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
                    for idx, a_idx in enumerate(a_sup):
                        U[a_idx][col] = a_sgn[idx]
                    for idx, b_idx in enumerate(b_sup):
                        V[b_idx][col] = b_sgn[idx]
                    for idx, c_idx in enumerate(c_sup):
                        W[c_idx][col] = c_sgn[idx]
                    col += 1
                    
                # Fill Block 2
                for r in range(7):
                    if r == r2: continue
                    a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
                    for idx, a_idx in enumerate(a_sup):
                        U[a_idx][col] = a_sgn[idx]
                    for idx, b_idx in enumerate(b_sup):
                        V[4 + b_idx][col] = b_sgn[idx]
                    for idx, c_idx in enumerate(c_sup):
                        W[4 + c_idx][col] = c_sgn[idx]
                    col += 1
                    
                # Fill Shared Term
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = u[i]
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][col] = v[j]
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][col] = w[k]
                        
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
    return None

def _fallback_rank16():
    """Absolute fallback: rank-16 schoolbook decomposition."""
    n, m, p = 2, 2, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Guaranteed fallback: rank-14 Strassen construction
    U14, V14, W14 = _build_rank14_strassen()
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Verify fallback
    if not _verify(U14, V14, W14, n, m, p):
        best_result = _fallback_rank16()
        best_rank = 16
        return best_result
    
    # Try structural algebraic rank-13 search
    if time.time() < deadline - 10:
        result = _search_rank13_via_subspace(n, m, p, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res)
