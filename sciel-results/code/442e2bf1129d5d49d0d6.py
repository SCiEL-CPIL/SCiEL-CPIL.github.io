import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110

    # Trivial rank 44 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # --- Phase 1: Find a valid rank-7 decomposition for 2x2x2 ---
    # We look for U, V, W of size 4x7 such that T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # This is equivalent to C = A @ B for 2x2 matrices.
    # We will generate candidate U and V matrices with entries in {-1, 0, 1}
    # and solve for W.
    
    valid_strassen = None
    # To limit search space, we can fix the first column of U and V to be non-zero
    # and use symmetry. But let's just brute force a subset.
    # Actually, we can be smarter. We know the rank of the 2x2x2 tensor is 7.
    # We can try to find U, V, W by random search + refinement or structured search.
    
    # Let's try a structured search based on the known Strassen factors but allowing variations.
    # Or better, just test the two candidates from the previous code but fix the verification.
    # The previous code likely failed because the W matrix was wrong or the verification was too strict/buggy.
    
    # Let's re-implement the verification correctly.
    # T_{i,j,k} = (A @ B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    def verify_strassen(U, V, W):
        # U: 4x7 (i*m+j), V: 4x7 (j*p+k), W: 4x7 (k*n+i)
        # For 2x2x2:
        # U index: 2i+j
        # V index: 2j+k
        # W index: 2k+i
        for _ in range(5):
            A = np.random.randint(-5, 5, (2, 2))
            B = np.random.randint(-5, 5, (2, 2))
            C_expected = A @ B
            
            C_calc = np.zeros((2, 2))
            for i in range(2):
                for k in range(2):
                    val = 0
                    for r in range(7):
                        for j in range(2):
                            a = i * 2 + j
                            b = j * 2 + k
                            c = k * 2 + i
                            val += U[a, r] * V[b, r] * W[c, r]
                    C_calc[i, k] = val
            if not np.allclose(C_calc, C_expected):
                return False
        return True

    # Candidate 1 from previous code
    U1 = np.array([
        [1, 0, 1, -1, 1, 0, -1], 
        [0, 0, 1, 0, 0, 1, 0],   
        [0, 1, 0, 0, 0, 0, 1],   
        [1, 1, 0, 1, 1, -1, 0]   
    ])
    V1 = np.array([
        [1, 1, 0, 1, 0, 0, 1],   
        [0, 0, 0, 1, 0, 0, 1],   
        [0, 0, 1, 0, 1, 1, 0],   
        [1, 0, -1, 0, 1, 1, 0]   
    ])
    W1 = np.array([
        [1, 0, 0, -1, 1, 0, -1], 
        [0, 0, 1, 1, 0, 0, 0],   
        [0, 1, 0, 0, -1, 1, 1],  
        [1, 0, -1, 1, 1, 0, 0]   
    ])
    
    if verify_strassen(U1, V1, W1):
        valid_strassen = (U1, V1, W1)
    else:
        # Candidate 2
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0], 
            [0, 0, 0, 0, 1, 0, 1],  
            [0, 1, 0, 0, 0, 1, 0],  
            [1, 1, 0, 1, 0, 0, -1]  
        ])
        V2 = np.array([
            [1, 1, 0, -1, 0, 1, 0], 
            [0, 0, 1, 0, 0, 1, 1],  
            [0, 0, 0, 1, 0, 0, 1],  
            [1, 0, -1, 0, 1, 0, 1]  
        ])
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, -1], 
            [0, 0, 1, 0, 1, 0, 0],   
            [0, 1, 0, 1, 0, 0, 0],   
            [1, 0, -1, 0, 0, 1, 1]   
        ])
        if verify_strassen(U2, V2, W2):
            valid_strassen = (U2, V2, W2)
        else:
            # Try to find a valid Strassen by random search
            # We know the rank is 7.
            # Let's try to solve for W given random U, V?
            # No, that's hard.
            # Let's try a different known Strassen variant.
            # Maybe the issue is just the specific numbers.
            # Let's try to construct one from scratch using the "standard" Strassen formulas
            # and derive W correctly.
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M6 + M7
            
            # Let's define U and V based on these.
            # U rows: A00, A01, A10, A11
            # V rows: B00, B01, B10, B11
            # W rows: C00, C01, C10, C11
            
            U3 = np.zeros((4, 7))
            V3 = np.zeros((4, 7))
            W3 = np.zeros((4, 7))
            
            # M1
            U3[0, 0] = 1; U3[3, 0] = 1
            V3[0, 0] = 1; V3[3, 0] = 1
            W3[0, 0] = 1; W3[3, 0] = 1 # C00 += M1, C11 += M1
            
            # M2
            U3[2, 1] = 1; U3[3, 1] = 1
            V3[0, 1] = 1
            W3[2, 1] = 1 # C10 += M2
            
            # M3
            U3[0, 2] = 1
            V3[1, 2] = 1; V3[3, 2] = -1
            W3[1, 2] = 1 # C01 += M3
            W3[3, 2] = -1 # C11 -= M3
            
            # M4
            U3[3, 3] = 1
            V3[2, 3] = 1; V3[0, 3] = -1
            W3[0, 3] = 1 # C00 += M4
            W3[2, 3] = 1 # C10 += M4
            
            # M5
            U3[0, 4] = 1; U3[1, 4] = 1
            V3[3, 4] = 1
            W3[0, 4] = -1 # C00 -= M5
            W3[1, 4] = 1 # C01 += M5
            
            # M6
            U3[2, 5] = 1; U3[0, 5] = -1
            V3[0, 5] = 1; V3[1, 5] = 1
            W3[3, 5] = 1 # C11 += M6
            
            # M7
            U3[1, 6] = 1; U3[3, 6] = -1
            V3[2, 6] = 1; V3[3, 6] = 1
            W3[0, 6] = -1 # C00 -= M7
            W3[3, 6] = 1 # C11 += M7
            
            if verify_strassen(U3, V3, W3):
                valid_strassen = (U3, V3, W3)

    if valid_strassen:
        U_s, V_s, W_s = valid_strassen
        # We have a rank 7 block for 2x2x2.
        # Now we need to cover 11 columns of B.
        # We can group columns into pairs (2x2x2) and singles (2x2x1).
        # Let x be the number of pairs, y be the number of singles.
        # 2x + y = 11
        # Rank = 7x + 2y (since 2x2x1 has rank 2? No, 2x2x1 is C_i,k = A_i0 B_0k + A_i1 B_1k.
        # This is rank 2. Terms: (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1).
        # Wait, for a single column k, C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
        # C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k}
        # This requires 4 terms in the trivial decomposition.
        # Can we do better for a single column?
        # The tensor for 2x2x1 is rank 2?
        # T_{i,j,k} = A_{i,j} B_{j,k}.
        # This is a 2x2 matrix of coefficients for each k? No.
        # The tensor is in V_i* (dim 2) x V_j* (dim 2) x V_k* (dim 1).
        # It's essentially the identity map on V_j*? No.
        # T_{i,j,k} = delta_{i,j}? No.
        # It's the tensor representing the bilinear map (A,B) -> C_k.
        # C_k is a 2x1 vector.
        # The rank of the 2x2x1 tensor is 2.
        # Proof: C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
        # C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k}
        # We can write this as sum of 2 rank-1 tensors?
        # Term 1: A_{0,0}B_{0,k} contributes to C_{0,k}.
        # Term 2: A_{0,1}B_{1,k} contributes to C_{0,k}.
        # Term 3: A_{1,0}B_{0,k} contributes to C_{1,k}.
        # Term 4: A_{1,1}B_{1,k} contributes to C_{1,k}.
        # These are 4 independent terms.
        # Is the rank 4?
        # Let's check the slice.
        # For fixed k, the matrix of coefficients is:
        # [[A00, A01], [A10, A11]] for B0k, B1k?
        # No, the tensor is T_{i,j,k}.
        # Slice k=0:
        # T_{0,0,0}=1, T_{0,1,0}=0, T_{1,0,0}=0, T_{1,1,0}=1?
        # No, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So T_{i,j,k} = 1 if the indices match the summation?
        # No, T_{i,j,k} is 1 if we are looking at the term A_{i,j} B_{j,k} contributing to C_{i,k}.
        # This is always 1 for the specific i,j,k combination?
        # Yes, the tensor T has T_{i,j,k} = 1 for all i,j,k?
        # No, that would be the trivial decomposition.
        # The tensor IS the multiplication tensor.
        # For 2x2x1, the tensor is:
        # T_{0,0,0} = 1 (A00 B00 -> C00)
        # T_{0,1,0} = 1 (A01 B10 -> C00)
        # T_{1,0,0} = 1 (A10 B00 -> C10)
        # T_{1,1,0} = 1 (A11 B10 -> C10)
        # Wait, C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
        # So T_{0,0,0} = 1, T_{0,1,0} = 1.
        # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}.
        # So T_{1,0,0} = 1, T_{1,1,0} = 1.
        # The matrix of T for k=0 is:
        # [[1, 1],
        #  [1, 1]]
        # This matrix has rank 1.
        # So the 2x2x1 tensor has rank 1?
        # No, the rank of a tensor is not the max rank of its slices.
        # But if all slices are rank 1 and consistent, it might be low.
        # Let's check if T can be written as sum of 1 rank-1 tensors.
        # u_i v_j w_k.
        # u0 v0 w0 = 1
        # u0 v1 w0 = 1
        # u1 v0 w0 = 1
        # u1 v1 w0 = 1
        # This implies u0=v0=w0=1, u0=v1=w0=1 => v1=1.
        # u1=v0=w0=1 => u1=1.
        # u1=v1=w0=1 => 1=1.
        # So yes, it is rank 1?
        # Wait, u_i v_j w_k.
        # If u=[1,1], v=[1,1], w=[1], then u_i v_j w_k = 1 for all i,j.
        # This matches T_{i,j,0}=1 for all i,j.
        # So the 2x2x1 tensor has rank 1!
        # This means we can handle a single column with rank 1.
        
        # So, if we use x pairs (rank 7 each) and y singles (rank 1 each):
        # 2x + y = 11
        # Rank = 7x + y
        # To minimize rank, we want to maximize x (since 7/2 = 3.5 < 1 is false?
        # Wait.
        # Cost per column for pair: 7/2 = 3.5
        # Cost per column for single: 1/1 = 1
        # So singles are much cheaper!
        # Why did I think pairs were better?
        # Because for 2x2x2, the trivial rank is 4. Strassen is 7.
        # 7 > 4. Strassen is WORSE than trivial for 2x2x2?
        # No, trivial 2x2x2 is rank 4.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # These are 4 independent terms.
        # So trivial rank for 2x2x2 is 4.
        # Strassen rank is 7.
        # So Strassen is NOT useful for 2x2x2! It increases the rank.
        
        # I made a huge mistake in the previous strategy.
        # Strassen is useful for 2x2x2?
        # No, Strassen reduces multiplications from 8 to 7.
        # The trivial decomposition has 8 terms?
        # Let's re-evaluate the trivial rank of 2x2x2.
        # T_{i,j,k}.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Terms:
        # (i=0, j=0, k=0): A00 B00 -> C00
        # (i=0, j=1, k=0): A01 B10 -> C00
        # (i=0, j=0, k=1): A00 B01 -> C01
        # (i=0, j=1, k=1): A01 B11 -> C01
        # (i=1, j=0, k=0): A10 B00 -> C10
        # (i=1, j=1, k=0): A11 B10 -> C10
        # (i=1, j=0, k=1): A10 B01 -> C11
        # (i=1, j=1, k=1): A11 B11 -> C11
        # These are 8 terms.
        # So trivial rank is 8.
        # Strassen rank is 7.
        # So Strassen saves 1 rank per 2x2x2 block.
        
        # Now back to 2x2x1.
        # Trivial rank is 4.
        # Can we do rank 2?
        # We found rank 1 above?
        # Let's re-verify.
        # T_{0,0,0}=1, T_{0,1,0}=1, T_{1,0,0}=1, T_{1,1,0}=1.
        # If T = u v w^T, then u0 v0 w0 = 1, u0 v1 w0 = 1, etc.
        # This works.
        # So rank of 2x2x1 is 1?
        # Wait, the tensor is T_{i,j,k}.
        # The output is C_{i,k}.
        # The term is A_{i,j} B_{j,k}.
        # For a fixed k, the mapping is (A,B) -> C_k.
        # C_k is a vector of size n=2.
        # The tensor slice is 2x2.
        # If the slice is all 1s, it is rank 1.
        # So yes, 2x2x1 has rank 1?
        # Let's check the definition of rank.
        # Rank 1 means T_{i,j,k} = u_i v_j w_k.
        # Here w_k is just a scalar.
        # So T_{i,j,k} = c_k u_i v_j.
        # If c_k=1, u=[1,1], v=[1,1], then T_{i,j,k}=1.
        # This matches.
        # So we can decompose a single column with rank 1.
        
        # So:
        # Pair (2 cols): Rank 7 (Strassen) or Rank 8 (Trivial).
        # Single (1 col): Rank 1.
        
        # We want to cover 11 columns.
        # Option A: 5 pairs + 1 single.
        # Rank = 5 * 7 + 1 * 1 = 36.
        # Option B: 4 pairs + 3 singles.
        # Rank = 4 * 7 + 3 * 1 = 31.
        # Option C: 0 pairs + 11 singles.
        # Rank = 11 * 1 = 11.
        
        # Is rank 11 possible?
        # If we use 11 rank-1 tensors, each handling one column.
        # For column k, we use 1 rank-1 tensor.
        # Total rank 11.
        # This is much better than 44.
        
        # Let's verify the rank 1 decomposition for 2x2x1.
        # U: 4x1. V: 4x1. W: 2x1.
        # U_{i,j} = 1 for all i,j.
        # V_{j,k} = 1 for all j. (k is fixed)
        # W_{k,i} = 1 for all i.
        
        # Let's construct the full decomposition with rank 11.
        
        rank_new = 11
        U_full = np.zeros((n * m, rank_new))
        V_full = np.zeros((m * p, rank_new))
        W_full = np.zeros((p * n, rank_new))
        
        for k in range(p):
            r = k
            # U: 1 for all i,j
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j, r] = 1
            # V: 1 for all j
            for j in range(m):
                V_full[j * p + k, r] = 1
            # W: 1 for all i
            for i in range(n):
                W_full[k * n + i, r] = 1
        
        # Verify this rank 11 solution
        def verify_full(U, V, W):
            for _ in range(10):
                A = np.random.randint(-5, 5, (n, m))
                B = np.random.randint(-5, 5, (m, p))
                C_expected = A @ B
                
                C_calc = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(U.shape[1]):
                            for j in range(m):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val += U[a, r] * V[b, r] * W[c, r]
                        C_calc[i, k] = val
                if not np.allclose(C_calc, C_expected):
                    return False
            return True

        if verify_full(U_full, V_full, W_full):
            best_rank = rank_new
            best_U = U_full
            best_V = V_full
            best_W = W_full

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
