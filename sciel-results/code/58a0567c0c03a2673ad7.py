import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition U, V, W exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Convention:
    A is n x m. B is m x p. C is n x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    Tensor T[i, j, k] = 1 for all i, j, k.
    Sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == 1
    """
    rank = U.shape[1]
    # For speed, we can vectorize the inner product check if needed, 
    # but for small dimensions, loops are fine.
    # However, to be safe against time limits, we'll check all.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = 2, 3, 10
    
    # We split the 2x3x10 problem into two 2x3x5 problems.
    # The tensor T_{2,3,10} is the direct sum of two T_{2,3,5} tensors 
    # corresponding to k in [0,5) and k in [5,10).
    # Rank(T_{2,3,10}) <= 2 * Rank(T_{2,3,5}).
    
    # We need to find a low-rank decomposition for T_{2,3,5}.
    # Schoolbook rank for 2x3x5 is 30.
    # We will construct a rank 15 decomposition for 2x3x5.
    
    # Let's define the indices for 2x3x5:
    # i in {0, 1}
    # j in {0, 1, 2}
    # k in {0, 1, 2, 3, 4}
    
    # A is 2x3:
    # a00 a01 a02
    # a10 a11 a12
    
    # B is 3x5:
    # b00 b01 b02 b03 b04
    # b10 b11 b12 b13 b14
    # b20 b21 b22 b23 b24
    
    # C is 2x5:
    # c00 c01 c02 c03 c04
    # c10 c11 c12 c13 c14
    
    # We want to compute C = A * B with fewer than 30 multiplications.
    # Standard Strassen works for 2x2.
    # For 2x3 and 3x5, we can partition A into 2x2 and 2x1 blocks?
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (3x5) ... wait, B is 3x5.
    # Let's partition B into B_top (2x5) and B_bottom (1x5)? No, A is 2x3, so B must be 3x5.
    # A * B = [A_11 A_12 A_13; A_21 A_22 A_23] * [B_1; B_2; B_3] where B_i are 1x5 rows.
    
    # Let's try to reduce the rank for the 2x3x5 case.
    # Known results: R(2,3,5) is not widely known to be 15, but let's try to construct one.
    # Actually, R(2,3,5) <= R(2,2,5) + R(1,1,5) + ...
    # R(2,2,5) <= 5 * R(2,2,2) = 5 * 7 = 35? No, that's not additive like that.
    # R(2,2,p) = 3p + 2 for p >= 2? No.
    # For 2x2x2, rank is 7.
    # For 2x2x5, we can view it as 2x2x2 + 2x2x2 + 2x2x1.
    # Rank(2x2x2)=7. Rank(2x2x1)=4.
    # Total = 7+7+4 = 18.
    # Then we have the 2x1x5 part?
    # A = [A_2x2, A_2x1].
    # C = A_2x2 * B_2x5 + A_2x1 * B_1x5.
    # We need a decomposition for 2x2x5 and 2x1x5.
    # R(2,2,5) <= 18 (as above).
    # R(2,1,5) = 10 (schoolbook, since 1x1 multiplication is trivial, it's just 2*5=10 scalar mults).
    # Total 18 + 10 = 28. Still close to 30.
    
    # Let's look for a better 2x2x5 decomposition.
    # Strassen for 2x2x2 uses 7 mults.
    # For 2x2x5, we can use 5 columns of B.
    # C = A * B.
    # Using Strassen on the 2x2 block of A and B?
    # B is 2x5. We can split B into two 2x2 blocks and one 2x1 block.
    # B = [B_1 (2x2), B_2 (2x2), B_3 (2x1)].
    # C_1 = A * B_1 (2x2 result). Uses 7 mults.
    # C_2 = A * B_2 (2x2 result). Uses 7 mults.
    # C_3 = A * B_3 (2x1 result). Uses 4 mults.
    # Total 18.
    
    # Can we do better for 2x2x5?
    # It is known that R(2,2,p) = 3p + 2 for p even? No.
    # Actually, R(2,2,5) is likely 16 or 17.
    # Let's assume we can get 16.
    # Then R(2,3,5) <= 16 + 10 = 26.
    # Then R(2,3,10) <= 2 * 26 = 52.
    # Score = 60/52 = 1.15.
    
    # Let's try to construct a specific 2x2x5 decomposition with rank 16.
    # Or maybe just use the 7+7+4=18 one, which gives rank 52.
    # 60/52 = 1.1538.
    # Current score is 1.0 (rank 60).
    # So even rank 52 is an improvement.
    
    # Let's refine the 2x2x5 decomposition.
    # A = [[a00, a01], [a10, a11]]
    # B_1 = [[b00, b01], [b10, b11]]
    # Strassen products:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b00-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    
    # For B_2 (columns 2,3):
    # m8..m14 similar.
    
    # For B_3 (column 4):
    # c04 = a00*b04 + a01*b14
    # c14 = a10*b04 + a11*b14
    # 4 mults.
    
    # Total 18 mults for 2x2x5.
    
    # Now for the 2x1x5 part (A_right is 2x1, B_bottom is 1x5).
    # A_r = [[a02], [a12]]
    # B_b = [b20, b21, b22, b23, b24]
    # C_r = A_r * B_b
    # c00 += a02*b20, c10 += a12*b20, etc.
    # This requires 2*5 = 10 mults.
    # Can we share?
    # The mults are a02*b2k and a12*b2k.
    # No obvious sharing with the 2x2 part unless we mix columns.
    
    # Total rank for 2x3x5 = 18 + 10 = 28.
    # Total rank for 2x3x10 = 2 * 28 = 56.
    # Score = 60/56 = 1.0714.
    
    # Can we do better?
    # What if we use a different split for 2x3x5?
    # Split A into 1x3 and 1x3? No, A is 2x3.
    # Split B into 3x2, 3x2, 3x1?
    # A * B_1 (2x2), A * B_2 (2x2), A * B_3 (2x1).
    # R(2,3,2) + R(2,3,2) + R(2,3,1).
    # R(2,3,1) = 6.
    # R(2,3,2)?
    # A is 2x3, B is 3x2.
    # Split A into 2x2 and 2x1.
    # R(2,2,2) + R(2,1,2) = 7 + 4 = 11.
    # So R(2,3,2) <= 11.
    # Total = 11 + 11 + 6 = 28.
    # Same result.
    
    # Is there a way to get R(2,3,2) < 11?
    # R(2,3,2) is the rank of the 2x3x2 tensor.
    # It is known that R(2,3,2) = 10? Or 11?
    # Let's check R(2,3,2).
    # If R(2,3,2) = 10, then total is 10+10+6 = 26.
    # Then R(2,3,10) = 52. Score 1.15.
    
    # Let's try to construct a rank 10 decomposition for 2x3x2.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    # C = [[c00, c01], [c10, c11]]
    
    # We can use Strassen on the 2x2 submatrix of A and B?
    # Let A = [A_2x2, A_2x1].
    # C = A_2x2 * B_2x2 + A_2x1 * B_1x2.
    # R(2,2,2) = 7.
    # R(2,1,2) = 4.
    # Total 11.
    
    # Can we merge?
    # The 4 mults for the second part are:
    # a02*b20, a02*b21, a12*b20, a12*b21.
    # The 7 mults for the first part involve a00,a01,a10,a11 and b00,b01,b10,b11.
    # No overlap in variables, so ranks add.
    # So R(2,3,2) <= 11.
    # Is it 10?
    # Lower bound for R(2,3,2) is max(2*3, 3*2, 2*2) = 6? No, rank of matrix mult tensor is at least n*m, m*p, n*p.
    # Actually, R(n,m,p) >= max(nm, mp, np).
    # For 2,3,2: max(6, 6, 4) = 6.
    # But Strassen shows 7 for 2,2,2.
    # So 11 is a strong upper bound.
    
    # Let's stick with the 56 rank solution first, as it's safer to implement correctly.
    # Then try to optimize if time permits.
    
    # Implementation of Rank 56 Decomposition for 2x3x10.
    # We will build two 2x3x5 decompositions, one for k=0..4 and one for k=5..9.
    # Each 2x3x5 decomposition has rank 28.
    # Total rank 56.
    
    def build_2x3x5_decomp(offset_k):
        """
        Builds U, V, W for a 2x3x5 block, where k indices are offset by offset_k.
        Returns U (6x28), V (15x28), W (10x28).
        Note: W size is p*n = 10*2 = 20? No, for the sub-problem p=5, so W is 5*2=10.
        But in the full problem, W is 20x56.
        The rows of W for this block will be at indices:
        k in [offset_k, offset_k+5).
        i in [0, 2).
        Index in full W: k_full * 2 + i.
        """
        n2, m2, p2 = 2, 3, 5
        rank2 = 28
        U_sub = np.zeros((n2 * m2, rank2))
        V_sub = np.zeros((m2 * p2, rank2))
        W_sub = np.zeros((p2 * n2, rank2))
        
        # Part 1: 2x2x5 part (Rank 18)
        # A_part1 is 2x2 (cols 0,1 of A)
        # B_part1 is 2x5 (rows 0,1 of B)
        # This computes C_part1 = A_part1 * B_part1 (2x5)
        
        # Strassen for 2x2x2 blocks of B.
        # B is 2x5. Split into B1 (2x2), B2 (2x2), B3 (2x1).
        
        # B1 (cols 0,1 of B_sub, which are cols offset_k, offset_k+1 of B_full)
        # Mults 1-7
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b00-b10)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b10+b11)
        
        # Indices in U_sub:
        # a00 -> i=0, j=0 -> 0
        # a01 -> i=0, j=1 -> 1
        # a10 -> i=1, j=0 -> 3
        # a11 -> i=1, j=1 -> 4
        
        # Indices in V_sub:
        # b00 -> j=0, k=0 -> 0
        # b01 -> j=0, k=1 -> 1
        # b10 -> j=1, k=0 -> 5
        # b11 -> j=1, k=1 -> 6
        
        # Indices in W_sub:
        # c00 -> k=0, i=0 -> 0
        # c01 -> k=1, i=0 -> 2
        # c10 -> k=0, i=1 -> 1
        # c11 -> k=1, i=1 -> 3
        
        # Let's define the 7 Strassen terms for a generic 2x2x2 block.
        # We'll create a helper to add these terms.
        
        def add_strassen_2x2x2(u_offset, v_offset, w_offset, b_col_offset):
            # u_offset: row offset in U_sub for the 2x2 block of A (always 0 for first 2 cols)
            # v_offset: row offset in V_sub for the 2x2 block of B (rows 0,1)
            # w_offset: row offset in W_sub for the 2x2 block of C (cols 0,1)
            # b_col_offset: column offset in B_sub for this 2x2 block
            
            # We need to return the column indices used.
            cols = []
            
            # m1 = (a00+a11)(b00+b11)
            # U: a00(+1), a11(+1)
            # V: b00(+1), b11(+1)
            # W: c00(+1), c01(-1), c10(-1), c11(+1) ? 
            # Strassen:
            # c00 = m1 - m4 - m5 + m7
            # c01 = m5 - m1
            # c10 = m1 - m2
            # c11 = m2 - m3 + m4 - m6
            # Wait, standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B11-B21)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B21)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1 - M4 - M5 + M7
            # C12 = M5 - M1
            # C21 = M1 - M2
            # C22 = M2 - M3 + M4 - M6
            
            # Let's map variables:
            # A11=a00, A12=a01, A21=a10, A22=a11
            # B11=b00, B12=b01, B21=b10, B22=b11
            
            # Term 1: M1
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1  # a00
            U_sub[4, r] += 1  # a11
            V_sub[0, r] += 1  # b00
            V_sub[5, r] += 1  # b10? No, b10 is B21. B11 is b00, B21 is b10.
            # V indices: j*p + k.
            # j=0, k=0 -> 0 (b00)
            # j=1, k=0 -> 5 (b10)
            # j=0, k=1 -> 1 (b01)
            # j=1, k=1 -> 6 (b11)
            
            # Correction:
            # B11 = b00 (j=0, k=0) -> V[0]
            # B12 = b01 (j=0, k=1) -> V[1]
            # B21 = b10 (j=1, k=0) -> V[5]
            # B22 = b11 (j=1, k=1) -> V[6]
            
            # M1 = (a00+a11)(b00+b11)
            # U: a00(0), a11(4)
            # V: b00(0), b11(6)
            # W: c00(0), c01(2), c10(1), c11(3)
            # C11 (c00) += M1
            # C12 (c01) -= M1
            # C21 (c10) += M1
            # C22 (c11) -= M1 ? No.
            # C11 = M1 - M4 - M5 + M7
            # C12 = M5 - M1
            # C21 = M1 - M2
            # C22 = M2 - M3 + M4 - M6
            
            # So for M1:
            # W[0] (c00) += 1
            # W[2] (c01) -= 1
            # W[1] (c10) += 1
            # W[3] (c11) -= 1 ? No, C22 doesn't have M1.
            # Wait, C22 = M2 - M3 + M4 - M6. No M1.
            # So W[3] is 0 for M1.
            
            W_sub[0, r] += 1
            W_sub[2, r] -= 1
            W_sub[1, r] += 1
            W_sub[3, r] += 0
            
            # Term 2: M2 = (a10+a11)b00
            r = len(cols)
            cols.append(r)
            U_sub[3, r] += 1 # a10
            U_sub[4, r] += 1 # a11
            V_sub[0, r] += 1 # b00
            # C21 (c10) -= M2
            # C22 (c11) += M2
            W_sub[1, r] -= 1
            W_sub[3, r] += 1
            
            # Term 3: M3 = a00(b10-b11)
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1 # a00
            V_sub[5, r] += 1 # b10
            V_sub[6, r] -= 1 # b11
            # C22 (c11) -= M3
            W_sub[3, r] -= 1
            
            # Term 4: M4 = a11(b00-b10)
            r = len(cols)
            cols.append(r)
            U_sub[4, r] += 1 # a11
            V_sub[0, r] += 1 # b00
            V_sub[5, r] -= 1 # b10
            # C11 (c00) -= M4
            # C22 (c11) += M4
            W_sub[0, r] -= 1
            W_sub[3, r] += 1
            
            # Term 5: M5 = (a00+a01)b11
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1 # a00
            U_sub[1, r] += 1 # a01
            V_sub[6, r] += 1 # b11
            # C11 (c00) -= M5
            # C12 (c01) += M5
            W_sub[0, r] -= 1
            W_sub[2, r] += 1
            
            # Term 6: M6 = (a10-a00)(b00+b10)
            r = len(cols)
            cols.append(r)
            U_sub[3, r] += 1 # a10
            U_sub[0, r] -= 1 # a00
            V_sub[0, r] += 1 # b00
            V_sub[5, r] += 1 # b10
            # C22 (c11) -= M6
            W_sub[3, r] -= 1
            
            # Term 7: M7 = (a01-a11)(b10+b11)
            r = len(cols)
            cols.append(r)
            U_sub[1, r] += 1 # a01
            U_sub[4, r] -= 1 # a11
            V_sub[5, r] += 1 # b10
            V_sub[6, r] += 1 # b11
            # C11 (c00) += M7
            W_sub[0, r] += 1
            
            return cols
        
        # We need to handle the offsets correctly.
        # For 2x2x5, B has 5 columns.
        # B1: cols 0,1. V offset 0. W offset 0.
        # B2: cols 2,3. V offset 2 (since p=5, j=0 -> 0,1,2,3,4; j=1 -> 5,6,7,8,9).
        # Wait, V index is j*p + k.
        # For B1 (k=0,1): V indices 0,1,5,6.
        # For B2 (k=2,3): V indices 2,3,7,8.
        # For B3 (k=4): V indices 4,9.
        
        # W offset for B1 (k=0,1): W indices 0,1,2,3.
        # W offset for B2 (k=2,3): W indices 4,5,6,7.
        # W offset for B3 (k=4): W indices 8,9.
        
        # U is always the same 2x2 block of A (cols 0,1).
        
        # Add B1
        add_strassen_2x2x2(0, 0, 0, 0)
        
        # Add B2
        # We need to modify add_strassen to accept offsets for V and W.
        # My current add_strassen uses hardcoded indices. Let's rewrite it to be more flexible or just call it with logic.
        
        # Let's redefine add_strassen to take v_base and w_base.
        def add_strassen_offset(v_base, w_base):
            # v_base is the index of b00 in V_sub.
            # w_base is the index of c00 in W_sub.
            
            cols = []
            
            # M1
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1
            U_sub[4, r] += 1
            V_sub[v_base, r] += 1       # b00
            V_sub[v_base + 5, r] += 1   # b11 (j=1, k=1 -> 1*5+1=6. v_base+5? No. v_base is j=0,k=0. b11 is j=1,k=1. Diff is 5+1-0=6? No.
            # v_base = 0*5+0 = 0. b11 = 1*5+1 = 6. Diff 6.
            # v_base + 6?
            # Let's just calculate indices explicitly.
            v_b00 = v_base
            v_b01 = v_base + 1
            v_b10 = v_base + 5
            v_b11 = v_base + 6
            
            w_c00 = w_base
            w_c01 = w_base + 2
            w_c10 = w_base + 1
            w_c11 = w_base + 3
            
            U_sub[0, r] += 1
            U_sub[4, r] += 1
            V_sub[v_b00, r] += 1
            V_sub[v_b11, r] += 1
            W_sub[w_c00, r] += 1
            W_sub[w_c01, r] -= 1
            W_sub[w_c10, r] += 1
            
            # M2
            r = len(cols)
            cols.append(r)
            U_sub[3, r] += 1
            U_sub[4, r] += 1
            V_sub[v_b00, r] += 1
            W_sub[w_c10, r] -= 1
            W_sub[w_c11, r] += 1
            
            # M3
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1
            V_sub[v_b10, r] += 1
            V_sub[v_b11, r] -= 1
            W_sub[w_c11, r] -= 1
            
            # M4
            r = len(cols)
            cols.append(r)
            U_sub[4, r] += 1
            V_sub[v_b00, r] += 1
            V_sub[v_b10, r] -= 1
            W_sub[w_c00, r] -= 1
            W_sub[w_c11, r] += 1
            
            # M5
            r = len(cols)
            cols.append(r)
            U_sub[0, r] += 1
            U_sub[1, r] += 1
            V_sub[v_b11, r] += 1
            W_sub[w_c00, r] -= 1
            W_sub[w_c01, r] += 1
            
            # M6
            r = len(cols)
            cols.append(r)
            U_sub[3, r] += 1
            U_sub[0, r] -= 1
            V_sub[v_b00, r] += 1
            V_sub[v_b10, r] += 1
            W_sub[w_c11, r] -= 1
            
            # M7
            r = len(cols)
            cols.append(r)
            U_sub[1, r] += 1
            U_sub[4, r] -= 1
            V_sub[v_b10, r] += 1
            V_sub[v_b11, r] += 1
            W_sub[w_c00, r] += 1
            
            return cols

        # B1: v_base=0, w_base=0
        add_strassen_offset(0, 0)
        
        # B2: v_base=2, w_base=4
        add_strassen_offset(2, 4)
        
        # B3: 2x1 block. 4 mults.
        # k=4.
        # c04 = a00*b04 + a01*b14
        # c14 = a10*b04 + a11*b14
        # v_b04 = 4
        # v_b14 = 9
        # w_c04 = 8
        # w_c14 = 9
        
        r = 14
        U_sub[0, r] = 1
        U_sub[1, r] = 0
        V_sub[4, r] = 1
        V_sub[9, r] = 0
        W_sub[8, r] = 1
        W_sub[9, r] = 0
        
        r = 15
        U_sub[0, r] = 0
        U_sub[1, r] = 1
        V_sub[4, r] = 0
        V_sub[9, r] = 1
        W_sub[8, r] = 1
        W_sub[9, r] = 0
        
        r = 16
        U_sub[3, r] = 1
        U_sub[4, r] = 0
        V_sub[4, r] = 1
        V_sub[9, r] = 0
        W_sub[8, r] = 0
        W_sub[9, r] = 1
        
        r = 17
        U_sub[3, r] = 0
        U_sub[4, r] = 1
        V_sub[4, r] = 0
        V_sub[9, r] = 1
        W_sub[8, r] = 0
        W_sub[9, r] = 1
        
        # Part 2: 2x1x5 part (Rank 10)
        # A_right (col 2 of A): a02 (idx 2), a12 (idx 5)
        # B_bottom (row 2 of B): b20..b24 (idx 10,11,12,13,14)
        # C_right: c00..c04 (idx 0,2,4,6,8), c10..c14 (idx 1,3,5,7,9)
        
        # 10 mults: a02*b2k and a12*b2k for k=0..4.
        
        for k in range(5):
            r = 18 + k
            # a02 * b2k -> c0k
            U_sub[2, r] = 1
            V_sub[10+k, r] = 1
            W_sub[2*k, r] = 1
            
            # a12 * b2k -> c1k
            r2 = 18 + 5 + k
            U_sub[5, r2] = 1
            V_sub[10+k, r2] = 1
            W_sub[2*k+1, r2] = 1
            
        # Total rank 28.
        return U_sub, V_sub, W_sub

    U_full = np.zeros((6, 56))
    V_full = np.zeros((15, 56))
    W_full = np.zeros((20, 56))
    
    # Block 1: k=0..4
    U1, V1, W1 = build_2x3x5_decomp(0)
    U_full[:, 0:28] = U1
    V_full[:, 0:28] = V1
    # W1 indices are local k=0..4.
    # Full W index = k_full * 2 + i.
    # Local W index = k_local * 2 + i.
    # Since offset is 0, they match.
    W_full[0:10, 0:28] = W1
    
    # Block 2: k=5..9
    U2, V2, W2 = build_2x3x5_decomp(5)
    U_full[:, 28:56] = U2
    V_full[:, 28:56] = V2
    # W2 local k=0..4 corresponds to full k=5..9.
    # Full W index = (5 + k_local) * 2 + i = 10 + k_local * 2 + i.
    # So we add 10 to the row indices of W2.
    W_full[10:20, 28:56] = W2
    
    # Verify
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = 60
        U = np.zeros((6, rank))
        V = np.zeros((15, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(10):
                    U[i*3+j, col] = 1
                    V[j*10+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
