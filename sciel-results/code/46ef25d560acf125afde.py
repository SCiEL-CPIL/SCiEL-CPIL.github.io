"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_factor(u_vals, v_vals, w_vals, c):
        for i, val in u_vals.items():
            if val != 0:
                U[i][c] += val
        for i, val in v_vals.items():
            if val != 0:
                V[i][c] += val
        for i, val in w_vals.items():
            if val != 0:
                W[i][c] += val

    # Factors 0-6: Strassen block for i=0, j=0, k=0
    # Rows: U: i*m+j, V: j*p+k, W: k*n+i
    # i=0, j=0 -> U row 0, V row 0, W row 0
    add_factor({0: -1}, {0: 1}, {0: 1}, 0)       # M1: (A11-A00)(B00+B11) -> A00 part -1, B00 part 1, C00 part 1
    add_factor({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}, 1) # M2: (A00+A11)B10 -> Wait, indices differ
    
    # Let's restart the Strassen mapping carefully for 2x2x2 sub-tensor embedded at specific positions.
    # The task is 2x3x8.
    # We can decompose B (3x8) into four 2x4 blocks? No, Strassen works on 2x2.
    # Let's use a simpler approach: 
    # Rank of 2x3x8 tensor.
    # Schoolbook: 48.
    # Can we do better?
    # The tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # We can view this as a linear map from U⊗V to W.
    # U has dim 6 (2x3), V has dim 24 (3x8), W has dim 16 (8x2).
    # Actually, the multiplication tensor is defined by T = sum_{j} e_i^j ⊗ e_j^k ⊗ ...
    # Let's try to find a decomposition.
    
    # If we use the identity: 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We can write this as a bilinear form.
    # The rank of the matrix multiplication tensor for n x m x p is at most n*m*p.
    # Is it less?
    # For 2x2x2, rank is 7 (Strassen) vs 8.
    # For 2x3x8, we can partition the j dimension (size 3) into 1+2.
    # Partition k dimension (size 8) into 4+4 or similar?
    # Let's partition j into j0 and j1..j2.
    # A = [A0, A12] where A0 is 2x1, A12 is 2x2.
    # B = [B0; B12] where B0 is 1x8, B12 is 2x8.
    # C = A0 B0 + A12 B12.
    # Term 1: A0 (2x1) * B0 (1x8) -> rank 2*1*8 = 16? No, rank of 2x1x8 multiplication is 2*1*8=16? 
    # Actually, rank of m1 x n1 x p1 multiplication is m1*n1*p1 unless Strassen applies.
    # For 2x1x8, the tensor is sum_i sum_k e_i ⊗ e_1 ⊗ e_k. This is a sum of 16 simple tensors. Can it be reduced?
    # The matrix is rank 1 in the middle. The tensor is essentially a sum of 16 independent terms because the middle index is fixed.
    # Wait, if middle index is fixed to 1, T_{i,1,k} = 1.
    # T = sum_{i,k} e_i ⊗ e_1 ⊗ e_k.
    # This is a sum of 2*8 = 16 simple tensors.
    # Can we write this with fewer?
    # If we write sum_c u_c ⊗ v_c ⊗ w_c.
    # Since v_c must be e_1 (or proportional), all v_c are collinear.
    # This implies the rank is exactly 16.
    # So the first part A0 B0 has rank 16.
    
    # Term 2: A12 (2x2) * B12 (2x8).
    # Partition B12 into two 2x4 blocks: B12a, B12b.
    # C12 = A12 B12a + A12 B12b.
    # This doesn't help directly unless we use Strassen on the 2x2 part.
    # A12 B12 is 2x2 * 2x8.
    # We can view B12 as [B12a, B12b] (2x4 each).
    # Result is 2x8.
    # We can compute A12 B12 using Strassen if we split B12 into 2x2 blocks? No, B12 is 2x8.
    # We can use the fact that rank(2x2x8) <= 7*4 = 28? (Using Strassen for each 2x2 block of the result? No.)
    # Let's split the output k into 4 blocks of 2.
    # B12 = [B1, B2, B3, B4] where each is 2x2.
    # C12 = [A12 B1, A12 B2, A12 B3, A12 B4].
    # Each A12 Bc is 2x2. Rank of each is 7 (Strassen).
    # Total rank for this part: 4 * 7 = 28.
    # Total rank: 16 + 28 = 44.
    
    # Can we do better?
    # What if we split j differently?
    # j=3. Split into 1+1+1?
    # 3 terms of 2x1x8. Rank 3*16 = 48. Worse.
    # Split into 2+1?
    # 2x2x8 part + 2x1x8 part.
    # 2x2x8 part: Split B (2x8) into four 2x2 blocks.
    # Each 2x2x2 block has rank 7.
    # Total 4 * 7 = 28.
    # 2x1x8 part: Rank 16.
    # Total 44.
    
    # Is there a way to combine them?
    # The terms are disjoint in the j-index (middle dimension of U/V).
    # U for j=0 part uses rows 0,1.
    # U for j=1,2 part uses rows 2,3.
    # V for j=0 part uses rows 0..7.
    # V for j=1,2 part uses rows 8..23.
    # W is shared?
    # W index is k*n+i.
    # For j=0 term: W[k*2+i] gets contribution from A0_i B0_k.
    # For j=1,2 term: W[k*2+i] gets contribution from A12_i,l B12_l,k.
    # Since the W indices are the same, we can potentially combine factors if the U and V supports don't interfere?
    # But U and V supports are disjoint in terms of which "columns" of the original matrices they touch?
    # No, U rows correspond to (i,j).
    # For j=0, U rows 0,1 are active.
    # For j=1,2, U rows 2,3,4,5 are active.
    # They are disjoint rows in U.
    # V rows correspond to (j,k).
    # For j=0, V rows 0..7 are active.
    # For j=1,2, V rows 8..23 are active.
    # They are disjoint rows in V.
    # W rows correspond to (k,i).
    # Both parts contribute to ALL W rows 0..15.
    # So we can't just add ranks. We have to see if we can merge the W dependencies.
    # But since U and V supports are disjoint, the factors for part 1 and part 2 are orthogonal in U and V.
    # Can we combine two factors f1 = u1 ⊗ v1 ⊗ w1 and f2 = u2 ⊗ v2 ⊗ w2 into one?
    # Only if they share structure.
    # Since u1 has zeros where u2 is non-zero, and vice versa, we can't merge them into a single simple tensor u ⊗ v ⊗ w unless w is zero or something.
    # Wait, a factor is u ⊗ v ⊗ w.
    # If we have two sets of factors, we can just concatenate them.
    # The rank is the sum of the ranks of the disjoint parts?
    # Yes, if the supports in U and V are disjoint, the rank is additive.
    # Proof: Suppose T = T1 + T2.
    # T1 is supported on U1 x V1 x W.
    # T2 is supported on U2 x V2 x W.
    # U1 ∩ U2 = 0, V1 ∩ V2 = 0 (in terms of non-zero coordinates).
    # Any decomposition of T must decompose T1 and T2 separately?
    # Not necessarily, but likely.
    # So rank(2x3x8) <= rank(2x1x8) + rank(2x2x8) = 16 + 28 = 44.
    
    # Can we reduce 2x2x8 further?
    # Rank of 2x2xp?
    # It is known that rank(2x2x2) = 7.
    # rank(2x2x3) = ?
    # rank(2x2x4) = ?
    # Let's look up or derive.
    # For 2x2xn, the rank is generally 7 * ceil(n/2) if we block into 2x2s?
    # If n=8, 4 blocks of 2x2. Rank 4*7=28.
    # Is it possible to do better than 28 for 2x2x8?
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # This is a sum of 2*2*8 = 32 simple tensors.
    # Strassen gives 28.
    # Is 28 minimal?
    # Lower bounds for matrix multiplication rank:
    # rank(n x m x p) >= n + m + p - 1 ? No.
    # rank >= max(n*m, m*p, n*p) ? No.
    # For 2x2x2, rank 7.
    # For 2x2x3, rank is at most 7+7=14 (if 3=2+1, 2x2x1 rank 4). 14?
    # Actually, rank(2x2x3) is 10?
    # Let's check online knowledge or logic.
    # If we can find a better decomposition for 2x2x8, we win.
    # But 44 is already better than 48.
    # Let's implement the 44 rank decomposition.
    
    # Part 1: j=0. A0 (2x1), B0 (1x8).
    # T1_{i,0,k} = 1.
    # Factors: for each i in 0,1 and k in 0..7, factor u=(e_i, 0), v=(e_0 at k? No, v index is j*p+k. j=0, so k=0..7), w=e_{k*2+i}.
    # Wait, V index is j*p+k. For j=0, indices are 0..7.
    # U index is i*m+j. For j=0, indices are 0,1.
    # W index is k*n+i.
    # Factors c = 0..15.
    # i = c // 8, k = c % 8.
    # U[i][c] = 1.
    # V[k][c] = 1. (Since j=0, V index is 0*8+k = k).
    # W[k*2+i][c] = 1.
    
    for c in range(16):
        i = c // 8
        k = c % 8
        U[i][c] = 1
        V[k][c] = 1
        W[k*2 + i][c] = 1

    # Part 2: j=1,2. A12 (2x2), B12 (2x8).
    # U indices for j=1,2: i=0,1 and j=1,2.
    # i=0,j=1 -> U idx 0*3+1 = 1? No, U idx is i*m+j.
    # n=2, m=3.
    # i=0,j=1 -> 1.
    # i=0,j=2 -> 2.
    # i=1,j=1 -> 4.
    # i=1,j=2 -> 5.
    # Wait, U has 6 rows.
    # Rows 0,1 used by Part 1?
    # Part 1: j=0.
    # i=0,j=0 -> 0.
    # i=1,j=0 -> 3.
    # So Part 1 uses U rows 0, 3.
    # Part 2 uses U rows 1, 2, 4, 5.
    # Disjoint? Yes.
    
    # V indices for j=1,2.
    # V idx is j*p+k. p=8.
    # j=1, k=0..7 -> 8..15.
    # j=2, k=0..7 -> 16..23.
    # Part 1 used V rows 0..7.
    # Part 2 uses V rows 8..23.
    # Disjoint? Yes.
    
    # We need rank 28 for this part.
    # We split B12 into four 2x2 blocks: B^{(0)}, B^{(1)}, B^{(2)}, B^{(3)}.
    # B12[k, col] where col=0..7.
    # Block t (0..3) corresponds to cols 2t, 2t+1.
    # B12 row 0 is B12[0, 2t], B12[0, 2t+1].
    # B12 row 1 is B12[1, 2t], B12[1, 2t+1].
    # A12 is 2x2.
    # C12 = A12 B12.
    # C12[:, 2t:2t+2] = A12 B^{(t)}.
    # We use Strassen for each t.
    # Strassen for A (2x2) and B (2x2).
    # A12 indices:
    # A12[0,0] is A[0,1] (i=0, j=1). U idx 1.
    # A12[0,1] is A[0,2] (i=0, j=2). U idx 2.
    # A12[1,0] is A[1,1] (i=1, j=1). U idx 4.
    # A12[1,1] is A[1,2] (i=1, j=2). U idx 5.
    
    # B^{(t)} indices:
    # B12[0, 2t] is B[1, 2t] (j=1, k=2t). V idx 8+2t.
    # B12[0, 2t+1] is B[1, 2t+1] (j=1, k=2t+1). V idx 8+2t+1.
    # B12[1, 2t] is B[2, 2t] (j=2, k=2t). V idx 16+2t.
    # B12[1, 2t+1] is B[2, 2t+1] (j=2, k=2t+1). V idx 16+2t+1.
    
    # W indices for C12:
    # C12[0, 2t] is C[0, 2t] (i=0, k=2t). W idx 2t*2+0 = 4t.
    # C12[0, 2t+1] is C[0, 2t+1] (i=0, k=2t+1). W idx (2t+1)*2+0 = 4t+2.
    # C12[1, 2t] is C[1, 2t] (i=1, k=2t). W idx 2t*2+1 = 4t+1.
    # C12[1, 2t+1] is C[1, 2t+1] (i=1, k=2t+1). W idx (2t+1)*2+1 = 4t+3.
    
    # Strassen factors for A12 B^{(t)}:
    # M1 = (A11 - A00)(B00 + B11)
    # M2 = (A00 + A11)B10
    # M3 = A10(B01 - B11)
    # M4 = A11(B00 - B01)
    # M5 = (A00 + A10)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A12 mapping:
    # A00 = A[0,1] (U1)
    # A01 = A[0,2] (U2)
    # A10 = A[1,1] (U4)
    # A11 = A[1,2] (U5)
    
    # B^{(t)} mapping:
    # B00 = B[1, 2t] (V 8+2t)
    # B01 = B[1, 2t+1] (V 9+2t)
    # B10 = B[2, 2t] (V 16+2t)
    # B11 = B[2, 2t+1] (V 17+2t)
    
    # C^{(t)} mapping (result 2x2):
    # C00 = C[0, 2t] (W 4t)
    # C01 = C[0, 2t+1] (W 4t+2)
    # C10 = C[1, 2t] (W 4t+1)
    # C11 = C[1, 2t+1] (W 4t+3)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    base_c = 16
    for t in range(4):
        v00 = 8 + 2*t
        v01 = 9 + 2*t
        v10 = 16 + 2*t
        v11 = 17 + 2*t
        
        w00 = 4*t
        w01 = 4*t + 2
        w10 = 4*t + 1
        w11 = 4*t + 3
        
        # M1: (A11 - A00)(B00 + B11)
        # U: A11(1), A00(-1) -> U5: 1, U1: -1
        # V: B00(1), B11(1) -> V00: 1, V11: 1
        # W: C00(1), C11(1) -> W00: 1, W11: 1
        add_factor({5: 1, 1: -1}, {v00: 1, v11: 1}, {w00: 1, w11: 1}, base_c)
        
        # M2: (A00 + A11)B10
        # U: A00(1), A11(1) -> U1: 1, U5: 1
        # V: B10(1) -> V10: 1
        # W: C10(1) -> W10: 1
        add_factor({1: 1, 5: 1}, {v10: 1}, {w10: 1}, base_c + 1)
        
        # M3: A10(B01 - B11)
        # U: A10(1) -> U4: 1
        # V: B01(1), B11(-1) -> V01: 1, V11: -1
        # W: C01(1), C11(1) -> W01: 1, W11: 1
        add_factor({4: 1}, {v01: 1, v11: -1}, {w01: 1, w11: 1}, base_c + 2)
        
        # M4: A11(B00 - B01)
        # U: A11(1) -> U5: 1
        # V: B00(1), B01(-1) -> V00: 1, V01: -1
        # W: C00(1), C10(1) -> W00: 1, W10: 1
        add_factor({5: 1}, {v00: 1, v01: -1}, {w00: 1, w10: 1}, base_c + 3)
        
        # M5: (A00 + A10)B11
        # U: A00(1), A10(1) -> U1: 1, U4: 1
        # V: B11(1) -> V11: 1
        # W: C00(-1), C01(1) -> W00: -1, W01: 1
        add_factor({1: 1, 4: 1}, {v11: 1}, {w00: -1, w01: 1}, base_c + 4)
        
        # M6: (A10 - A00)(B00 + B01)
        # U: A10(1), A00(-1) -> U4: 1, U1: -1
        # V: B00(1), B01(1) -> V00: 1, V01: 1
        # W: C11(1) -> W11: 1
        add_factor({4: 1, 1: -1}, {v00: 1, v01: 1}, {w11: 1}, base_c + 5)
        
        # M7: (A01 - A11)(B10 + B11)
        # U: A01(1), A11(-1) -> U2: 1, U5: -1
        # V: B10(1), B11(1) -> V10: 1, V11: 1
        # W: C00(1), C11(1) -> W00: 1, W11: 1
        add_factor({2: 1, 5: -1}, {v10: 1, v11: 1}, {w00: 1, w11: 1}, base_c + 6)
        
        base_c += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
