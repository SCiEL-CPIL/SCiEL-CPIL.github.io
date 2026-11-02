import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# --- Helper to build a term ---
def make_term(n, m, p, U_entries, V_entries, W_entries):
    """U_entries: list of (idx, val), etc."""
    return U_entries, V_entries, W_entries

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    terms = []  # list of (U_entries, V_entries, W_entries)
    
    def add_term(U_e, V_e, W_e):
        terms.append((U_e, V_e, W_e))
    
    # ============================================================
    # Block decomposition strategy:
    # Split A(3x4) into horizontal blocks: [A1(3x2) | A2(3x2)]
    # Split B(4x7) into vertical blocks: [B1(2x7); B2(2x7)]
    # C = A1*B1 + A2*B2
    #
    # For A1 (3x2) * B1 (2x7): 
    #   We can use a rank-10 decomposition (vs schoolbook 14) 
    #   Actually let's just use schoolbook for simplicity and correctness first,
    #   then optimize.
    #
    # Let's do pure schoolbook but try to find a better rank.
    # Actually, let me think about this more carefully.
    #
    # The matrix multiplication tensor T(3,4,7) has dimensions:
    #   U dim: n*m = 12
    #   V dim: m*p = 28  
    #   W dim: p*n = 21
    #   Schoolbook rank: 12*7 = 84 (sum over all i,j,k)
    #
    # Let me try a block approach:
    #   Split the 7 columns of the output (k-index) into groups.
    #   Split the 3 rows of A (i-index) into groups.
    #   Split the 4 columns of A / 4 rows of B (j-index) into groups.
    #
    # Option 1: Split j into 2+2, keep i=3, k=7
    #   Each block: (3,2,7) tensor. Rank per block: 3*2*7=42 schoolbook.
    #   Total: 42+42=84. No gain.
    #
    # Option 2: Use Strassen-like for (2,2,2) subblocks
    #   But we have 3x4x7 which doesn't tile nicely with 2x2x2.
    #
    # Let me try: split n=3 into 2+1, m=4 into 2+2, p=7 into 2+2+2+1
    # That gives blocks of sizes (2,2,2), (2,2,2), (2,2,2), (2,2,1),
    #                                (1,2,2), (1,2,2), (1,2,2), (1,2,1)
    #
    # For (2,2,2): Strassen rank 7 (vs 8 schoolbook) -> save 1
    # For (2,2,1): rank 4 (vs 4 schoolbook) -> no save... actually 2*2*1=4
    # For (1,2,2): rank 4 (vs 4 schoolbook) -> no save
    # For (1,2,1): rank 2 (vs 2 schoolbook) -> no save
    #
    # Total: 7+7+7+4+4+4+4+2 = 39 vs schoolbook 84. Big improvement!
    #
    # Wait, let me verify: n=3=2+1, m=4=2+2, p=7=2+2+2+1
    # Blocks:
    #   (i=2, j=2, k=2) x 4 (since p has 3 groups of 2 and 1 group of 1)
    #   Actually p=7 = 2+2+2+1, so 4 k-groups.
    #   n=3 = 2+1, so 2 i-groups.
    #   m=4 = 2+2, so 2 j-groups.
    #   Total blocks: 2*2*4 = 16
    #
    # Block types:
    #   (2,2,2): 2 i-groups(2) x 2 j-groups(2) x 3 k-groups(2) = 12 blocks
    #   (2,2,1): 2 x 2 x 1 k-group(1) = 4 blocks
    #   (1,2,2): 1 i-group(1) x 2 j-groups(2) x 3 k-groups(2) = 6 blocks
    #   (1,2,1): 1 x 2 x 1 = 2 blocks
    #   Total: 12+4+6+2 = 24. Wait that's wrong.
    #
    # Let me redo: 
    #   n=3: groups of size [2,1] -> 2 groups
    #   m=4: groups of size [2,2] -> 2 groups
    #   p=7: groups of size [2,2,2,1] -> 4 groups
    #   Blocks: 2*2*4 = 16
    #   
    #   (2,2,2): i-group-0(2) x j-group(2) x k-group-0,1,2(2) = 1*2*3 = 6 blocks
    #   (2,2,1): i-group-0(2) x j-group(2) x k-group-3(1) = 1*2*1 = 2 blocks
    #   (1,2,2): i-group-1(1) x j-group(2) x k-group-0,1,2(2) = 1*2*3 = 6 blocks
    #   (1,2,1): i-group-1(1) x j-group(2) x k-group-3(1) = 1*2*1 = 2 blocks
    #   Total: 6+2+6+2 = 16. Good.
    #
    # Rank:
    #   (2,2,2): 6 blocks x 7 = 42
    #   (2,2,1): 2 blocks x 4 = 8  (schoolbook for 2x2x1 is 4, can we do better? 2*2*1=4)
    #   (1,2,2): 6 blocks x 4 = 24 (schoolbook for 1x2x2 is 4)
    #   (1,2,1): 2 blocks x 2 = 4  (schoolbook for 1x2x1 is 2)
    #   Total rank: 42+8+24+4 = 78
    #
    # Hmm, only saved 6 terms. Let me think if (1,2,2) can be done in rank 3.
    # 1x2x2 tensor: C[k,i] = sum_j A[i,j]*B[j,k], where i has 1 element, j has 2, k has 2.
    # This is essentially a 2x2 matrix multiplication where one input row is fixed.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # This is just multiplying a 1x2 vector by a 2x2 matrix. Rank 4 schoolbook.
    # Can we do rank 3? The tensor is equivalent to the identity map from R^2 to R^2,
    # tensored with the row vector [A[0,0], A[0,1]]. So the "multiplication" part is 2x2 identity,
    # which has rank... actually this is a different kind of problem.
    #
    # Actually for 1x2*2x2, the output is 1x2, and the "multiplication" involves 2*2=4 products.
    # I don't think you can reduce below 4.
    #
    # Let me try a different partition: n=3, m=4=2+2, p=7
    # Maybe keep m as is and do something else.
    #
    # Alternatively: n=3=1+1+1, m=4=2+2, p=7
    # Blocks: (1,2,k) for each k-group
    #   (1,2,2): 3 i-groups x 2 j-groups x 3 k-groups(2) = 18 blocks, rank 4 each = 72
    #   (1,2,1): 3 x 2 x 1 = 6 blocks, rank 2 each = 12
    #   Total: 84. No improvement.
    #
    # Let me stick with the first approach: rank 78.
    # Actually wait, can I do (2,2,1) in rank 3?
    # 2x2*2x1: C[k,j] = sum... no wait.
    # A is 2x2, B is 2x1, C is 1x2 (since p=1 means C has 1 column, n=2 means 2 rows)
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is a 2x1 output from 2x2 * 2x1. Schoolbook rank 4.
    # Can we do rank 3? 
    # M[i,j] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # N[i,j] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Hmm, this looks like it might be reducible. Let's see:
    # Let's define:
    # P1 = (A[0,0]+A[1,0])*(B[0,0]+B[1,0])
    # P2 = (A[0,0]+A[1,0])*(B[0,0]-B[1,0])
    # P3 = A[0,0]*B[0,0]
    # P4 = A[1,1]*B[1,0]  -- wait, B only has 1 column here.
    #
    # B is 2x1, so B[0,0] and B[1,0].
    # A[0,0], A[0,1], A[1,0], A[1,1].
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #
    # This is the product of a 2x2 matrix with a 2x1 vector. 
    # I believe the rank is 4 (it's essentially the same as 2x2x1 multiplication).
    # Actually, no. Let me think again. This is not a square matrix multiplication.
    # The tensor has dimensions U:2*2=4, V:2*1=2, W:1*2=2.
    # I think rank 4 is minimal here.
    #
    # So total rank = 42 + 8 + 24 + 4 = 78.
    
    # Let me define the partitions:
    n_groups = [2, 1]  # sizes for n=3
    m_groups = [2, 2]  # sizes for m=4
    p_groups = [2, 2, 2, 1]  # sizes for p=7
    
    n_offsets = [0]
    for g in n_groups[:-1]:
        n_offsets.append(n_offsets[-1] + g)
    
    m_offsets = [0]
    for g in m_groups[:-1]:
        m_offsets.append(m_offsets[-1] + g)
    
    p_offsets = [0]
    for g in p_groups[:-1]:
        p_offsets.append(p_offsets[-1] + g)
    
    for gi, (n_start, n_sz) in enumerate(zip(n_offsets, n_groups)):
        for gj, (m_start, m_sz) in enumerate(zip(m_offsets, m_groups)):
            for gk, (p_start, p_sz) in enumerate(zip(p_offsets, p_groups)):
                # Extract block
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    # Strassen 7-term
                    i0, i1 = n_start, n_start+1
                    j0, j1 = m_start, m_start+1
                    k0, k1 = p_start, p_start+1
                    # A indices
                    a00 = i0*m+j0
                    a01 = i0*m+j1
                    a10 = i1*m+j0
                    a11 = i1*m+j1
                    # B indices
                    b00 = j0*p+k0
                    b01 = j0*p+k1
                    b10 = j1*p+k0
                    b11 = j1*p+k1
                    # C indices
                    c00 = k0*n+i0
                    c01 = k0*n+i1
                    c10 = k1*n+i0
                    c11 = k1*n+i1
                    
                    # Strassen
                    # S1 = B00 + B11
                    # S2 = A10 + A11
                    # S3 = A00 - A10
                    # S4 = A11 - A01
                    # S5 = A00 + A01
                    # S6 = B10 - B00
                    # S7 = B01 - B11
                    # M1 = S2*S6
                    # M2 = S3*S1
                    # M3 = S4*S7
                    # M4 = S5*S00
                    # M5 = A00*B01
                    # M6 = S6*B01  -- wait, standard Strassen:
                    # P1 = (A00+A11)(B00+B11)
                    # P2 = (A10+A11)B00
                    # P3 = A00(B11-B01)
                    # P4 = A11(B00-B10)
                    # P5 = (A00+A01)B11
                    # P6 = (A10-A00)(B00+B10)
                    # P7 = (A01-A11)(B01+B11)
                    # C00 = P1+P4-P5+P7
                    # C01 = P3+P5
                    # C10 = P2+P4
                    # C11 = P1-P2+P3+P6
                    
                    # Each P is a rank-1 term in the tensor.
                    # But wait, Strassen combines terms, so the final C entries are sums.
                    # In the tensor decomposition, we need rank-1 terms that sum to give C.
                    
                    # Let me write out each Strassen product as a separate rank-1 term,
                    # then combine.
                    
                    # P1 = (A00+A11)*(B00+B11)
                    add_term(
                        [(a00, 1), (a11, 1)],
                        [(b00, 1), (b11, 1)],
                        [(c00, 1), (c11, -1)]  # wait, C11 has P1 with +1
                    )
                    # Hmm, this is getting complicated. Let me use a different approach.
                    #
                    # Actually, for the tensor decomposition, I should directly construct
                    # the 7 Strassen terms as rank-1 tensors.
                    
                    # Let me clear this block and redo.
                    terms.pop()  # Remove the incorrect term above
                    
                    # Strassen 7-term for 2x2x2:
                    # Define the 7 rank-1 terms:
                    # t1: U={A00:+1,A11:+1}, V={B00:+1,B11:+1}, W={C00:+1,C11:+1}  -- No, this is wrong.
                    #
                    # The Strassen algorithm computes:
                    # P1 = (A00+A11)(B00+B11)
                    # This is NOT a rank-1 tensor in the original space. It's a bilinear form
                    # that's a product of sums. So it decomposes into 4 rank-1 terms if written naively,
                    # but the point of Strassen is that these 7 bilinear forms suffice.
                    #
                    # For the tensor decomposition, each "product" in Strassen corresponds to
                    # a rank-1 term where U, V, W each have multiple entries.
                    
                    # Term 1: (A00+A11)(B00+B11) contributes to C00(+1) and C11(+1)
                    # Actually no. C00 = P1+P4-P5+P7, C11 = P1-P2+P3+P6
                    # So P1 contributes +1 to C00 and +1 to C11.
                    
                    # But P1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
                    # This is a sum of 4 rank-1 terms in the (A,B,C) space.
                    # 
                    # I think I'm overcomplicating this. Let me just use the schoolbook
                    # for now and get a correct answer, then optimize.
                    #
                    # Actually, let me reconsider the whole approach. The cleanest way
                    # is to just output the schoolbook decomposition (rank 84) and make sure
                    # it's correct. Then if I have time, try to reduce.
                    #
                    # But the task says to find rank-minimal. Score 1.0 corresponds to rank 84.
                    # To improve, I need rank < 84.
                    
                    # Let me try a cleaner Strassen implementation.
                    # I'll define the 7 Strassen multilinear forms explicitly.
                    
                    # Clear the erroneous term we added
                    # (We already popped one, but we haven't added the correct ones yet)
                    
                    # The 7 Strassen terms for multiplying A (2x2) by B (2x2) to get C (2x2):
                    # Indices in local 2x2 block: A[li,lj], B[lj,lk], C[lk,li]
                    # li: 0,1; lj: 0,1; lk: 0,1
                    
                    # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                    #   U entries: (A00, 1), (A11, 1)
                    #   V entries: (B00, 1), (B11, 1)  
                    #   W entries: (C00, 1), (C11, 1)  -- C00 gets P1, C11 gets P1
                    
                    # Wait, I need to double-check the Strassen formulas.
                    # Standard Strassen:
                    # p1 = (a00+a11)(b00+b11)
                    # p2 = (a10+a11)b00
                    # p3 = a00(b11-b01)
                    # p4 = a11(b00-b10)
                    # p5 = (a00+a01)b11
                    # p6 = (a10-a00)(b00+b10)
                    # p7 = (a01-a11)(b01+b11)
                    # c00 = p1 + p4 - p5 + p7
                    # c01 = p3 + p5
                    # c10 = p2 + p4
                    # c11 = p1 - p2 + p3 + p6
                    
                    # So:
                    # p1: U=[a00,a11], V=[b00,b11], W=[c00,c11] with coeffs [1,1],[1,1],[1,1]
                    # p2: U=[a10,a11], V=[b00], W=[c10,c11] with coeffs [1,1],[1],[1,-1]
                    # p3: U=[a00], V=[b11,b01], W=[c01,c11] with coeffs [1],[1,-1],[1,1]
                    # p4: U=[a11], V=[b00,b10], W=[c00,c10] with coeffs [1],[1,-1],[1,1]
                    # p5: U=[a00,a01], V=[b11], W=[c00,c01] with coeffs [1,1],[1],[-1,1]
                    # p6: U=[a10,a00], V=[b00,b10], W=[c11] with coeffs [1,-1],[1,1],[1]
                    # p7: U=[a01,a11], V=[b01,b11], W=[c00] with coeffs [1,-1],[1,1],[1]
                    
                    # Let me verify p1: (a00+a11)(b00+b11) = a00*b00+a00*b11+a11*b00+a11*b11
                    # This contributes to c00 and c11. In c00, the coefficient is +1. In c11, +1.
                    # The rank-1 term is: sum_{u,v} U_u V_v = (a00+a11)(b00+b11)
                    # And W has c00:+1, c11:+1. So the full contribution to C is:
                    # C[k,i] += W_ki * (a00+a11)(b00+b11)
                    # For c00: +1*(a00+a11)(b00+b11) ✓
                    # For c11: +1*(a00+a11)(b00+b11) ✓
                    
                    # This is valid! Each Strassen product is a single rank-1 term
                    # where U and V can have multiple non-zero entries.
                    
                    # So I add 7 terms for each 2x2x2 block.
                    
                    # p1
                    add_term(
                        [(a00, 1), (a11, 1)],
                        [(b00, 1), (b11, 1)],
                        [(c00, 1), (c11, 1)]
                    )
                    # p2 = (a10+a11)*b00
                    add_term(
                        [(a10, 1), (a11, 1)],
                        [(b00, 1)],
                        [(c10, 1), (c11, -1)]
                    )
                    # p3 = a00*(b11-b01)
                    add_term(
                        [(a00, 1)],
                        [(b11, 1), (b01, -1)],
                        [(c01, 1), (c11, 1)]
                    )
                    # p4 = a11*(b00-b10)
                    add_term(
                        [(a11, 1)],
                        [(b00, 1), (b10, -1)],
                        [(c00, 1), (c10, 1)]
                    )
                    # p5 = (a00+a01)*b11
                    add_term(
                        [(a00, 1), (a01, 1)],
                        [(b11, 1)],
                        [(c00, -1), (c01, 1)]
                    )
                    # p6 = (a10-a00)*(b00+b10)
                    add_term(
                        [(a10, 1), (a00, -1)],
                        [(b00, 1), (b10, 1)],
                        [(c11, 1)]
                    )
                    # p7 = (a01-a11)*(b01+b11)
                    add_term(
                        [(a01, 1), (a11, -1)],
                        [(b01, 1), (b11, 1)],
                        [(c00, 1)]
                    )
                    
                elif n_sz == 2 and m_sz == 2 and p_sz == 1:
                    # 2x2 * 2x1 -> 1x2 (C has p*n = 1*2 = 2 elements)
                    # Schoolbook: 4 terms
                    i0, i1 = n_start, n_start+1
                    j0, j1 = m_start, m_start+1
                    k0 = p_start
                    # A indices
                    a00 = i0*m+j0
                    a01 = i0*m+j1
                    a10 = i1*m+j0
                    a11 = i1*m+j1
                    # B indices
                    b00 = j0*p+k0
                    b10 = j1*p+k0
                    # C indices (p*n = 1*2, so C[k*n+i] = C[k0*n+i])
                    c00 = k0*n+i0
                    c10 = k0*n+i1
                    
                    # C[i,0] = sum_j A[i,j]*B[j,0]
                    # c00 = a00*b00 + a01*b10
                    # c10 = a10*b00 + a11*b10
                    
                    # Can we use Strassen-like? This is 2x2*2x1.
                    # Let's just use schoolbook: 4 terms.
                    add_term([(a00, 1)], [(b00, 1)], [(c00, 1)])
                    add_term([(a01, 1)], [(b10, 1)], [(c00, 1)])
                    add_term([(a10, 1)], [(b00, 1)], [(c10, 1)])
                    add_term([(a11, 1)], [(b10, 1)], [(c10, 1)])
                    
                elif n_sz == 1 and m_sz == 2 and p_sz == 2:
                    # 1x2 * 2x2 -> 2x1 (C has p*n = 2*1 = 2 elements)
                    i0 = n_start
                    j0, j1 = m_start, m_start+1
                    k0, k1 = p_start, p_start+1
                    a00 = i0*m+j0
                    a01 = i0*m+j1
                    b00 = j0*p+k0
                    b01 = j0*p+k1
                    b10 = j1*p+k0
                    b11 = j1*p+k1
                    c00 = k0*n+i0
                    c10 = k1*n+i0
                    
                    # C[k,0] = sum_j A[0,j]*B[j,k]
                    # c00 = a00*b00 + a01*b10
                    # c10 = a00*b01 + a01*b11
                    
                    # Schoolbook: 4 terms
                    add_term([(a00, 1)], [(b00, 1)], [(c00, 1)])
                    add_term([(a01, 1)], [(b10, 1)], [(c00, 1)])
                    add_term([(a00, 1)], [(b01, 1)], [(c10, 1)])
                    add_term([(a01, 1)], [(b11, 1)], [(c10, 1)])
                    
                elif n_sz == 1 and m_sz == 2 and p_sz == 1:
                    # 1x2 * 2x1 -> 1x1
                    i0 = n_start
                    j0, j1 = m_start, m_start+1
                    k0 = p_start
                    a00 = i0*m+j0
                    a01 = i0*m+j1
                    b00 = j0*p+k0
                    b10 = j1*p+k0
                    c00 = k0*n+i0
                    
                    # c00 = a00*b00 + a01*b10
                    add_term([(a00, 1)], [(b00, 1)], [(c00, 1)])
                    add_term([(a01, 1)], [(b10, 1)], [(c00, 1)])
    
    rank = len(terms)
    
    # Build matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t, (U_e, V_e, W_e) in enumerate(terms):
        for (idx, val) in U_e:
            U[idx, t] += val
        for (idx, val) in V_e:
            V[idx, t] += val
        for (idx, val) in W_e:
            W[idx, t] += val
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Emergency fallback: return schoolbook
        pass  # Shouldn't happen since this is O(1) work
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
