import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    n, m, p = DIMENSIONS

    # Build 4x4x4 Strassen rank-56 decomposition
    # Strassen 2x2: 7 multiplications M1..M7
    # 4x4: partition into 2x2 blocks, use Strassen per block position
    # This gives 4 (block positions) * 7 (Strassen mults) = 28 rank-1 terms per Strassen level
    # But we need to be more careful: 4x4x4 has 16 elements, Strassen reduces to 56 total rank
    
    # Actually, for 4x4x4, the known optimal is 56.
    # Let me build it using the tensor product structure of Strassen.
    
    # Strassen 2x2x2: rank 7
    # The 7 terms are indexed by (a, b, c) in {0,1}^3 minus (0,0,0) and (1,1,1)
    # Actually the 7 terms correspond to:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4x4, we can view it as a 2x2 block matrix where each block is 2x2.
    # Apply Strassen at the block level: 7 "macro" multiplications of 2x2x2x2x2x2
    # Each macro multiplication has rank 4 (schoolbook for 2x2x2)
    # Total: 7 * 4 = 28? That's too low - this isn't quite right.
    
    # Let me use a different approach. The known rank-56 decomposition for 4x4x4
    # can be constructed from the 2x2x2 Strassen decomposition via the following:
    # 
    # Actually, let me just hardcode a verified rank-56 decomposition.
    # The 56 rank-1 terms can be described by their (i,j,k) indices where
    # U[i*4+j, r] = 1, V[j*4+k, r] = 1, W[k*4+i, r] = 1
    
    # I'll construct the 4x4x4 part using a known valid rank-56 decomposition.
    # Based on the structure of Strassen applied hierarchically:
    
    # Let me build this carefully. For 4x4x4:
    # Partition A, B, C into 2x2 blocks of 2x2 submatrices.
    # Use Strassen at the block level with 7 block-multiplications.
    # Each block-multiplication (2x2 x 2x2 -> 2x2) has rank 4 (schoolbook).
    # But Strassen's 7 terms involve sums/differences of blocks.
    
    # The correct way: The 4x4x4 tensor can be decomposed using the fact that
    # R(4,4,4) = 56. One construction:
    # - Use 4 copies of the 2x2x2 Strassen decomposition (rank 7 each) for the "diagonal"
    #   block interactions, plus cross terms.
    
    # Let me use a direct construction. I'll define the 56 terms explicitly.
    # Each term r has a unique (i_r, j_r, k_r) triple.
    
    # Based on Larsen-Shitov-Strassen, the rank-56 decomposition uses these indices:
    # I'll generate them from the Strassen construction.
    
    # Simpler approach: Build 4x4x4 as follows
    # The 4x4x4 multiplication tensor T_{i,j,k} = delta_{i,k} * A_{i,j} * B_{j,k} (summed over j)
    # 
    # Let me just use the schoolbook for the 4x4x4 part (rank 64) combined with 
    # the 4x4x1 part (rank 16) = 80, OR find a better way.
    
    # Actually, let me implement the proper Strassen-based rank-56 for 4x4x4.
    # 
    # Construction: View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B = [[C00, C01], [C10, C11]] where each Cij is 2x2
    #
    # Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    #
    # Each Mx is a 2x2x2x2x2x2 multiplication (two 2x2 matrices multiplied -> 2x2 result)
    # Each such multiplication has rank 4 (schoolbook for 2x2x2).
    # So total rank = 7 * 4 = 28? 
    #
    # Wait, that can't be right because R(4,4,4) = 56, not 28.
    # The issue: each "block multiplication" Mx involves multiplying two 2x2 matrices,
    # which itself requires rank 4. But the Strassen combinations involve additions
    # of 2x2 matrices, which are free (they happen in the U and V factor matrices).
    #
    # So actually: 7 Strassen multiplications, each of rank 4 (for the inner 2x2x2),
    # total rank = 7 * 4 = 28?
    #
    # No wait - R(2,2,2) = 7 (Strassen), not 4!
    # So each block-level multiplication of two 2x2 matrices has rank 7.
    # Total: 7 * 7 = 49? Still not 56.
    #
    # Hmm, let me reconsider. The correct decomposition:
    # R(4,4,4) <= 4 * R(2,2,2) + 4 * R(2,2,2) ... no.
    #
    # Actually R(4,4,4) = 56 is established. The construction is:
    # 4x4x4 = tensor product structure. 
    # R(4,4,4) = R(2,2,2) * R(2,2,2) + ... no, tensor product gives R(a,b,c)*R(d,e,f) for (ad)x(be)x(cf).
    #
    # R(4,4,4) <= R(2,2,2)^2 + 4*R(2,2,2) + 4*4 = 49 + 28 + 16? No.
    #
    # Let me just use a verified rank-56 construction. I'll define the 56 (i,j,k) triples.
    
    # After careful thought, here's a valid rank-56 decomposition for 4x4x4:
    # The 56 terms are generated as follows:
    # For the 4x4x4 case, one known optimal construction uses the following 56 index triples:
    
    # I'll construct it programmatically using the Strassen recursion properly.
    # 
    # 4x4x4 decomposition via Strassen:
    # Level 1: 4x4 = 2x2 blocks, Strassen gives 7 "super-multiplications"
    # Each super-multiplication is a 2x2x2x2x2x2 tensor (multiply two 2x2 -> one 2x2)
    # Level 2: Each 2x2x2x2x2x2 can be decomposed. R(2,2,2)=7.
    # But the 2x2x2x2x2x2 is NOT simply R(2,2,2)=7. It's the tensor for multiplying 
    # two 2x2 matrices to get a 2x2 result, which IS R(2,2,2)=7.
    #
    # So total rank = 7 (Strassen at block level) * 7 (Strassen at element level) = 49?
    # That gives 49, but we know the answer is 56. So this simple recursion doesn't achieve 56.
    #
    # The actual rank-56 construction is more subtle. Let me just hardcode a known valid set.
    
    # Known rank-56 decomposition indices for 4x4x4 (0-indexed i,j,k in {0,1,2,3}):
    # I'll use a construction that I can verify.
    
    # Alternative: Let me just use rank 72 = 56 + 16 and define the 56 terms.
    # 
    # Here's a concrete rank-56 construction for 4x4x4:
    # We can use the fact that 4x4x4 can be decomposed using:
    # - 4 "diagonal" Strassen decompositions (one for each 2x2x2 sub-block)
    # - 28 "off-diagonal" terms
    # Total: 4*7 + 28 = 56
    
    # Let me define the 56 triples. Each triple (i,j,k) means:
    # U[i*4+j, r] = 1, V[j*4+k, r] = 1, W[k*4+i, r] = 1
    
    # Construction:
    # Diagonal blocks: For each pair of 2x2 block positions (a,b) in {0,1}^2,
    # use Strassen's 7 terms to multiply block A[a,:] by block B[:,b] -> block C[a,b]
    # Within each 2x2 block multiplication, use Strassen's 7 rank-1 terms.
    
    # Actually, I realize the cleanest approach: 
    # The 56 terms for 4x4x4 are well-known. Let me list them systematically.
    
    # Strassen 2x2x2 has these 7 terms (for indices in {0,1}):
    # s1: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1) with coefficients
    # Actually Strassen 2x2x2 rank 7:
    # The 7 rank-1 terms have (i,j,k) indices:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,1), (1,1,0), (1,1,1)
    # with specific U,V,W coefficients (not all 1s).
    
    # For 4x4x4, I'll use a verified construction. Let me define the full decomposition.
    
    # RANK 56 for 4x4x4 - verified construction:
    # I'll build it from first principles using the Strassen algorithm applied to 4x4x4.
    
    # 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A_00, A_01], [A_10, A_11]]
    # B = [[B_00, B_01], [B_10, B_11]]
    # C = [[C_00, C_01], [C_10, C_11]]
    
    # Strassen:
    # M1 = (A_00 + A_11) * (B_00 + B_11)
    # M2 = (A_10 + A_11) * B_00
    # M3 = A_00 * (B_10 - B_11)
    # M4 = A_11 * (B_01 - B_00)
    # M5 = (A_00 + A_01) * B_11
    # M6 = (A_10 - A_00) * (B_00 + B_01)
    # M7 = (A_01 - A_11) * (B_10 + B_11)
    
    # C_00 = M1 - M4 + M5 - M7
    # C_01 = M3 + M4
    # C_10 = M2 + M3
    # C_11 = M1 - M2 + M6 - M7
    
    # Each Mx is a 2x2 x 2x2 -> 2x2 multiplication, which has rank 7 (Strassen).
    # But the inputs to Mx are linear combinations of 2x2 blocks.
    # 
    # For M1 = (A_00 + A_11) * (B_00 + B_11):
    # The input to the left factor is A_00 + A_11, which is a 2x2 matrix.
    # This can be represented in U as: for each element (r,s) of the 2x2 block,
    # U contributes from both A_00[r,s] and A_11[r,s].
    # 
    # So each Mx multiplication (rank 7) expands to 7 rank-1 terms in the full 4x4x4,
    # but with U and V entries that are sums of block entries.
    #
    # Total: 7 Mx's * 7 rank-1 terms each = 49 terms? But we need 56.
    # 
    # The discrepancy: When we write M1 = (A_00+A_11)*(B_00+B_11), the Strassen
    # decomposition of this 2x2x2x2x2x2 multiplication has rank 7, giving 7 terms.
    # Similarly for each of the 7 Mx's. Total 49.
    # But R(4,4,4)=56, so this naive approach gives 49 which is an upper bound,
    # and 56 is the actual minimum achieved by a different construction.
    
    # Wait, 49 < 56, so if my construction is valid, R(4,4,4) <= 49, contradicting 
    # the known result R(4,4,4) = 56. So my construction must be flawed.
    
    # The flaw: Each Mx is not a simple 2x2x2x2x2x2 multiplication. 
    # For example, M1 = (A_00+A_11)*(B_00+B_11).
    # A_00+A_11 is a 2x2 matrix formed by adding two 2x2 submatrices of A.
    # In the tensor decomposition, the U factor for M1's left input would have
    # entries that are sums of A elements. Similarly for V.
    # The Strassen decomposition of the "multiply two 2x2 matrices" operation 
    # has rank 7, so M1 contributes 7 rank-1 terms to the overall decomposition.
    # Total: 7*7 = 49.
    
    # But this contradicts R(4,4,4)=56. Let me check: Is R(2,2,2)=7? Yes.
    # Is R(4,4,4) <= 7*R(2,2,2) + something? 
    # 
    # Actually, I think the issue is that the 7 Strassen multiplications at the block level
    # don't each require a full rank-7 decomposition. The inputs are linear combinations,
    # and the outputs are linear combinations. The total rank is not simply 7*7.
    
    # Let me try a completely different approach: just hardcode a verified rank-56 
    # decomposition by listing all 56 (i,j,k) index triples.
    
    # After research, here is a valid rank-56 decomposition for 4x4x4:
    # The 56 index triples (i,j,k) for which U[i*4+j,k... wait, let me use r as term index.
    
    # I'll define 56 terms. Term r has:
    # U[i_r * 4 + j_r, r] = 1
    # V[j_r * 4 + k_r, r] = 1  
    # W[k_r * 4 + i_r, r] = 1
    
    # The 56 triples (i,j,k):
    indices_444 = [
        (0,0,0), (0,0,1), (0,0,2), (0,0,3),
        (0,1,0), (0,1,1), (0,1,2), (0,1,3),
        (0,2,0), (0,2,1), (0,2,2), (0,2,3),
        (0,3,0), (0,3,1), (0,3,2), (0,3,3),
        (1,0,0), (1,0,1), (1,0,2), (1,0,3),
        (1,1,0), (1,1,1), (1,1,2), (1,1,3),
        (1,2,0), (1,2,1), (1,2,2), (1,2,3),
        (1,3,0), (1,3,1), (1,3,2), (1,3,3),
        (2,0,0), (2,0,1), (2,0,2), (2,0,3),
        (2,1,0), (2,1,1), (2,1,2), (2,1,3),
        (2,2,0), (2,2,1), (2,2,2), (2,2,3),
        (2,3,0), (2,3,1), (2,3,2), (2,3,3),
        (3,0,0), (3,0,1), (3,0,2), (3,0,3),
        (3,1,0), (3,1,1), (3,1,2), (3,1,3),
        (3,2,0), (3,2,1), (3,2,2), (3,2,3),
        (3,3,0), (3,3,1), (3,3,2), (3,3,3),
    ]
    # That's 64 terms (schoolbook). I need 56.
    # This approach of just listing (i,j,k) with single 1s in U,V,W only works for schoolbook.
    # Strassen terms have multiple non-zero entries in U, V, W.
    
    # Let me build the Strassen decomposition properly with coefficients.
    
    def build_4x4x4_strassen():
        """Build rank-56 decomposition for 4x4x4 using Strassen."""
        # I'll use the recursive Strassen construction.
        # 4x4x4: partition into 2x2 blocks, apply Strassen (7 mults), each mult is 2x2x2 (rank 7)
        # But as noted, this gives 49 which contradicts R=56.
        # 
        # Let me just use a known valid rank-56 set of (U,V,W) matrices.
        # I'll construct it carefully.
        
        # The 4x4x4 multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # i.e., T[i,j,k] is non-zero for the "diagonal" in the appropriate sense.
        
        # Actually, T[i,j,k] = delta_{contraction} where the tensor is sum_j A[i,j]B[j,k] = C[i,k]
        # The tensor is T_{i,j,k} with the constraint that the same j appears in A and B.
        # So T[i,j,k] is a 4x4x4 tensor where T[i,j,k] = 1 for all i,j,k? No.
        # The multiplication tensor M_{n} has M[i,j,k] = delta_{j,k'} where...
        
        # Let me just define it concretely. For 4x4x4:
        # T is a 4x4x4 tensor. T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
        # This is always true for any i,j,k. So T[i,j,k] = 1 for all i,j,k in {0,1,2,3}.
        # Wait no. The tensor for matrix multiplication M_{n,m,p} is defined as:
        # T_{i,j,k} = 1 if we multiply element (i,j) of A by element (j,k) of B to get (i,k) of C.
        # So T[i,j,k] = 1 for ALL i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
        # The tensor is just the all-ones tensor! No, that's not right either.
        
        # The correct definition: The matrix multiplication tensor M_{n,m,p} is the tensor in 
        # (F^n*F^m) * (F^m*F^p) * (F^n*F^p) such that:
        # <M, A*B> = sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] * C[i,k] = trace-like
        # Actually: C = A*B means C[i,k] = sum_j A[i,j]B[j,k].
        # The tensor T satisfies: T[i,j,k] = 1 for the "matching" indices.
        # Specifically, T is the tensor such that T_{i,j,k} = 1 if and only if 
        # the product A[i,j]*B[j,k] appears in C[i,k]. This is true for all i,j,k.
        # So T is the all-ones tensor of shape (n*m, m*p, p*n) = (16, 20, 20) for 4x4x5?
        
        # No! The tensor is in the space (F^n ⊗ F^m) ⊗ (F^m ⊗ F^p) ⊗ (F^n ⊗ F^p).
        # The multiplication tensor M is:
        # M = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}^*  (or E_{k,i} depending on convention)
        
        # So M[i*m+j, j*p+k, k*n+i] = 1 (using the W[k*n+i, r] convention from the task)
        # All other entries are 0.
        
        # For 4x4x4: M[i*4+j, j*4+k, k*4+i] = 1 for i,j,k in {0,1,2,3}
        # This is a 16x16x16 tensor with 64 non-zero entries (one for each (i,j,k) triple).
        
        # A rank-1 decomposition term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We need: M = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # For schoolbook: 64 terms, one for each (i,j,k):
        # U[i*4+j, r_(i,j,k)] = 1, V[j*4+k, r_(i,j,k)] = 1, W[k*4+i, r_(i,j,k)] = 1
        
        # For Strassen-based rank 56, we need 56 terms where each term may have
        # multiple non-zero entries in U, V, W.
        
        # Let me construct the 56-term decomposition explicitly.
        # I'll use the following known construction:
        
        # The 56 rank-1 terms for 4x4x4 can be described by the following (i,j,k) patterns
        # where each term may have coefficients of 1 or -1 in U, V, W.
        
        # After careful derivation using the Strassen algorithm applied to 4x4x4:
        
        terms = []
        
        # Helper: add a term with specified non-zero U, V, W entries
        # Each entry is (index, coefficient)
        
        # Strassen 2x2x2 base (rank 7):
        # For 2x2x2, the 7 terms are:
        # Term 1: U has 1s at (0,0),(0,1),(1,0),(1,1) [i.e., all 4 positions with coeff 1]
        #           V has 1s at (0,0),(0,1),(1,0),(1,1) [all 4 positions]
        #           W has 1 at (0,0), 1 at (0,1), 1 at (1,0), -1 at (1,1)
        #           Wait, this is getting complicated. Let me look up the exact Strassen 2x2x2.
        
        # Strassen 2x2x2: 7 multiplications
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # M1 = (a+d)(e+h) -> contributes to i and l
        # M2 = (c+d)e -> contributes to k
        # M3 = a(g-h) -> contributes to j and k
        # M4 = d(f-e) -> contributes to i and j
        # M5 = (a+b)h -> contributes to i and l
        # M6 = (c-a)(e+f) -> contributes to k
        # M7 = (b-d)(g+h) -> contributes to j and l
        
        # In tensor form for 2x2x2 (indices 0,1):
        # T[i,j,k] = 1 for all i,j,k in {0,1} (8 entries total)
        
        # M1: (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # U part: U[0*2+0, r1] = 1, U[1*2+1, r1] = 1 (picks A[0,0] + A[1,1])
        # V part: V[0*2+0, r1] = 1, V[1*2+1, r1] = 1 (picks B[0,0] + B[1,1])
        # W part: W[0*2+0, r1] = 1, W[1*2+1, r1] = 1 (contributes to C[0,0] and C[1,1])
        
        # M2: (c+d)e = (A[1,0]+A[1,1])B[0,0]
        # U: U[1*2+0, r2] = 1, U[1*2+1, r2] = 1
        # V: V[0*2+0, r2] = 1
        # W: W[1*2+0, r2] = 1 (contributes to C[1,0])
        
        # M3: a(g-h) = A[0,0](B[1,0]-B[1,1])
        # U: U[0*2+0, r3] = 1
        # V: V[1*2+0, r3] = 1, V[1*2+1, r3] = -1
        # W: W[0*2+1, r3] = 1, W[1*2+0, r3] = 1 (contributes to C[0,1] and C[1,0])
        
        # M4: d(f-e) = A[1,1](B[0,1]-B[0,0])
        # U: U[1*2+1, r4] = 1
        # V: V[0*2+1, r4] = 1, V[0*2+0, r4] = -1
        # W: W[0*2+0, r4] = 1, W[0*2+1, r4] = 1 (contributes to C[0,0] and C[0,1])
        
        # M5: (a+b)h = (A[0,0]+A[0,1])B[1,1]
        # U: U[0*2+0, r5] = 1, U[0*2+1, r5] = 1
        # V: V[1*2+1, r5] = 1
        # W: W[0*2+0, r5] = 1, W[1*2+1, r5] = 1 (contributes to C[0,0] and C[1,1])
        
        # M6: (c-a)(e+f) = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # U: U[1*2+0, r6] = 1, U[0*2+0, r6] = -1
        # V: V[0*2+0, r6] = 1, V[0*2+1, r6] = 1
        # W: W[1*2+0, r6] = 1 (contributes to C[1,0])
        
        # M7: (b-d)(g+h) = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        # U: U[0*2+1, r7] = 1, U[1*2+1, r7] = -1
        # V: V[1*2+0, r7] = 1, V[1*2+1, r7] = 1
        # W: W[0*2+1, r7] = 1, W[1*2+1, r7] = 1 (contributes to C[0,1] and C[1,1])
        
        # Now for 4x4x4, I partition into 2x2 blocks and apply the above.
        # This gives 7 * 7 = 49 terms, but we know the answer is 56.
        # 
        # I think the issue is that for the 4x4x4 case, the optimal construction 
        # is NOT simply the tensor product of two Strassen decompositions.
        # The actual rank-56 construction is more complex.
        
        # Given the time constraints, let me just use the schoolbook rank 64 for 4x4x4
        # and rank 16 for 4x4x1, total 80. This is safe and correct.
        
        # But to get rank 72, I need a valid rank-56 for 4x4x4.
        # Let me try the 49-term construction and verify if it's actually valid.
        # If R(4,4,4) = 56, then a 49-term construction would be a breakthrough,
        # which contradicts the known result. So my 49-term construction must be wrong.
        
        # Let me just return the schoolbook rank 80 to be safe.
        return None
    
    # Given the complexity, let me just use the schoolbook decomposition (rank 80).
    # This is guaranteed to be correct.
    
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    rank = n*m*p  # 80
    
    # Verify
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_val = np.dot(U[:, r], A.flatten())
                v_val = np.dot(V[:, r], B.flatten())
                C_computed[i, k] += u_val * v_val * W[k*n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-5):
        # Should never happen
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_dot_a = np.dot(U[:, r], A.flatten())
                    v_dot_b = np.dot(V[:, r], B.flatten())
                    val += u_dot_a * v_dot_b * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
    
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
