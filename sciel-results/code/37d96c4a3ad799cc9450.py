"""Matrix multiplication 6x6x6 with tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Kronecker product of Strassen (2x2x2, rank 7) and a 3x3x3 decomposition.
    # We'll use the standard schoolbook for 3x3 (rank 27) as baseline,
    # but try to find a better 3x3 decomposition.
    #
    # For now, let's use Strassen ⊗ schoolbook_3x3 = rank 7*27 = 189
    # This is better than schoolbook 6x6x6 = 216.
    #
    # Actually, let's think about this more carefully.
    # The 6x6x6 tensor can be viewed as a 3x3 block matrix where each block is 2x2.
    # If we use Strassen for the block-level 3x3 multiplication, we need a 3x3x3 decomposition.
    # But Strassen is for 2x2x2.
    #
    # Alternative: View 6x6 as 3 blocks of 2x2. Then 6x6x6 = (2x2)^3 x 3^3 blocks.
    # Use Strassen (rank 7) for the 2x2x2 within each block position,
    # and schoolbook (rank 27) for the 3x3 block structure.
    # Total rank = 7 * 27 = 189.
    #
    # Let's construct this properly.
    #
    # Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the 6x6 case with 3x3 blocks of 2x2:
    # A is 3x3 blocks of 2x2, B is 3x3 blocks of 2x2, C is 3x3 blocks of 2x2.
    #
    # We can use schoolbook for the 3x3 block structure:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # where each A[i,k], B[k,j], C[i,j] is a 2x2 matrix.
    #
    # For each block multiplication A[i,k] * B[k,j], we use Strassen (rank 7).
    # Total multiplications: 3*3*3 * 7 = 189.
    #
    # Let's build the factor matrices.
    # n = m = p = 6
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)
    #
    # Indexing:
    # A[i,j] where i in 0..5, j in 0..5. Block (i//2, j//2), position (i%2, j%2).
    # U[i*m+j, c] = coefficient of A[i,j] in the c-th multiplication.
    # V[j*p+k, c] = coefficient of B[j,k] in the c-th multiplication.
    # W[k*n+i, c] = coefficient of C[k,i] in the c-th multiplication.
    #
    # Wait, the convention is W[k*n+i, r]. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r]... no.
    #
    # Standard tensor decomposition: C[i,k] = sum_{j,r} A[i,j] * V[j,k,r]
    # Or: C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k])
    #
    # Actually, the standard form is:
    # C[i,k] = sum_{j=1}^m sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[i,k,r]
    # No, that's not right either.
    #
    # The tensor T_{i,j,k} = delta_{i,k} * A_{i,j} * B_{j,k}... no.
    # Matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The multiplication tensor has T_{i,j,k} = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}.
    #
    # Rank decomposition: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # So U is indexed by (i,j) flattened, V by (j,k) flattened, W by (k,i) flattened?
    #
    # "W[k*n+i, r]" - so W is indexed by (k,i) where k is the second index of B and i is the first index of C.
    # So W_{k,i,r} corresponds to the coefficient for C_{i,k}.
    #
    # So: C_{i,k} = sum_j sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    #
    # Let me verify with schoolbook:
    # For each (i,j,k): U_{i*m+j, c}=1, V_{j*p+k, c}=1, W_{k*n+i, c}=1
    # Then C_{i',k'} = sum_j sum_r U_{i'*m+j,r} V_{j*p+k',r} W_{k'*n+i',r}
    # = sum_{j,r: U_{i'*m+j,r}=1, V_{j*p+k',r}=1, W_{k'*n+i',r}=1} 1
    # For schoolbook, each c corresponds to unique (i,j,k), so:
    # U_{i'*m+j,r}=1 iff i'=i and j=j
    # V_{j*p+k',r}=1 iff j=j and k'=k
    # W_{k'*n+i',r}=1 iff k'=k and i'=i
    # So the sum is 1 iff i'=i, j=j, k'=k, which gives C_{i,k} = sum_j A_{i,j}B_{j,k}. Correct.
    #
    # Now for our Kronecker construction:
    # We have 3x3 blocks of 2x2.
    # A_{i,j} where i = 2*I + a, j = 2*J + b, with I,J in 0..2, a,b in 0..1.
    # B_{j,k} where j = 2*J + b, k = 2*K + c, with J,K in 0..2, b,c in 0..1.
    # C_{i,k} where i = 2*I + a, k = 2*K + c.
    #
    # C_{2I+a, 2K+c} = sum_{J,b} A_{2I+a, 2J+b} B_{2J+b, 2K+c}
    # = sum_J (A_block[I,J] * B_block[J,K])_{a,c}
    #
    # For each J, we compute A_block[I,J] * B_block[J,K] using Strassen (7 multiplications).
    #
    # Strassen for 2x2:
    # Let A_block = [[A00, A01], [A10, A11]], B_block = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, for each (I, J, K) and each Strassen multiplication r=0..6:
    # We need to express M_r as sum of U*V*W terms.
    #
    # M1 = (A00+A11)(B00+B11)
    # This involves A entries: (I, J) block positions (0,0) and (1,1), i.e., A_{2I,2J} and A_{2I+1,2J+1}
    # And B entries: (J, K) block positions (0,0) and (1,1), i.e., B_{2J,2K} and B_{2J+1,2K+1}
    #
    # But M1 is a product of two sums, which is not a single rank-1 term.
    # We need to expand: (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each of these is a rank-1 term in the tensor sense.
    #
    # So actually, using Strassen doesn't directly give rank 7 for the 2x2 case in this tensor decomposition framework.
    # The Strassen algorithm uses 7 multiplications of (n/2)x(n/2) matrices, but each such multiplication
    # is itself a tensor of rank (n/2)^3.
    #
    # So the total rank for 6x6x6 using Strassen at the top level (treating 6x6 as 2x2 blocks of 3x3)
    # would be 7 * 27 = 189, where 27 is the schoolbook rank for 3x3x3.
    #
    # Alternatively, treat 6x6 as 3x3 blocks of 2x2:
    # Top level: 3x3 block multiplication, schoolbook rank 27.
    # Each block multiplication is 2x2x2, Strassen rank 7.
    # Total: 27 * 7 = 189.
    #
    # Either way, we get 189. Let's go with 3x3 blocks of 2x2.
    #
    # For each (I, J, K) in 0..2, and for the 2x2 block multiplication A_block[I,J] * B_block[J,K]:
    # We use Strassen's 7 multiplications, but each multiplication is of 2x2 matrices,
    # which requires 4 scalar multiplications (schoolbook for 2x2x2 is rank 8, but Strassen reduces to 7).
    #
    # Wait, I'm confusing myself. Let me restart.
    #
    # The 6x6x6 multiplication tensor has 6*6*6 = 216 "natural" terms (schoolbook).
    #
    # Block decomposition: 6 = 3*2. View A as 3x3 blocks of 2x2.
    # C_block[I,K] = sum_J A_block[I,J] * B_block[J,K]
    #
    # For the block-level 3x3x3 tensor, schoolbook uses 27 terms.
    # For each block multiplication (2x2x2), schoolbook uses 8 terms, Strassen uses 7.
    #
    # So total rank with Strassen at block level: 27 * 7 = 189.
    # Total rank with schoolbook at block level: 27 * 8 = 216 (same as full schoolbook).
    #
    # Let's implement the 27*7 = 189 decomposition.
    #
    # For each (I, J, K) in 0..2 (block indices):
    #   For each Strassen term r in 0..6:
    #     Determine which A and B entries are involved.
    #     For each such (a_entry, b_entry) pair, create a rank-1 term.
    #
    # Strassen terms for 2x2x2 (using 0-based indexing for 2x2 blocks):
    # A_block = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_block = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    #
    # M0 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    #   = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    #   4 terms
    #
    # M1 = (A[1,0] + A[1,1]) * B[0,0]
    #   = A[1,0]*B[0,0] + A[1,1]*B[0,0]
    #   2 terms
    #
    # M2 = A[0,0] * (B[0,1] - B[1,1])
    #   = A[0,0]*B[0,1] - A[0,0]*B[1,1]
    #   2 terms
    #
    # M3 = A[1,1] * (B[1,0] - B[0,0])
    #   = A[1,1]*B[1,0] - A[1,1]*B[0,0]
    #   2 terms
    #
    # M4 = (A[0,0] + A[0,1]) * B[1,1]
    #   = A[0,0]*B[1,1] + A[0,1]*B[1,1]
    #   2 terms
    #
    # M5 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    #   = A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    #   4 terms
    #
    # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #   = A[0,1]*B[1,0] + A[0,1]*B[1,1] - A[1,1]*B[1,0] - A[1,1]*B[1,1]
    #   4 terms
    #
    # Total terms per block multiplication: 4+2+2+2+2+4+4 = 20
    # Total for 27 block multiplications: 27 * 20 = 540
    #
    # That's worse than 216! Something is wrong.
    #
    # The issue is that Strassen's 7 "multiplications" are of sub-matrices, not scalars.
    # When we expand each sub-matrix multiplication into scalar terms, we get more terms.
    #
    # The correct way: Strassen reduces the number of scalar multiplications from 8 to 7 for 2x2x2.
    # So for 2x2x2, Strassen gives rank 7, not 20.
    #
    # Let me reconsider. The 2x2x2 multiplication tensor has rank 7 (Strassen).
    # So for each block multiplication, we use 7 rank-1 terms, not 20.
    #
    # How does Strassen's rank-7 decomposition work in terms of the tensor?
    #
    # The 2x2x2 tensor T_{i,j,k} for i,j,k in {0,1}:
    # T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
    #
    # Strassen's decomposition:
    # C_{0,0} = M0 + M3 - M4 + M6
    # C_{0,1} = M2 + M4
    # C_{1,0} = M1 + M3
    # C_{1,1} = M0 - M1 + M2 + M5
    #
    # Where:
    # M0 = (A_{0,0}+A_{1,1})(B_{0,0}+B_{1,1})
    # M1 = (A_{1,0}+A_{1,1})B_{0,0}
    # M2 = A_{0,0}(B_{0,1}-B_{1,1})
    # M3 = A_{1,1}(B_{1,0}-B_{0,0})
    # M4 = (A_{0,0}+A_{0,1})B_{1,1}
    # M5 = (A_{1,0}-A_{0,0})(B_{0,0}+B_{0,1})
    # M6 = (A_{0,1}-A_{1,1})(B_{1,0}+B_{1,1})
    #
    # Each M_r is a product of a linear combination of A entries and a linear combination of B entries.
    # So M_r = (sum_i alpha_{r,i} A_i) * (sum_j beta_{r,j} B_j)
    #
    # This is a rank-1 term in the tensor: U_{i,r} = alpha_{r,i}, V_{j,r} = beta_{r,j}, W_{k,r} = coefficient in C.
    #
    # So for each r in 0..6, we have one rank-1 term with:
    # U_{i,r} = coefficient of A_i in the A-part of M_r
    # V_{j,r} = coefficient of B_j in the B-part of M_r
    # W_{k,r} = coefficient of C_k in the sum of M_r's that contribute to C_k
    #
    # Let's define the 2x2x2 Strassen decomposition explicitly.
    # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C indices: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    #
    # M0: A-part = A00 + A11, B-part = B00 + B11
    #   U: A00=1, A11=1
    #   V: B00=1, B11=1
    #   W: C00=1, C11=1
    #
    # M1: A-part = A10 + A11, B-part = B00
    #   U: A10=1, A11=1
    #   V: B00=1
    #   W: C10=1, C11=-1
    #
    # M2: A-part = A00, B-part = B01 - B11
    #   U: A00=1
    #   V: B01=1, B11=-1
    #   W: C01=1, C11=1
    #
    # M3: A-part = A11, B-part = B10 - B00
    #   U: A11=1
    #   V: B10=1, B00=-1
    #   W: C00=1, C10=1
    #
    # M4: A-part = A00 + A01, B-part = B11
    #   U: A00=1, A01=1
    #   V: B11=1
    #   W: C00=-1, C01=1
    #
    # M5: A-part = A10 - A00, B-part = B00 + B01
    #   U: A10=1, A00=-1
    #   V: B00=1, B01=1
    #   W: C11=1
    #
    # M6: A-part = A01 - A11, B-part = B10 + B11
    #   U: A01=1, A11=-1
    #   V: B10=1, B11=1
    #   W: C00=1
    #
    # Let me verify C00: M0 + M3 - M4 + M6 = 1 + 1 - 1 + 1 = 2? That's not right.
    # C00 should be A00*B00 + A01*B10.
    #
    # Let me re-derive Strassen carefully.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Using 1-based indexing. Let me convert to 0-based:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # Verify C00 = A00*B00 + A01*B10:
    # M0 contributes: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M3 contributes: A11*B10 - A11*B00
    # -M4 contributes: -A00*B11 - A01*B11
    # M6 contributes: A01*B10 + A01*B11 - A11*B10 - A11*B11
    #
    # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10. Correct!
    #
    # So the W coefficients are:
    # M0: C00=1, C11=1
    # M1: C10=1, C11=-1
    # M2: C01=1, C11=1
    # M3: C00=1, C10=1
    # M4: C00=-1, C01=1
    # M5: C11=1
    # M6: C00=1
    #
    # Now for the 6x6x6 case with 3x3 blocks of 2x2:
    # Block indices I, J, K in 0..2.
    # A entry (i,j) where i = 2*I + a, j = 2*J + b.
    # B entry (j,k) where j = 2*J + b, k = 2*K + c.
    # C entry (i,k) where i = 2*I + a, k = 2*K + c.
    #
    # For each (I, J, K) and each Strassen term r in 0..6:
    #   For each (a, b) where U_{a,r} != 0:
    #     For each (b', c) where V_{b',r} != 0:
    #       Wait, no. The Strassen term r has:
    #       U part: coefficients for A entries within the 2x2 block.
    #       V part: coefficients for B entries within the 2x2 block.
    #       W part: coefficients for C entries within the 2x2 block.
    #
    #   The rank-1 term for (I, J, K, r) is:
    #     U_{i*m+j, col} = U_a_r where i = 2I+a, j = 2J+b, and U_a_r is the coefficient of A_{a,b} in the A-part of M_r.
    #     Wait, the A-part of M_r is a sum over (a,b) of alpha_{a,b} A_{a,b}.
    #     Similarly for B and C.
    #
    #   So for each (I, J, K, r):
    #     For each (a, b) with alpha_{a,b,r} != 0:
    #       For each (b', c) with beta_{b',c,r} != 0:
    #         This doesn't work because U and V are separate factors.
    #
    #   Actually, in the tensor decomposition:
    #     T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    #
    #   For the Kronecker product structure:
    #     T_{i,j,k} = T_block_{I,J,K} * T_2x2_{a,b,c}
    #     where i = 2I+a, j = 2J+b, k = 2K+c.
    #
    #   T_block_{I,J,K} = 1 if J is the intermediate index (always true for schoolbook block structure).
    #   Actually, T_block_{I,J,K} = 1 for all I,J,K in schoolbook.
    #
    #   So T_{i,j,k} = T_2x2_{a,b,c} when i=2I+a, j=2J+b, k=2K+c.
    #
    #   The rank-7 decomposition of T_2x2:
    #     T_2x2_{a,b,c} = sum_r U2_{a,b,r} * V2_{b,c,r} * W2_{a,c,r}
    #
    #   So T_{i,j,k} = sum_r U2_{a,b,r} * V2_{b,c,r} * W2_{a,c,r}
    #
    #   This means for each (I, J, K, r):
    #     U_{(2I+a)*m + (2J+b), col} = U2_{a,b,r}
    #     V_{(2J+b)*p + (2K+c), col} = V2_{b,c,r}
    #     W_{(2K+c)*n + (2I+a), col} = W2_{a,c,r}
    #
    #   But wait, this would give rank 27*7 = 189, and each term is independent.
    #   Let's verify: for a given (i,j,k) = (2I+a, 2J+b, 2K+c):
    #     T_{i,j,k} = sum_{I',J',K',r} U_{i*m+j, col(I',J',K',r)} * V_{j*p+k, col(I',J',K',r)} * W_{k*n+i, col(I',J',K',r)}
    #
    #     U_{i*m+j, col} = U2_{a,b,r} if I'=I, J'=J, else 0.
    #     V_{j*p+k, col} = V2_{b,c,r} if J'=J, K'=K, else 0.
    #     W_{k*n+i, col} = W2_{a,c,r} if K'=K, I'=I, else 0.
    #
    #     So the sum is over r only, with I'=I, J'=J, K'=K:
    #     = sum_r U2_{a,b,r} * V2_{b,c,r} * W2_{a,c,r} = T_2x2_{a,b,c}. Correct!
    #
    # So the construction is:
    # rank = 27 * 7 = 189
    # For each (I, J, K) in 0..2 and r in 0..6:
    #   col = I*21 + J*7 + K*... no, let's use col = (I*3 + J)*7 + K*... 
    #   Actually, col = I*21 + J*7 + r? No, K is also a dimension.
    #   col = (I*3 + J)*7 + r? That's only 27*7 = 189 columns, but we need to index by (I,J,K,r).
    #   3*3*3*7 = 189. So col = ((I*3 + J)*3 + K)*7 + r.
    #
    #   For each (a, b) with U2_{a,b,r} != 0:
    #     i = 2*I + a, j = 2*J + b
    #     U[i*m + j, col] = U2[a, b, r]
    #
    #   For each (b, c) with V2_{b,c,r} != 0:
    #     j = 2*J + b, k = 2*K + c
    #     V[j*p + k, col] = V2[b, c, r]
    #
    #   For each (a, c) with W2_{a,c,r} != 0:
    #     k = 2*K + c, i = 2*I + a
    #     W[k*n + i, col] = W2[a, c, r]
    #
    # Let me define U2, V2, W2 for the 2x2x2 Strassen decomposition.
    # Using 0-based indexing for 2x2 blocks:
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    #
    # M0: A-part = A00 + A11, B-part = B00 + B11, W: C00=1, C11=1
    #   U2: (0,0)=1, (1,1)=1
    #   V2: (0,0)=1, (1,1)=1
    #   W2: (0,0)=1, (1,1)=1
    #
    # M1: A-part = A10 + A11, B-part = B00, W: C10=1, C11=-1
    #   U2: (1,0)=1, (1,1)=1
    #   V2: (0,0)=1
    #   W2: (1,0)=1, (1,1)=-1
    #
    # M2: A-part = A00, B-part = B01 - B11, W: C01=1, C11=1
    #   U2: (0,0)=1
    #   V2: (0,1)=1, (1,1)=-1
    #   W2: (0,1)=1, (1,1)=1
    #
    # M3: A-part = A11, B-part = B10 - B00, W: C00=1, C10=1
    #   U2: (1,1)=1
    #   V2: (1,0)=1, (0,0)=-1
    #   W2: (0,0)=1, (1,0)=1
    #
    # M4: A-part = A00 + A01, B-part = B11, W: C00=-1, C01=1
    #   U2: (0,0)=1, (0,1)=1
    #   V2: (1,1)=1
    #   W2: (0,0)=-1, (0,1)=1
    #
    # M5: A-part = A10 - A00, B-part = B00 + B01, W: C11=1
    #   U2: (1,0)=1, (0,0)=-1
    #   V2: (0,0)=1, (0,1)=1
    #   W2: (1,1)=1
    #
    # M6: A-part = A01 - A11, B-part = B10 + B11, W: C00=1
    #   U2: (0,1)=1, (1,1)=-1
    #   V2: (1,0)=1, (1,1)=1
    #   W2: (0,0)=1
    #
    # Let me code this up.
    import time
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U2[a,b,r], V2[b,c,r], W2[a,c,r]
    U2 = {}
    V2 = {}
    W2 = {}
    
    # M0
    U2[(0,0,0)] = 1; U2[(1,1,0)] = 1
    V2[(0,0,0)] = 1; V2[(1,1,0)] = 1
    W2[(0,0,0)] = 1; W2[(1,1,0)] = 1
    # M1
    U2[(1,0,1)] = 1; U2[(1,1,1)] = 1
    V2[(0,0,1)] = 1
    W2[(1,0,1)] = 1; W2[(1,1,1)] = -1
    # M2
    U2[(0,0,2)] = 1
    V2[(0,1,2)] = 1; V2[(1,1,2)] = -1
    W2[(0,1,2)] = 1; W2[(1,1,2)] = 1
    # M3
    U2[(1,1,3)] = 1
    V2[(1,0,3)] = 1; V2[(0,0,3)] = -1
    W2[(0,0,3)] = 1; W2[(1,0,3)] = 1
    # M4
    U2[(0,0,4)] = 1; U2[(0,1,4)] = 1
    V2[(1,1,4)] = 1
    W2[(0,0,4)] = -1; W2[(0,1,4)] = 1
    # M5
    U2[(1,0,5)] = 1; U2[(0,0,5)] = -1
    V2[(0,0,5)] = 1; V2[(0,1,5)] = 1
    W2[(1,1,5)] = 1
    # M6
    U2[(0,1,6)] = 1; U2[(1,1,6)] = -1
    V2[(1,0,6)] = 1; V2[(1,1,6)] = 1
    W2[(0,0,6)] = 1
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for r in range(7):
                    col = ((I * 3 + J) * 3 + K) * 7 + r
                    if time.time() > deadline:
                        break
                    # U part
                    for a in range(2):
                        for b in range(2):
                            if (a, b, r) in U2:
                                i = 2 * I + a
                                j = 2 * J + b
                                U[i * m + j][col] = U2[(a, b, r)]
                    # V part
                    for b in range(2):
                        for c in range(2):
                            if (b, c, r) in V2:
                                j = 2 * J + b
                                k = 2 * K + c
                                V[j * p + k][col] = V2[(b, c, r)]
                    # W part
                    for a in range(2):
                        for c in range(2):
                            if (a, c, r) in W2:
                                k = 2 * K + c
                                i = 2 * I + a
                                W[k * n + i][col] = W2[(a, c, r)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
