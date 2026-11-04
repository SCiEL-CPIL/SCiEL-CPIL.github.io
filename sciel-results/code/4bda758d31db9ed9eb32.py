"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We will build a decomposition of rank 28.
    # Structure:
    # - Use Strassen (rank 7) for the 2x2x2 block (i=0,1; j=0,1; k=0,1).
    # - Use Schoolbook (rank 3) for the remaining j=2, k=0,1 (i=0,1).
    # - Use Schoolbook (rank 3) for the remaining j=0,1, k=2 (i=0,1).
    # - Use Schoolbook (rank 1) for the corner i=0,1; j=2; k=2.
    # Wait, let's partition carefully to avoid double counting or missing parts.
    
    # Full tensor is sum over i,j,k.
    # Let's define the support of each rank-1 term.
    # A rank-1 term is a triad (u, v, w).
    # u is length 6 (n*m), v is length 15 (m*p), w is length 10 (p*n).
    
    # We can view the multiplication as a block matrix.
    # A is 2x3, B is 3x5.
    # Let's split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    # Let's split B into B_top (2x5, rows 0,1) and B_bottom (1x5, row 2).
    # Then AB = A_left @ B_top + A_left @ B_bottom + A_right @ B_top + A_right @ B_bottom.
    
    # Term 1: A_left @ B_top. This is a 2x2 @ 2x5 multiplication.
    # We can split B_top into B_top_left (2x2, cols 0,1) and B_top_right (2x3, cols 2,3,4).
    # A_left @ B_top_left: 2x2 @ 2x2. Use Strassen (rank 7).
    # A_left @ B_top_right: 2x2 @ 2x3. This is 6 scalar multiplications (2*2*3).
    # Can we do better? 
    # 2x2 @ 2x3 is equivalent to 2x2 @ 2x2 + 2x2 @ 2x1.
    # Or just schoolbook 6.
    # Actually, 2x2 @ 2x3 can be done in rank 6? Yes, schoolbook is 6.
    # Is there a faster algorithm?
    # The 2x2 @ 2x3 tensor is the same as two 2x2 @ 2x1 tensors? No.
    # It's the sum of 6 rank-1 terms.
    # Let's stick to schoolbook for non-square or non-power-of-2 parts to ensure correctness, 
    # unless I am confident.
    # The 2x2 @ 2x2 part is the main gain (7 vs 8).
    
    # Term 2: A_left @ B_bottom. 2x2 @ 1x5.
    # This is 2 * 1 * 5 = 10 scalar products?
    # A_left (2x2) times B_bottom (1x5).
    # Result is 2x5.
    # Each entry C[i, k] = sum_j A[i, j] B[j, k].
    # Here j=0,1. B is row 2. So B[2, k].
    # C[i, k] = A[i, 0] B[2, k] + A[i, 1] B[2, k] = (A[i, 0] + A[i, 1]) B[2, k].
    # This can be done in rank 2!
    # For each i (0,1), we compute S_i = A[i, 0] + A[i, 1].
    # Then C[i, k] = S_i * B[2, k].
    # So we need 2 terms for the S_i calculation?
    # Wait, the decomposition is global.
    # A term is (u, v, w).
    # If we set u to have 1 at i,0 and i,1, v to have 1 at 2,k, w to have 1 at k,i...
    # Let's check:
    # Term (u, v, w): u[i,0]=1, u[i,1]=1. v[2, k]=1. w[k, i]=1.
    # Contribution to C[i, k] is u[i,0]v[0,2]? No, indices.
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r).
    # Index U: i*m + j. Index V: j*p + k. Index W: k*n + i.
    # C[i, k] = sum_r U[i*m+?, r] * V[?*p+k, r] * W[k*n+i, r]?
    # The standard formula is C = U^T M V? No.
    # The tensor T_{ijk} = C_{ik} from A_{ij} B_{jk}.
    # T = sum_r U_r \otimes V_r \otimes W_r.
    # T_{ijk} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    # The contract says:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Usually, this implies the tensor is sum_r U_r \otimes V_r \otimes W_r where U_r is a vector of size nm, etc.
    # And the multiplication is:
    # C_{ik} = sum_j sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}?
    # No, the standard decomposition is:
    # C_{ik} = sum_r (sum_j A_{ij} U_{ij}^{(r)}) (sum_k B_{jk} V_{jk}^{(r)})? No.
    
    # Let's look at the seed code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to:
    # C_{ik} = sum_j sum_r A_{ij} B_{jk} W_{ki}^{(r)}?
    # Actually, if U, V, W are 1 at specific indices, the term contributes A_{ij} B_{jk} to C_{ik} if W_{ki}=1.
    # So the condition is:
    # sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = delta_{ij} delta_{jk} (in the tensor sense, i.e. 1 if we are multiplying A_ij and B_jk to form C_ik).
    # Wait, the tensor T has entries T_{i,j,k} = 1 if we want the product A_{ij} B_{jk} to contribute to C_{ik}.
    # Actually, the multiplication tensor is T_{i,j,k} = 1.
    # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # So T_{i,j,k} = sum_r u_r[i,j] v_r[j,k] w_r[k,i].
    # (Note: W is indexed by k*n+i, which is (k,i)).
    
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k?
    # No, that would be the identity tensor.
    # The matrix multiplication tensor is T_{i,j,k} = 1.
    # Wait, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
    # So yes, T_{i,j,k} = 1.
    # And we want to decompose T as sum of rank-1 tensors.
    # Rank-1 tensor is u \otimes v \otimes w where u_{ij}, v_{jk}, w_{ki}.
    # So we need sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1 for all i,j,k.
    
    # Let's re-evaluate the "Strassen" part.
    # For 2x2x2 (i,j,k in {0,1}), we need sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1.
    # Standard Strassen gives rank 7.
    
    # Let's re-evaluate the "A_left @ B_bottom" part.
    # i in {0,1}, j in {0,1}, k in {2,3,4}. (B_bottom is row 2, so j=2? No.)
    # In A_left @ B_bottom:
    # A_left is cols 0,1 of A. So j=0,1.
    # B_bottom is row 2 of B. So j=2.
    # Wait, the multiplication is sum_j A_{ij} B_{jk}.
    # If we restrict j=0,1 for A and j=2 for B, there is no overlap in j!
    # A_{ij} uses j. B_{jk} uses j.
    # So A_left (j=0,1) multiplies B_top (j=0,1).
    # A_right (j=2) multiplies B_bottom (j=2).
    # There is no cross term like A_left * B_bottom because the index j must match.
    
    # So the partition is:
    # 1. j in {0,1}: A_{i, 0..1} * B_{0..1, k}.
    #    This is a 2x2 @ 2x5 multiplication.
    #    Split k into {0,1} and {2,3,4}.
    #    a) k in {0,1}: 2x2 @ 2x2. (Rank 7 Strassen).
    #    b) k in {2,3,4}: 2x2 @ 2x3. (Rank 6 Schoolbook).
    # 2. j=2: A_{i, 2} * B_{2, k}.
    #    This is a 2x1 @ 1x5 multiplication.
    #    a) k in {0,1,2,3,4}: 2x1 @ 1x5.
    #    C_{ik} = A_{i,2} B_{2,k}.
    #    This is rank 2?
    #    T_{i,2,k} = 1.
    #    We need sum_r U_{i,2}^{(r)} V_{2,k}^{(r)} W_{k,i}^{(r)} = 1.
    #    If we set U_{i,2}=1, V_{2,k}=1, W_{k,i}=1, that's rank 10 (for all i,k).
    #    Can we reduce?
    #    U_{i,2} depends only on i. V_{2,k} depends only on k.
    #    This is an outer product of a vector of size 2 (i) and size 5 (k), mapped to W.
    #    Actually, T_{i,2,k} = 1 is a rank 1 tensor?
    #    No, T_{i,2,k} is non-zero for all i,k.
    #    It is the outer product of e_2 (in j) with I_2 (in i) and I_5 (in k)?
    #    Let's check: u_{i,2} = \delta_{i,i'}? No.
    #    u_{i,2} is 1 for all i? No, u is a vector in A-space.
    #    If u_{i,2} = 1 for all i, v_{2,k} = 1 for all k, w_{k,i} = 1 for all k,i.
    #    Then sum_r ... = 1 * 1 * 1 = 1.
    #    So YES, this is rank 1!
    #    U has 1 at (i,2) for i=0,1.
    #    V has 1 at (2,k) for k=0..4.
    #    W has 1 at (k,i) for k=0..4, i=0,1.
    #    This single term covers all 10 entries.
    
    # So total rank:
    # 1. 2x2 @ 2x2 (j=0,1; k=0,1): Rank 7.
    # 2. 2x2 @ 2x3 (j=0,1; k=2,3,4): Rank 6 (Schoolbook).
    # 3. 2x1 @ 1x5 (j=2; k=0..4): Rank 1.
    # Total Rank = 7 + 6 + 1 = 14.
    
    # Wait, is 2x2 @ 2x3 really rank 6?
    # T_{i,j,k} for i in {0,1}, j in {0,1}, k in {2,3,4}.
    # This is 2*2*3 = 12 entries.
    # Schoolbook is 6.
    # Can it be done in less?
    # 2x2 @ 2x1 is rank 2?
    # T_{i,j,k} (k fixed).
    # C_{ik} = A_{i0} B_{0k} + A_{i1} B_{1k}.
    # This is rank 2?
    # Term 1: U_{i0}=1, V_{0k}=1, W_{ki}=1. (Covers A_{i0} B_{0k}).
    # Term 2: U_{i1}=1, V_{1k}=1, W_{ki}=1. (Covers A_{i1} B_{1k}).
    # So 2x2 @ 2x1 is rank 2.
    # 2x2 @ 2x3 is sum of three 2x2 @ 2x1 blocks (k=2, k=3, k=4).
    # Rank = 2 + 2 + 2 = 6.
    # So yes, 6 is correct for schoolbook.
    
    # So Rank 14 is possible?
    # Let's check if there are overlaps or if I missed something.
    # The tensor is the sum of these disjoint blocks?
    # The indices (i,j,k) partition the space.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}.
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4}.
    # Block 3: i in {0,1}, j=2, k in {0,1,2,3,4}.
    # These cover all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    # Since the supports are disjoint in (i,j,k), the rank of the sum is the sum of the ranks.
    # So Rank = 7 + 6 + 1 = 14.
    
    # This is a huge improvement from 30.
    
    # Let's verify the Strassen decomposition for 2x2x2.
    # We need 7 terms (u_r, v_r, w_r) such that sum_r u_{ij} v_{jk} w_{ki} = 1.
    # Standard Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map this to U, V, W.
    # U_{ij}^{(r)}, V_{jk}^{(r)}, W_{ki}^{(r)}.
    # Term 1:
    # u1: A00, A11 are 1. -> u1[0,0]=1, u1[1,1]=1.
    # v1: B00, B11 are 1. -> v1[0,0]=1, v1[1,1]=1.
    # w1: C00 gets +1, C11 gets +1. -> w1[0,0]=1, w1[1,1]=1.
    # Check: sum u1 v1 w1.
    # (0,0,0): 1*1*1 = 1.
    # (1,1,1): 1*1*1 = 1.
    # (0,0,1): 1*0*0 = 0.
    # (0,1,0): 0*1*0 = 0.
    # (1,0,0): 0*0*1 = 0.
    # (1,1,0): 1*0*0 = 0.
    # (0,1,1): 0*1*0 = 0.
    # (1,0,1): 0*0*1 = 0.
    # Wait, this only covers (0,0,0) and (1,1,1).
    # But M1 contributes to C00 and C11.
    # M1 = (A00+A11)(B00+B11).
    # The term in the tensor expansion is:
    # (A00+A11)(B00+B11) * (delta_{i0}delta_{k0} + delta_{i1}delta_{k1})?
    # No, the product is a scalar. It is multiplied by 1 to form the tensor entry?
    # The tensor T_{ijk} is the coefficient.
    # The expansion of M1 in terms of A_{ij} B_{jk} is:
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # So it contributes to (0,0,0), (0,1,1), (1,0,0), (1,1,1).
    # Wait, my w1 was only 1 at (0,0) and (1,1).
    # If w1[0,0]=1 and w1[1,1]=1, then:
    # T_{0,0,0} += u1[0,0] v1[0,0] w1[0,0] = 1.
    # T_{0,1,1} += u1[0,1] v1[1,1] w1[1,0]? No.
    # Indices: i,j,k.
    # u1[i,j], v1[j,k], w1[k,i].
    # For (0,1,1): i=0, j=1, k=1.
    # u1[0,1]=0. So 0.
    # But M1 has an A00 B11 term. A00 is i=0,j=0. B11 is j=1,k=1.
    # So the term is A00 * B11. This corresponds to (i=0, j=0, k=1)? No.
    # A_{ij} is indexed by i,j. B_{jk} by j,k.
    # A00 is i=0, j=0.
    # B11 is j=1, k=1.
    # They cannot be multiplied directly because j must match (0 != 1).
    # So M1 = (A00+A11)(B00+B11) is NOT a sum of valid A_{ij}B_{jk} terms?
    # A00 (j=0) multiplies B00 (j=0) and B10? No B11 (j=1).
    # So (A00+A11)(B00+B11) = A00B00 + A00B11 (invalid) + A11B00 (invalid) + A11B11.
    # So M1 is not a valid intermediate product in the standard algebraic sense unless we assume specific structures?
    # No, Strassen works because we compute these products of sums.
    # But the tensor decomposition represents the *linear* combination of scalar multiplications.
    # The scalar multiplication is x * y.
    # If we compute M1 = (A00+A11)(B00+B11), this is ONE scalar multiplication.
    # The inputs to this multiplication are S_A = A00+A11 and S_B = B00+B11.
    # So u1 should be the coefficients of S_A. u1[0,0]=1, u1[1,1]=1.
    # v1 should be the coefficients of S_B. v1[0,0]=1, v1[1,1]=1.
    # The result M1 is used to form C00 and C11.
    # C00 += M1. C11 += M1.
    # So w1[0,0]=1, w1[1,1]=1.
    # Does this decompose the tensor?
    # T_{0,0,0} += 1*1*1 = 1.
    # T_{1,1,1} += 1*1*1 = 1.
    # T_{0,1,0} += 0*1*0 = 0.
    # T_{0,0,1} += 1*0*0 = 0.
    # T_{0,1,1} += 0*1*1 = 0.
    # T_{1,0,1} += 0*0*1 = 0.
    # T_{1,0,0} += 0*0*1 = 0.
    # T_{1,1,0} += 1*0*0 = 0.
    # So this term only contributes to (0,0,0) and (1,1,1).
    # But the original tensor T has 1s at (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1).
    # Wait, T_{ijk} = 1 for all i,j,k?
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So for a fixed i,k, we sum over j.
    # The tensor T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # This coefficient is 1 for all i,j,k.
    # So T_{0,0,0}=1, T_{0,0,1}=1 (A00 B01 contributes to C01? No, A00 B01 -> C01. So T_{0,0,1} corresponds to i=0,j=0,k=1. C_{0,1}. Yes.)
    # So T is the all-ones tensor of size 2x2x2.
    
    # So my Term 1 only covers 2 of the 8 entries.
    # We need to cover all 8.
    # The other terms will cover the rest.
    
    # Let's verify the sum of the 7 Strassen terms equals the all-ones tensor.
    # This is a known fact.
    
    # So, Rank 14 is likely correct.
    
    # Implementation:
    # U: 6x14
    # V: 15x14
    # W: 10x14
    
    # Block 1: 2x2x2 Strassen (7 terms)
    # Indices i,j,k in 0,1.
    # U cols 0-5, V rows 0-3 (j*p+k: 0*5+0..1, 1*5+0..1), W rows 0-3 (k*n+i: 0*2+0..1, 1*2+0..1).
    
    # Strassen Terms (u, v, w)
    # 1. u(0,0;1,1), v(0,0;1,1), w(0,0;1,1) -> C00+, C11+
    # 2. u(1,0;1,1), v(0,0), w(1,0) -> C10+ (M2: (A10+A11)B00 -> C10). w(1,0) means C_{1,0}.
    #    Check: M2 = (A10+A11)B00. C10 += M2.
    #    u2[1,0]=1, u2[1,1]=1. v2[0,0]=1. w2[0,1]=1? No, w is indexed k*n+i.
    #    C_{1,0} -> i=1, k=0. w[0*2+1] = w[1] = 1.
    # 3. u(0,0), v(0,1;1,1), w(0,1) -> C01+ (M3: A00(B01-B11)).
    #    u3[0,0]=1. v3[0,1]=1, v3[1,1]=-1. w3[1*2+0]=w3[2]=1. (C_{0,1}).
    # 4. u(1,1), v(1,0;0,0), w(1,0) -> C10+ (M4: A11(B10-B00)).
    #    u4[1,1]=1. v4[1,0]=1, v4[0,0]=-1. w4[0*2+1]=w4[1]=1. (C_{1,0}).
    # 5. u(0,0;0,1), v(1,1), w(0,0) -> C00- (M5: (A00+A01)B11).
    #    u5[0,0]=1, u5[0,1]=1. v5[1,1]=1. w5[0*2+0]=w5[0]=-1. (C_{0,0}).
    # 6. u(1,0;0,0), v(0,0;0,1), w(1,1) -> C11+ (M6: (A10-A00)(B00+B01)).
    #    u6[1,0]=1, u6[0,0]=-1. v6[0,0]=1, v6[0,1]=1. w6[1*2+1]=w6[3]=1. (C_{1,1}).
    # 7. u(0,1;1,1), v(1,0;1,1), w(0,0) -> C00+ (M7: (A01-A11)(B10+B11)).
    #    u7[0,1]=1, u7[1,1]=-1. v7[1,0]=1, v7[1,1]=1. w7[0*2+0]=w7[0]=1. (C_{0,0}).
    
    # Let's double check signs and indices.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My w assignments:
    # T1: w(0,0)=1, w(1,1)=1. (M1). OK.
    # T2: w(1,0)=1. (M2). OK.
    # T3: w(0,1)=1. (M3). OK.
    # T4: w(1,0)=1. (M4). OK.
    # T5: w(0,0)=-1. (M5). OK.
    # T6: w(1,1)=1. (M6). OK.
    # T7: w(0,0)=1. (M7). OK.
    
    # Check C00: T1(1) + T4(1) + T5(-1) + T7(1) = 2?
    # Wait.
    # T1 contributes to C00 via w(0,0)=1.
    # T4 contributes to C10 via w(1,0)=1. Does T4 contribute to C00?
    # T4: u(1,1), v(1,0;0,0), w(1,0).
    # i=1, j=1. v has j=1 and j=0.
    # If j=1, k=0. w[0,1]=1. So contributes to C_{1,0}.
    # If j=0, k=0. w[0,1]=1. But u(1,0)=0. So 0.
    # So T4 only contributes to C10.
    # But M4 is in C00 formula?
    # C00 = M1 + M4 - M5 + M7.
    # My T4 w was w(1,0)=1 (C10).
    # M4 = A11(B10-B00).
    # This contributes to C10?
    # A11 (i=1, j=1). B10 (j=1, k=0). Product A11B10 -> C10.
    # A11 (i=1, j=1). B00 (j=0, k=0). Invalid (j mismatch).
    # So M4 only contributes to C10?
    # Standard Strassen:
    # C10 = M2 + M4.
    # C00 = M1 + M4 - M5 + M7.
    # This implies M4 contributes to both?
    # M4 = A11(B10 - B00).
    # A11B10 -> C10.
    # A11B00 -> Invalid.
    # So M4 is just A11B10?
    # No, Strassen works on 2x2 matrices.
    # M4 = A11 * (B10 - B00).
    # B10 is element (1,0). B00 is (0,0).
    # A11 is (1,1).
    # A11 * B10 -> (1,1)*(1,0) -> C10.
    # A11 * B00 -> (1,1)*(0,0) -> Invalid.
    # So why is M4 in C00?
    # Ah, the standard Strassen formulas are:
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check M4 in C00.
    # M4 = A11(B10 - B00).
    # If we expand C00 = M1 + M4 - M5 + M7.
    # The term M4 contributes A11B10 to C00?
    # A11B10 is A[1,1]B[1,0]. This should contribute to C[1,0].
    # So why is it in C00?
    # Maybe my indices for Strassen are swapped?
    # Some sources define M4 differently or C differently.
    # Let's stick to the one where M4 contributes to C10 only?
    # If M4 is A11B10, it is in C10.
    # If it is in C00 formula, there is a sign error or index swap.
    # Let's use the code to verify or find a reliable Strassen tensor.
    
    # To be safe, I will use the schoolbook for the 2x2x2 block (Rank 8).
    # Total Rank = 8 + 6 + 1 = 15.
    # This is still better than 30.
    # And 15 is safer than 14 if I get Strassen wrong.
    # Actually, let's try to get Strassen right.
    # I will use the explicit tensor vectors from a reliable source or derive them.
    # If I fail, I fallback to 15.
    
    # Let's try 15 (Schoolbook for 2x2x2).
    # 8 terms for 2x2x2.
    # 6 terms for 2x2x3.
    # 1 term for 2x1x5.
    # Total 15.
    
    U = np.zeros((n*m, 15))
    V = np.zeros((m*p, 15))
    W = np.zeros((p*n, 15))
    
    col = 0
    # Block 1: 2x2x2 (i=0,1; j=0,1; k=0,1)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*3 + j, col] = 1
                V[j*5 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    # col = 8
    
    # Block 2: 2x2x3 (i=0,1; j=0,1; k=2,3,4)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*3 + j, col] = 1
                V[j*5 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    # col = 14
    
    # Block 3: 2x1x5 (i=0,1; j=2; k=0,1,2,3,4)
    # Rank 1 term.
    for i in range(2):
        for k in range(5):
            U[i*3 + 2, col] = 1
            V[2*5 + k, col] = 1
            W[k*2 + i, col] = 1
    # col = 15
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
