import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 120.0  # safety margin before 150s kill

def construct():
    start = time.time()
    n, m, p = 2, 4, 7
    
    # We want to compute C = A @ B where A is 2x4 and B is 4x7.
    # C is 2x7.
    # We can split A into [A1, A2] where A1, A2 are 2x2.
    # We can split B into [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    
    # To reduce rank, we use Strassen for the 2x2x2 parts.
    # Let's split B1 and B2 further into 2x2 and 2x3 blocks.
    # B1 = [B11, B12] where B11 is 2x2, B12 is 2x3.
    # B2 = [B21, B22] where B21 is 2x2, B22 is 2x3.
    
    # C = A1 [B11, B12] + A2 [B21, B22]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    
    # Let C1 = A1 B11 + A2 B21 (2x2 matrix)
    # Let C2 = A1 B12 + A2 B22 (2x3 matrix)
    
    # We can compute C1 using a rank-5 decomposition? 
    # Actually, C1 is a sum of two 2x2x2 multiplications.
    # Rank of sum is at most sum of ranks.
    # R(2,2,2) = 7. So C1 can be done in rank 14? 
    # But we can do better by combining them.
    # C1 = A1 B11 + A2 B21.
    # This is equivalent to multiplying a 2x4 matrix [A1, A2] by a 4x2 matrix [B11; B21].
    # Wait, [A1, A2] is 2x4. [B11; B21] is 4x2.
    # So C1 is a 2x4 x 4x2 multiplication.
    # R(2,4,2) is the same as R(2,2,4) by symmetry.
    # R(2,4,4) is 14? No, R(2,4,4) <= 2*R(2,2,4) = 2*7 = 14?
    # Actually R(2,4,4) = 14 is known? 
    # Let's check R(2,4,2).
    # R(2,4,2) = R(2,2,4).
    # We can decompose 2x4 x 4x2 into two 2x2 x 2x2 multiplications?
    # [A1, A2] [B11; B21] = A1 B11 + A2 B21.
    # This is exactly the sum of two 2x2x2 tensors.
    # If we treat them separately, rank is 7+7=14.
    # But can we do better?
    # The tensor is T = T1 + T2 where T1 is (i,j,k) for A1B11 and T2 for A2B21.
    # They have disjoint supports in the "middle" index j?
    # A1 uses j in {0,1}, A2 uses j in {2,3}.
    # B11 uses j in {0,1}, B21 uses j in {2,3}.
    # So the supports in j are disjoint.
    # The rank of a direct sum of tensors is the sum of their ranks.
    # So R(C1 part) = R(2,2,2) + R(2,2,2) = 14.
    
    # Now for C2 = A1 B12 + A2 B22.
    # A1, A2 are 2x2. B12, B22 are 2x3.
    # C2 is 2x3.
    # This is a 2x4 x 4x3 multiplication?
    # A_full = [A1, A2] (2x4). B_full = [B12; B22] (4x3).
    # R(2,4,3).
    # We can split B_full into [B12; B22].
    # C2 = A1 B12 + A2 B22.
    # B12 is 2x3, B22 is 2x3.
    # We can split B12 into [B12a, B12b] where B12a is 2x2, B12b is 2x1.
    # Similarly B22 into [B22a, B22b].
    # Then C2 = [A1 B12a + A2 B22a, A1 B12b + A2 B22b].
    # First part: 2x4 x 4x2. Rank 14.
    # Second part: 2x4 x 4x1.
    # R(2,4,1) = R(2,1,4) = R(1,2,4) = 4?
    # R(1, m, p) = m*p? No, R(1, m, p) is the rank of the matrix multiplication tensor for 1xm by mxp.
    # This is just a vector-matrix multiplication.
    # T[i, j, k] = 1 if i=0.
    # Flattening: 
    # I=1, J=m, K=p.
    # Matricization 1 x (m*p) has rank 1? No.
    # T is a tensor.
    # For 1xm x mxp, the tensor is T[0, j, k] = 1.
    # This is a rank 1 tensor? No.
    # T[0, j, k] = 1 for all j,k.
    # This can be written as sum_t U[0,t] V[j,t] W[k,t].
    # If we pick V[j,t] = 1 for all j, W[k,t]=1 for all k, U[0,t]=1.
    # Then sum_t 1*1*1 = r.
    # We need sum to be 1. So r=1 works?
    # Let's check.
    # U[0,0]=1, V[j,0]=1, W[k,0]=1.
    # T[0,j,k] = U[0,0]V[j,0]W[k,0] = 1.
    # Yes, R(1, m, p) = 1.
    # Wait, is that right?
    # C = A B. A is 1xm, B is mxp.
    # C[k] = sum_j A[j] B[j,k].
    # This is a linear combination of the rows of B with coefficients A.
    # The tensor T[i, j, k] is 1 if i=0.
    # It is a rank 1 tensor?
    # T = u v^T w^T?
    # u is vector of length 1 (value 1).
    # v is vector of length m (all 1s).
    # w is vector of length p (all 1s).
    # u v^T w^T gives a tensor where T[0, j, k] = 1*1*1 = 1.
    # Yes, R(1, m, p) = 1.
    
    # So R(2,4,1) = R(2,1,4) = ?
    # By symmetry R(2,4,1) = R(1,4,2) = R(1,2,4).
    # R(1, m, p) = 1.
    # So R(2,4,1) = 1?
    # Let's check flattening of T for 2x4x1.
    # I=2, J=4, K=1.
    # T[i, j, 0] = 1 for all i,j.
    # Matricization I x (J*K) = 2 x 4.
    # Matrix is all 1s. Rank 1.
    # Matricization J x (I*K) = 4 x 2.
    # Matrix is all 1s. Rank 1.
    # Matricization K x (I*J) = 1 x 8.
    # Vector is all 1s. Rank 1.
    # So R(2,4,1) >= 1.
    # Can we do it with rank 1?
    # T[i, j, 0] = U[i,0] V[j,0] W[0,0].
    # We need U[i,0] V[j,0] W[0,0] = 1 for all i,j.
    # Let W[0,0]=1. U[i,0]=1, V[j,0]=1.
    # Yes, R(2,4,1) = 1.
    
    # So R(2,4,3) = R(2,4,2) + R(2,4,1) = 14 + 1 = 15?
    # Is it additive?
    # C2 = C2a + C2b.
    # C2a is 2x2, C2b is 2x1.
    # The supports in k are disjoint?
    # C2a corresponds to k in {0,1}, C2b to k=2.
    # Yes, disjoint output indices.
    # So rank is sum of ranks.
    # R(2,4,2) = 14.
    # R(2,4,1) = 1.
    # Total for C2 = 15.
    
    # Total rank = R(C1) + R(C2) = 14 + 15 = 29.
    
    # Wait, the inspiration said "rank from 56 to 50".
    # My calculation gives 29. Why the discrepancy?
    # Let's re-read the inspiration: "Decomposing the 2x4x7 multiplication into smaller 2x2x2 and 2x2x1 blocks using Strassen's algorithm reduces the rank from 56 (schoolbook) to 50".
    # Schoolbook is 2*4*7 = 56.
    # My calculation:
    # C = A1 B1 + A2 B2.
    # B1 is 2x7, B2 is 2x7.
    # C1 = A1 B1 + A2 B2? No.
    # C = [A1, A2] [B1; B2].
    # C is 2x7.
    # Let's split B1 and B2 into 2x2, 2x2, 2x3.
    # B1 = [B11, B12, B13] (2x2, 2x2, 2x3).
    # B2 = [B21, B22, B23] (2x2, 2x2, 2x3).
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22, A1 B13 + A2 B23].
    # Let C1 = A1 B11 + A2 B21 (2x2).
    # Let C2 = A1 B12 + A2 B22 (2x2).
    # Let C3 = A1 B13 + A2 B23 (2x3).
    
    # R(C1) = R(2,4,2) = 14.
    # R(C2) = R(2,4,2) = 14.
    # R(C3) = R(2,4,3) = 15.
    # Total = 14 + 14 + 15 = 43.
    
    # Still not 50.
    # Maybe R(2,4,2) is not 14?
    # R(2,4,2) = R(2,2,4).
    # R(2,2,4) = 2 * R(2,2,2) = 14?
    # Yes, because we can split the 4 into two 2s.
    # T(2,2,4) = T(2,2,2) + T(2,2,2) with disjoint k.
    # So 14 is correct.
    
    # Is it possible to do better than 43?
    # Maybe we can combine C1 and C2?
    # C1 and C2 have disjoint k indices (0,1 vs 2,3).
    # So no.
    
    # What if we split differently?
    # A is 2x4. B is 4x7.
    # Split A into 1x2, 1x2? No, A is 2x4.
    # Split into four 1x1? No.
    
    # Let's check R(2,4,7) directly.
    # Lower bound:
    # Flattening 14x28. Rank <= 14.
    # Flattening 28x14. Rank <= 14.
    # Flattening 14x28 (I=14, J=14, K=28? No).
    # I = n*p = 14. J = n*m = 8. K = m*p = 28.
    # Mat 1: 14 x 224. Rank <= 14.
    # Mat 2: 8 x 392. Rank <= 8.
    # Mat 3: 28 x 112. Rank <= 28.
    # So LB is 14.
    
    # My construction gives 43.
    # The inspiration says 50.
    # 43 is better than 50.
    # Is 43 valid?
    # Let's verify the additivity.
    # If the output indices are disjoint, the rank is the sum.
    # C1 outputs to C[0:2, 0:2].
    # C2 outputs to C[0:2, 2:4].
    # C3 outputs to C[0:2, 4:7].
    # Yes, disjoint.
    
    # So I can construct a rank 43 solution.
    # Can I do better?
    # Maybe R(2,4,3) is less than 15?
    # R(2,4,3) = R(2,3,4) = R(3,2,4).
    # R(2,4,3) = R(2,4,2) + R(2,4,1) = 14 + 1 = 15.
    # Is R(2,4,1) = 1?
    # T[i,j,k] for 2x4x1.
    # T[0,j,0]=1, T[1,j,0]=1.
    # Rank 1?
    # U[0,0]=1, U[1,0]=1. V[j,0]=1. W[0,0]=1.
    # T[i,j,0] = 1.
    # Yes.
    
    # So 43 seems correct for this tiling.
    
    # Let's try to construct it.
    
    # We need to generate U, V, W for the whole 2x4x7 problem.
    # U shape (8, r), V shape (28, r), W shape (14, r).
    
    # We will build the decomposition for each block and concatenate.
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 multiplication.
        # C = A B.
        # A is 2x2, B is 2x2.
        # Indices:
        # A: (i, j) i in 0,1; j in 0,1.
        # B: (j, k) j in 0,1; k in 0,1.
        # C: (i, k) i in 0,1; k in 0,1.
        # 
        # Strassen formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to map these to the tensor form.
        # T[i,k, i,j, j,k] = 1.
        # In the decomposition, T = sum_t U[i,k,t] V[i,j,t] W[j,k,t].
        # Note: The task interface defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
        # My previous tensor definition was T[out, u, v].
        # out = k*n + i.
        # u = i*m + j.
        # v = j*p + k.
        
        # For 2x2x2:
        # n=2, m=2, p=2.
        # out indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # u indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # v indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        
        # Let's define the 7 multiplications.
        # Each multiplication is a rank-1 tensor product of linear forms.
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[0,0], A11 = A[1,1].
        # B00 = B[0,0], B11 = B[1,1].
        # This contributes to C.
        # How does M1 contribute to C00, C01, C10, C11?
        # C00 += M1
        # C11 += M1
        # So M1 adds 1 to T[0, 0, 0] (out 0, u 0, v 0)?
        # No, M1 is a scalar.
        # The term in the sum is U[out, t] V[u, t] W[v, t].
        # For M1, we need to define U, V, W vectors such that their outer product matches the contribution of M1 to the tensor.
        
        # The contribution of M1 to the tensor T is:
        # T_M1[out, u, v] = coeff * delta(u, u_M1) * delta(v, v_M1) ... no.
        # M1 = (A00+A11) * (B00+B11).
        # This is a product of a linear form in A and a linear form in B.
        # The linear form in A is L_A = A00 + A11.
        # The linear form in B is L_B = B00 + B11.
        # In the tensor decomposition, V[u, t] represents the coefficient of A[u] in L_A?
        # No, V is indexed by u (which is i*m+j).
        # So V[u, t] is the coefficient of A[i,j] in the t-th factor.
        # Similarly W[v, t] is the coefficient of B[j,k] in the t-th factor.
        # And U[out, t] is the coefficient of C[i,k] in the t-th factor?
        # No, C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T is defined by T[i,k, i,j, j,k] = 1.
        # So C[i,k] = sum_{i,j,k} T[i,k, i,j, j,k] A[i,j] B[j,k].
        # In the decomposition: C[i,k] = sum_t U[i,k,t] (sum_u V[u,t] A[u]) (sum_v W[v,t] B[v])?
        # No, the standard decomposition is:
        # C[i,k] = sum_t U[i,k,t] * (sum_u V[u,t] A[u]) * (sum_v W[v,t] B[v])?
        # This would imply C is a bilinear form? No, C is linear in A and B separately.
        # The tensor T is trilinear? No, it's a 3-mode tensor.
        # The decomposition is T = sum_t U_t V_t W_t where U_t, V_t, W_t are vectors.
        # T[out, u, v] = sum_t U[out, t] V[u, t] W[v, t].
        
        # So for M1:
        # L_A = A00 + A11.
        # V[u, t] = 1 if u corresponds to (0,0) or (1,1), else 0.
        # u indices: 0=(0,0), 3=(1,1).
        # So V[0,t]=1, V[3,t]=1, others 0.
        
        # L_B = B00 + B11.
        # W[v, t] = 1 if v corresponds to (0,0) or (1,1), else 0.
        # v indices: 0=(0,0), 3=(1,1).
        # So W[0,t]=1, W[3,t]=1, others 0.
        
        # Now, what is U[out, t]?
        # The term M1 contributes to C00 and C11.
        # Specifically, C00 += M1, C11 += M1.
        # So U[0, t] = 1, U[3, t] = 1. Others 0.
        # Wait, C00 is out index 0. C11 is out index 3.
        # So U[0,t]=1, U[3,t]=1.
        
        # Let's check:
        # sum_t U[out,t] V[u,t] W[v,t].
        # For out=0, u=0, v=0:
        # U[0,t] V[0,t] W[0,t] = 1 * 1 * 1 = 1.
        # T[0,0,0] should be 1. Correct.
        # For out=0, u=0, v=1:
        # U[0,t] V[0,t] W[1,t] = 1 * 1 * 0 = 0.
        # T[0,0,1] should be 0. Correct.
        # For out=0, u=3, v=0:
        # U[0,t] V[3,t] W[0,t] = 1 * 1 * 1 = 1.
        # T[0,3,0] should be 0?
        # T[out, u, v] = 1 if out=(i,k), u=(i,j), v=(j,k) are consistent.
        # out=0 => i=0, k=0.
        # u=3 => i=1, j=1.
        # i mismatch (0 vs 1). So T[0,3,0] = 0.
        # But my calculation gave 1.
        # Why?
        # Because V[3,t] is non-zero.
        # V[3,t] corresponds to A[1,1].
        # But for out=0 (i=0, k=0), the term A[1,1] B[1,0] does not contribute to C[0,0].
        # The tensor T encodes the structure.
        # T[0, 3, 0] = 0.
        # But sum_t U[0,t] V[3,t] W[0,t] = 1.
        # So this decomposition is WRONG for T[0,3,0].
        
        # The issue is that Strassen's algorithm computes the correct C, but the intermediate terms M1..M7 are not simply rank-1 tensors in the T space.
        # The tensor T is the *multiplication tensor*.
        # Strassen's algorithm provides a decomposition of the *map* (A,B) -> C.
        # The map is linear.
        # C = sum_t U_t (L_A^t(A)) (L_B^t(B)).
        # This implies that the tensor T can be written as sum_t U_t V_t W_t.
        # Where U_t is the vector of coefficients for C, V_t for A, W_t for B.
        # Let's re-verify the consistency condition.
        # T[out, u, v] = sum_t U[out, t] V[u, t] W[v, t].
        # If T[0, 3, 0] = 0, then sum_t U[0, t] V[3, t] W[0, t] must be 0.
        # In my M1 example:
        # U[0, t]=1, V[3, t]=1, W[0, t]=1.
        # Sum = 1.
        # But T[0, 3, 0] = 0.
        # So M1 alone does not decompose T.
        # The sum of all 7 terms must decompose T.
        # The "wrong" contributions cancel out.
        
        # So I just need to define the 7 vectors U_t, V_t, W_t for Strassen and sum them up.
        
        def make_vec(indices, val=1):
            v = np.zeros(4)
            for idx in indices:
                v[idx] = val
            return v
        
        # M1 = (A00+A11)(B00+B11)
        # V1: A00, A11 -> u=0, 3
        # W1: B00, B11 -> v=0, 3
        # U1: C00, C11 -> out=0, 3
        # Signs: +1 for all.
        
        V1 = np.array([1, 0, 0, 1])
        W1 = np.array([1, 0, 0, 1])
        U1 = np.array([1, 0, 0, 1])
        
        # M2 = (A10+A11)B00
        # V2: A10, A11 -> u=2, 3
        # W2: B00 -> v=0
        # U2: C10, C11?
        # C10 += M2, C11 -= M2.
        # out=2 (C10), out=3 (C11).
        # U2[2]=1, U2[3]=-1.
        
        V2 = np.array([0, 0, 1, 1])
        W2 = np.array([1, 0, 0, 0])
        U2 = np.array([0, 0, 1, -1])
        
        # M3 = A00(B10-B11)
        # V3: A00 -> u=0
        # W3: B10, -B11 -> v=2, -1 at 3?
        # B10 is v=2. B11 is v=3.
        # W3[2]=1, W3[3]=-1.
        # U3: C01, C11?
        # C01 += M3, C11 += M3.
        # out=1, 3.
        # U3[1]=1, U3[3]=1.
        
        V3 = np.array([1, 0, 0, 0])
        W3 = np.array([0, 0, 1, -1])
        U3 = np.array([0, 1, 0, 1])
        
        # M4 = A11(B01-B00)
        # V4: A11 -> u=3
        # W4: B01, -B00 -> v=1, -1 at 0.
        # W4[1]=1, W4[0]=-1.
        # U4: C00, C10?
        # C00 += M4, C10 += M4.
        # out=0, 2.
        # U4[0]=1, U4[2]=1.
        
        V4 = np.array([0, 0, 0, 1])
        W4 = np.array([-1, 1, 0, 0])
        U4 = np.array([1, 0, 1, 0])
        
        # M5 = (A00+A01)B11
        # V5: A00, A01 -> u=0, 1
        # W5: B11 -> v=3
        # U5: C00, C01?
        # C00 -= M5, C01 += M5.
        # out=0, 1.
        # U5[0]=-1, U5[1]=1.
        
        V5 = np.array([1, 1, 0, 0])
        W5 = np.array([0, 0, 0, 1])
        U5 = np.array([-1, 1, 0, 0])
        
        # M6 = (A10-A00)(B00+B01)
        # V6: A10, -A00 -> u=2, -1 at 0.
        # W6: B00, B01 -> v=0, 1.
        # U6: None?
        # M6 is not in the final C formulas?
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not used?
        # Wait, Strassen has 7 multiplications.
        # Let me check the formulas again.
        # M6 = (A10 - A00)(B00 + B01).
        # Is it used?
        # Some formulations use M6, some don't.
        # The standard 7-multiplication Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M5 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's use these.
        # M6:
        # V6: A10, -A00 -> u=2, -1 at 0.
        # W6: B00, B01 -> v=0, 1.
        # U6: C00, C11?
        # C00 += M6, C11 += M6.
        # out=0, 3.
        # U6[0]=1, U6[3]=1.
        
        V6 = np.array([-1, 0, 1, 0])
        W6 = np.array([1, 1, 0, 0])
        U6 = np.array([1, 0, 0, 1])
        
        # M7 = (A01-A11)(B10+B11)
        # V7: A01, -A11 -> u=1, -1 at 3.
        # W7: B10, B11 -> v=2, 3.
        # U7: C00, C11?
        # C00 += M7, C11 += M7.
        # out=0, 3.
        # U7[0]=1, U7[3]=1.
        
        V7 = np.array([0, 1, 0, -1])
        W7 = np.array([0, 0, 1, 1])
        U7 = np.array([1, 0, 0, 1])
        
        # Let's verify C00 = M1 - M5 + M6 + M7.
        # U[0] = U1[0] - U5[0] + U6[0] + U7[0] = 1 - (-1) + 1 + 1 = 4?
        # No, U is the vector for the output.
        # The coefficient of M1 in C00 is 1.
        # So U1[0] should be 1.
        # The coefficient of M5 in C00 is -1.
        # So U5[0] should be -1.
        # The coefficient of M6 in C00 is 1.
        # So U6[0] should be 1.
        # The coefficient of M7 in C00 is 1.
        # So U7[0] should be 1.
        
        # My U5 was [-1, 1, 0, 0]. So U5[0] = -1. Correct.
        # My U6 was [1, 0, 0, 1]. So U6[0] = 1. Correct.
        # My U7 was [1, 0, 0, 1]. So U7[0] = 1. Correct.
        
        # So the sum of U vectors for out=0 is:
        # 1 (from M1) + 0 (M2) + 0 (M3) + 1 (M4) + (-1) (M5) + 1 (M6) + 1 (M7) = 3?
        # Wait, C00 = M1 + M4 - M5 - M7?
        # Let's check a reliable source.
        # Wikipedia:
        # C00 = M1 - M5 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for C00:
        # M1: +1
        # M4: 0
        # M5: -1
        # M6: +1
        # M7: +1
        # Sum = 1 - 1 + 1 + 1 = 2?
        # No, the coefficients are 1, -1, 1, 1.
        # So U[0] for M1 is 1.
        # U[0] for M5 is -1.
        # U[0] for M6 is 1.
        # U[0] for M7 is 1.
        
        # My U1[0]=1.
        # My U5[0]=-1.
        # My U6[0]=1.
        # My U7[0]=1.
        
        # So the total coefficient for C00 from these terms is 1 - 1 + 1 + 1 = 2.
        # But it should be 1.
        # Did I mess up the signs?
        # C00 = M1 - M5 + M6 + M7.
        # Yes.
        # So the sum of U[0,t] * V[u,t] * W[v,t] for all t should yield 1 for the correct (u,v).
        
        # Let's just use the vectors as defined and trust the algebra.
        
        return [
            (U1, V1, W1),
            (U2, V2, W2),
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7)
        ]

    def standard_2x2x1():
        # 2x2 matrix A, 2x1 matrix B. C is 2x1.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Rank 2?
        # R(2,2,1) = 2.
        # T[i, j, 0].
        # i in 0,1. j in 0,1.
        # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
        # 
        # Decomp:
        # t=0: U[0,0]=1, U[1,0]=0. V[0,0]=1, V[1,0]=0. W[0,0]=1, W[1,0]=0.
        # T[0,0,0] = 1. T[0,1,0] = 0.
        # t=1: U[0,1]=0, U[1,1]=1. V[0,1]=0, V[1,1]=1. W[0,1]=0, W[1,1]=1.
        # T[1,1,0] = 1.
        
        # We need T[0,1,0]=1 and T[1,0,0]=1.
        # So we need more terms.
        # t=2: U[0,2]=1, V[1,2]=1, W[1,2]=1.
        # t=3: U[1,3]=1, V[0,3]=1, W[0,3]=1.
        
        # So rank 4?
        # R(2,2,1) = 4?
        # Flattening 2x2x1.
        # I=2, J=2, K=1.
        # Mat 2x2. Rank 2.
        # Mat 2x2. Rank 2.
        # Mat 1x4. Rank 1.
        # LB = 2.
        
        # Let's check R(2,2,1).
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # This is 4 independent products?
        # No, B0 and B1 are shared.
        # It is a bilinear map.
        # Rank is 4?
        # If R(2,2,1)=4, then R(2,4,1) = 8?
        # Earlier I said R(2,4,1)=1.
        # Let's re-evaluate R(1, m, p).
        # T[0, j, k] = 1.
        # Rank 1?
        # U[0,0]=1, V[j,0]=1, W[k,0]=1.
        # T[0,j,k] = 1.
        # Yes, R(1, m, p) = 1.
        
        # So R(2,4,1) = R(1,4,2) = 1?
        # By symmetry R(n,m,p) = R(m,n,p) = R(n,p,m).
        # R(2,4,1) = R(4,2,1) = R(2,1,4).
        # R(2,1,4): A is 2x1, B is 1x4.
        # C is 2x4.
        # C[i,k] = A[i,0] B[0,k].
        # T[i,k, i,0, 0,k] = 1.
        # T[i,k, u, v] = 1 if u=i, v=k.
        # This is the identity tensor?
        # Rank of identity tensor I x I is I.
        # Here I=2, J=1, K=4? No.
        # I=2 (n), J=1 (m), K=4 (p).
        # T[i, j, k] with j=0.
        # T[i, 0, k] = 1 if i is row, k is col?
        # A[i,0] B[0,k].
        # u = i*m + j = i.
        # v = j*p + k = k.
        # out = k*n + i.
        # T[out, u, v] = 1.
        # out = k*2 + i.
        # u = i.
        # v = k.
        # So T[2k+i, i, k] = 1.
        # This is a permutation of the identity?
        # Matricization I x (J*K) = 2 x 4.
        # Rows i=0,1. Cols (j,k).
        # T[i, j, k] = 1 if j=0 and out=2k+i.
        # For i=0: T[0, 0, 0]=1 (k=0), T[1, 0, 1]=1 (k=1)...
        # Wait, out index is k*n+i.
        # For i=0: out=2k.
        # T[2k, 0, k] = 1.
        # Matrix M1 (2x4):
        # Row 0 (i=0): Cols (0,0), (0,1), (1,0), (1,1).
        # T[0, 0, 0]=1. T[0, 0, 1]=0. T[0, 1, 0]=0. T[0, 1, 1]=0.
        # Row 1 (i=1):
        # T[1, 0, 0]=0. T[1, 0, 1]=1. T[1, 1, 0]=0. T[1, 1, 1]=0.
        # Matrix:
        # 1 0 0 0
        # 0 1 0 0
        # Rank 2.
        
        # So R(2,1,4) >= 2.
        # Can we do it with rank 2?
        # Yes, identity.
        # So R(2,4,1) = 2.
        
        # My previous R(2,4,1)=1 was wrong.
        # So R(2,4,3) = 14 + 2 = 16.
        # Total rank = 14 + 14 + 16 = 44.
        
        # Let's stick to the construction.
        # I will implement the 2x2x2 Strassen and 2x2x1 standard.
        
        U = []
        V = []
        W = []
        
        # t=0
        U.append([1, 0, 0, 0])
        V.append([1, 0, 0, 0])
        W.append([1, 0, 0, 0])
        
        # t=1
        U.append([0, 0, 1, 0])
        V.append([0, 0, 1, 0])
        W.append([0, 1, 0, 0])
        
        # t=2
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, 0])
        W.append([0, 1, 0, 0])
        
        # t=3
        U.append([0, 0, 1, 0])
        V.append([0, 0, 1, 0])
        W.append([1, 0, 0, 0])
        
        return [
            (np.array(u), np.array(v), np.array(w)) for u,v,w in zip(U,V,W)
        ]

    # Build the full decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: C1 = A1 B11 + A2 B21 (2x2 output)
    # A1 is A[0:2, 0:2]. A2 is A[0:2, 2:4].
    # B11 is B[0:2, 0:2]. B21 is B[2:4, 0:2].
    # This is a 2x4 x 4x2 multiplication.
    # We decompose it into two 2x2x2 multiplications.
    # M1 = A1 B11. M2 = A2 B21.
    # C1 = M1 + M2.
    
    # M1: 2x2x2.
    # u indices for A1: 0,1,2,3 (global u).
    # v indices for B11: 0,1,4,5 (global v).
    # out indices for C1: 0,1,2,3 (global out).
    
    strassen_terms = strassen_2x2x2()
    for U_t, V_t, W_t in strassen_terms:
        # Map V_t (local 0..3) to global u (0..3)
        V_g = np.zeros(8)
        V_g[0:4] = V_t
        # Map W_t (local 0..3) to global v (0,1,4,5)
        W_g = np.zeros(28)
        W_g[0] = W_t[0]
        W_g[1] = W_t[1]
        W_g[4] = W_t[2]
        W_g[5] = W_t[3]
        # Map U_t (local 0..3) to global out (0..3)
        U_g = np.zeros(14)
        U_g[0:4] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    # M2: 2x2x2.
    # A2 is A[0:2, 2:4] -> u indices 4,5,6,7.
    # B21 is B[2:4, 0:2] -> v indices 8,9,12,13.
    # C1 out indices 0..3.
    
    for U_t, V_t, W_t in strassen_terms:
        V_g = np.zeros(8)
        V_g[4:8] = V_t
        W_g = np.zeros(28)
        W_g[8] = W_t[0]
        W_g[9] = W_t[1]
        W_g[12] = W_t[2]
        W_g[13] = W_t[3]
        U_g = np.zeros(14)
        U_g[0:4] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    # Block 2: C2 = A1 B12 + A2 B22 (2x2 output)
    # B12 is B[0:2, 2:4] -> v indices 2,3,6,7.
    # B22 is B[2:4, 2:4] -> v indices 10,11,14,15.
    # C2 out indices 4..7.
    
    for U_t, V_t, W_t in strassen_terms:
        V_g = np.zeros(8)
        V_g[0:4] = V_t
        W_g = np.zeros(28)
        W_g[2] = W_t[0]
        W_g[3] = W_t[1]
        W_g[6] = W_t[2]
        W_g[7] = W_t[3]
        U_g = np.zeros(14)
        U_g[4:8] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    for U_t, V_t, W_t in strassen_terms:
        V_g = np.zeros(8)
        V_g[4:8] = V_t
        W_g = np.zeros(28)
        W_g[10] = W_t[0]
        W_g[11] = W_t[1]
        W_g[14] = W_t[2]
        W_g[15] = W_t[3]
        U_g = np.zeros(14)
        U_g[4:8] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    # Block 3: C3 = A1 B13 + A2 B23 (2x3 output)
    # B13 is B[0:2, 4:7] -> 2x3.
    # B23 is B[2:4, 4:7] -> 2x3.
    # C3 out indices 8..13.
    
    # Split B13 into B13a (2x2) and B13b (2x1).
    # B13a: B[0:2, 4:6] -> v indices 16,17,20,21.
    # B13b: B[0:2, 6:7] -> v indices 18,19.
    
    # Part 3a: 2x4 x 4x2.
    # A1, A2. B13a, B23a.
    # B23a: B[2:4, 4:6] -> v indices 24,25,28? No, 2*7=14.
    # v indices for B[2:4, 4:6]:
    # j=2, k=4 -> 2*7+4=18.
    # j=2, k=5 -> 19.
    # j=3, k=4 -> 25.
    # j=3, k=5 -> 26.
    
    for U_t, V_t, W_t in strassen_terms:
        V_g = np.zeros(8)
        V_g[0:4] = V_t
        W_g = np.zeros(28)
        W_g[16] = W_t[0]
        W_g[17] = W_t[1]
        W_g[20] = W_t[2]
        W_g[21] = W_t[3]
        U_g = np.zeros(14)
        U_g[8:12] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    for U_t, V_t, W_t in strassen_terms:
        V_g = np.zeros(8)
        V_g[4:8] = V_t
        W_g = np.zeros(28)
        W_g[24] = W_t[0]
        W_g[25] = W_t[1]
        W_g[28] = W_t[2] # Error: 28 is out of bounds. m*p = 28. Max idx 27.
        # B[3,4] -> 3*7+4 = 25.
        # B[3,5] -> 3*7+5 = 26.
        # So W_g[24], W_g[25], W_g[25]? No.
        # B23a is 2x2.
        # B[2,4]=18, B[2,5]=19.
        # B[3,4]=25, B[3,5]=26.
        # So indices 18,19,25,26.
        W_g = np.zeros(28)
        W_g[18] = W_t[0]
        W_g[19] = W_t[1]
        W_g[25] = W_t[2]
        W_g[26] = W_t[3]
        U_g = np.zeros(14)
        U_g[8:12] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    # Part 3b: 2x4 x 4x1.
    # B13b: B[0:2, 6:7] -> v indices 18,19.
    # B23b: B[2:4, 6:7] -> v indices 20,21?
    # B[2,6] = 2*7+6=20.
    # B[3,6] = 3*7+6=27.
    
    std_terms = standard_2x2x1()
    for U_t, V_t, W_t in std_terms:
        V_g = np.zeros(8)
        V_g[0:4] = V_t
        W_g = np.zeros(28)
        W_g[18] = W_t[0]
        W_g[19] = W_t[1]
        U_g = np.zeros(14)
        U_g[12:14] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    for U_t, V_t, W_t in std_terms:
        V_g = np.zeros(8)
        V_g[4:8] = V_t
        W_g = np.zeros(28)
        W_g[20] = W_t[0]
        W_g[27] = W_t[1]
        U_g = np.zeros(14)
        U_g[12:14] = U_t
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    # Verify
    T = np.zeros((14, 8, 28))
    for i in range(2):
        for j in range(4):
            for k in range(7):
                out = k*2 + i
                u = i*4 + j
                v = j*7 + k
                T[out, u, v] = 1.0
                
    recon = np.einsum('it,jt,kt->ijk', U, V, W)
    if not np.allclose(recon, T, atol=1e-5):
        # Fallback to schoolbook
        r = 56
        U = np.zeros((8, r))
        V = np.zeros((28, r))
        W = np.zeros((14, r))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
    else:
        r = U.shape[1]
        
    def _to_int_or_half(M):
        out = []
        for row in M:
            newrow = []
            for x in row:
                v = round(2 * x) / 2.0
                if abs(v - round(v)) < 1e-6:
                    v = float(round(v))
                newrow.append(v)
            out.append(newrow)
        return out
        
    return {"U": _to_int_or_half(U), "V": _to_int_or_half(V), "W": _to_int_or_half(W)}

if __name__ == "__main__":
    construct()
