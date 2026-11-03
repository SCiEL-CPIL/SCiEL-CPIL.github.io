"""Rewritten program: block-partitioned Strassen for 2x4 * 4x8."""
import time

DIMENSIONS = [2, 4, 8]

# Deadline guard: kill after 150s wall-clock, return best with safety margin.
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def construct():
    """
    Represent the 2x4 * 4x8 multiplication as a batch of four independent
    2x2 * 2x2 matrix multiplications by partitioning the output into 2x2 blocks
    along the 8-column dimension, and apply the rank-7 Strassen decomposition
    to each block independently, yielding a total rank of 28.
    
    Structure:
      A is 2x4, B is 4x8, C = A*B is 2x8.
      Partition B into 4 blocks of 4x2: B = [B0, B1, B2, B3] where Bt is 4x2.
      Partition A into 2 blocks of 2x2: A = [A0, A1] where Ai is 2x2.
      C = [A0*B0, A0*B1, A0*B2, A0*B3] ... wait, that's not quite right.
      
      Actually: A is 2x4, B is 4x8.
      C[i,k] = sum_j A[i,j]*B[j,k]
      
      Let's partition differently. Split the 4 middle dimension into two groups of 2:
      A = [A_L | A_R] where A_L is 2x2, A_R is 2x2.
      B = [B_T; B_B] where B_T is 2x8, B_B is 2x8.
      C = A_L * B_T + A_R * B_B
      
      Each of A_L*B_T is 2x2 * 2x8. We can further partition B_T into four 2x2 blocks:
      B_T = [B_T0, B_T1, B_T2, B_T3] each 2x2.
      Similarly B_B = [B_B0, B_B1, B_B2, B_B3] each 2x2.
      
      So C = A_L * B_T + A_R * B_B
         = [A_L*B_T0 + A_R*B_B0, A_L*B_T1 + A_R*B_B1, A_L*B_T2 + A_R*B_B2, A_L*B_T3 + A_R*B_B3]
      
      Each output 2x2 block is a sum of two 2x2*2x2 products. This doesn't directly give independent 2x2*2x2 multiplications.
      
      Let me reconsider the directive: "partitioning the output into 2x2 blocks along the 8-column dimension"
      
      Output C is 2x8. Partition into four 2x2 blocks: C = [C0, C1, C2, C3] where Ct is 2x2.
      Ct = A * Bt where Bt is the t-th 4x2 block of B.
      
      So we have 4 independent problems: compute 2x2 = (2x4) * (4x2).
      
      For each such sub-problem: A_sub is 2x4, B_sub is 4x2, C_sub is 2x2.
      We can further decompose A_sub = [A_sub_L | A_sub_R] (each 2x2) and B_sub = [B_sub_T; B_sub_B] (each 2x2).
      C_sub = A_sub_L * B_sub_T + A_sub_R * B_sub_B.
      
      This is a sum of two 2x2*2x2 products. Using Strassen on each 2x2*2x2 gives 7+7=14 per sub-problem, total 4*14=56.
      
      But the directive says "apply the rank-7 Strassen decomposition to each block independently, yielding a total rank of 28."
      
      Let me re-read: "Represent the 2×4·4×8 multiplication as a batch of four independent 2×2·2×2 matrix multiplications"
      
      So each of the 4 output 2x2 blocks should be computed as a single 2x2*2x2 product. That means:
      Ct = M_t * N_t where M_t is 2x2 and N_t is 2x2.
      
      But Ct = A * Bt where A is 2x4 and Bt is 4x2. This is NOT a 2x2*2x2 product in general.
      
      Unless... we restructure. Let me think about this differently.
      
      Actually, I think the idea is:
      C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
      
      Group j into pairs: j = 2*q + r where q in {0,1}, r in {0,1}.
      C[i,k] = sum_q sum_r A[i, 2q+r] * B[2q+r, k]
      
      For fixed q, define A_q[i, r] = A[i, 2q+r] (2x2) and B_q[r, k] = B[2q+r, k] (2x8).
      C[i,k] = sum_q A_q * B_q
      
      Now B_q is 2x8, which we can split into four 2x2 blocks: B_q = [B_q0, B_q1, B_q2, B_q3].
      C = [sum_q A_q*B_q0, sum_q A_q*B_q1, sum_q A_q*B_q2, sum_q A_q*B_q3]
      
      Each column block C_t = sum_q A_q * B_qt, which is a sum of two 2x2*2x2 products.
      
      To get rank 28, we need each C_t to be computed with rank 7, not 14.
      
      Hmm, this is tricky. Let me just implement the 56-rank solution first (4 sub-problems, each with 2 Strassen decompositions), and see if we can do better.
      
      Actually, let me just implement rank 28 as stated in the directive. Perhaps there's a clever way.
      
      Wait - maybe the approach is different. Let me think of it as:
      We have 4 independent 2x2*2x2 multiplications. Each uses Strassen (rank 7). Total rank = 4*7 = 28.
      
      For this to work, each output 2x2 block must be exactly one 2x2*2x2 product.
      
      C_t = M_t * N_t where M_t, N_t are 2x2 matrices that are linear combinations of entries of A and B.
      
      This is actually possible if we use the fact that the tensor for 2x4*4x8 can be viewed as 4 copies of a smaller tensor. Let me just implement it and verify.
      
      Let me define:
      For t = 0,1,2,3 (column block index):
        B_t is the 4x2 submatrix of B with columns 2t, 2t+1.
        B_t = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
        
        A is 2x4: [[a00, a01, a02, a03], [a10, a11, a12, a13]]
        
        C_t = A * B_t (2x2 result)
        
        C_t[i,j] = sum_{r=0}^{3} A[i,r] * B_t[r,j]
        
      To express C_t as M_t * N_t (2x2 * 2x2), we need to find 2x2 matrices M_t, N_t (linear in A, B entries) such that M_t * N_t = C_t.
      
      This is generally not possible with just 2x2 matrices unless we allow M_t and N_t to depend on all entries. But a 2x2*2x2 product has only 4 "independent" bilinear terms (well, 8 products of entries), while C_t has 8 bilinear terms (2*2*4). So it's not directly a 2x2*2x2 product.
      
      I think the directive might be slightly imprecise, but the key insight from the cell findings is: "Partitioning the 2x4*4x8 multiplication into 8 independent 2x2*2x2 sub-problems (2 j-groups × 4 k-groups) and applying rank-7 Strassen to each yields total rank 56."
      
      So rank 56 was achieved. The directive says 28, but let me go with 56 first since that's measured, and try to see if 28 is feasible.
      
      Actually, let me just implement the rank-56 solution (8 sub-problems, each 2x2*2x2 with Strassen rank 7).
    """
    if _time_left() < 5:
        # Fallback to schoolbook if no time
        return _schoolbook()
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Partition into 8 sub-problems: 2 j-groups (each of size 2) x 4 k-groups (each of size 2)
    # For each (jg, kg) pair, we have a 2x2 * 2x2 sub-multiplication.
    # A_sub is 2x2: rows i=0,1; cols j=2*jg, 2*jg+1
    # B_sub is 2x2: rows j=2*jg, 2*jg+1; cols k=2*kg, 2*kg+1
    # C_sub is 2x2: rows i=0,1; cols k=2*kg, 2*kg+1
    # C[i, 2*kg+r] += sum over j-groups of A[i, 2*jg+s] * B[2*jg+s, 2*kg+r]
    
    # Each sub-problem contributes to a 2x2 block of C.
    # There are 2 j-groups and 4 k-groups, so 8 sub-problems.
    # Each sub-problem is 2x2 * 2x2 -> 2x2, using Strassen (rank 7).
    # Total rank = 8 * 7 = 56.
    
    # Strassen for 2x2 * 2x2:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    # z00 = M1+M4-M5+M7
    # z01 = M3+M5
    # z10 = M2+M4
    # z11 = M1-M2+M3+M6
    
    # For each sub-problem (jg, kg), we need to express the 7 Strassen products
    # in terms of the global A and B entries.
    
    # A[i, 2*jg+s] for i in {0,1}, s in {0,1}
    # B[2*jg+s, 2*kg+r] for s in {0,1}, r in {0,1}
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for jg in range(2):  # j-group: 0, 1
        for kg in range(4):  # k-group: 0, 1, 2, 3
            # X = A_sub: X[i, s] = A[i, 2*jg + s]
            # Y = B_sub: Y[s, r] = B[2*jg + s, 2*kg + r]
            
            # Strassen intermediates:
            # M1 = (x00+x11)*(y00+y11)
            # M2 = (x10+x11)*y00
            # M3 = x00*(y01-y11)
            # M4 = x11*(y10-y00)
            # M5 = (x00+x01)*y11
            # M6 = (x10-x00)*(y00+y10)
            # M7 = (x01-x11)*(y01+y11)
            
            # For each Strassen product, we need to express it as a bilinear form.
            # Each M_t is a product of a linear combo of A entries and a linear combo of B entries.
            # So each M_t corresponds to ONE rank-1 term in the tensor decomposition.
            
            # M1: (A[0,2jg] + A[1,2jg+1]) * (B[2jg,2kg] + B[2jg+1,2kg+1])
            # U part: A[0,2jg] has coeff 1, A[1,2jg+1] has coeff 1
            # V part: B[2jg,2kg] has coeff 1, B[2jg+1,2kg+1] has coeff 1
            
            # z00 = M1+M4-M5+M7
            # z01 = M3+M5
            # z10 = M2+M4
            # z11 = M1-M2+M3+M6
            
            # W coefficients for each output element:
            # C[0, 2kg] = z00 = M1+M4-M5+M7
            # C[1, 2kg] = z10 = M2+M4
            # C[0, 2kg+1] = z01 = M3+M5
            # C[1, 2kg+1] = z11 = M1-M2+M3+M6
            
            # W index: W[k*n+i, c] where k is column index, i is row index.
            # So C[i, k] corresponds to W[k*2+i, c].
            
            # M1: contributes to C[0,2kg] (+1), C[1,2kg+1] (+1)
            # M2: contributes to C[1,2kg] (+1), C[1,2kg+1] (-1)
            # M3: contributes to C[0,2kg+1] (+1), C[1,2kg+1] (+1)
            # M4: contributes to C[0,2kg] (+1), C[1,2kg] (+1)
            # M5: contributes to C[0,2kg] (-1), C[0,2kg+1] (+1)
            # M6: contributes to C[1,2kg+1] (+1)
            # M7: contributes to C[0,2kg] (+1)
            
            # Define the 7 Strassen terms:
            # Each term is (U_coeff_vector, V_coeff_vector, W_coeff_vector)
            
            # U indices: i*m + j = i*4 + j
            # V indices: j*p + k = j*8 + k
            # W indices: k*n + i = k*2 + i
            
            strassen_terms = []
            
            # M1 = (x00+x11)*(y00+y11)
            # x00 = A[0, 2jg], x11 = A[1, 2jg+1]
            # y00 = B[2jg, 2kg], y11 = B[2jg+1, 2kg+1]
            U_m1 = {0*4 + 2*jg: 1, 1*4 + (2*jg+1): 1}
            V_m1 = {2*jg*8 + 2*kg: 1, (2*jg+1)*8 + (2*kg+1): 1}
            W_m1 = {(2*kg)*2 + 0: 1, (2*kg+1)*2 + 1: 1}  # C[0,2kg], C[1,2kg+1]
            strassen_terms.append((U_m1, V_m1, W_m1))
            
            # M2 = (x10+x11)*y00
            # x10 = A[1, 2jg], x11 = A[1, 2jg+1]
            # y00 = B[2jg, 2kg]
            U_m2 = {1*4 + 2*jg: 1, 1*4 + (2*jg+1): 1}
            V_m2 = {2*jg*8 + 2*kg: 1}
            W_m2 = {(2*kg)*2 + 1: 1, (2*kg+1)*2 + 1: -1}  # C[1,2kg], C[1,2kg+1]
            strassen_terms.append((U_m2, V_m2, W_m2))
            
            # M3 = x00*(y01-y11)
            # x00 = A[0, 2jg]
            # y01 = B[2jg, 2kg+1], y11 = B[2jg+1, 2kg+1]
            U_m3 = {0*4 + 2*jg: 1}
            V_m3 = {2*jg*8 + (2*kg+1): 1, (2*jg+1)*8 + (2*kg+1): -1}
            W_m3 = {(2*kg+1)*2 + 0: 1, (2*kg+1)*2 + 1: 1}  # C[0,2kg+1], C[1,2kg+1]
            strassen_terms.append((U_m3, V_m3, W_m3))
            
            # M4 = x11*(y10-y00)
            # x11 = A[1, 2jg+1]
            # y10 = B[2jg+1, 2kg], y00 = B[2jg, 2kg]
            U_m4 = {1*4 + (2*jg+1): 1}
            V_m4 = {(2*jg+1)*8 + 2*kg: 1, 2*jg*8 + 2*kg: -1}
            W_m4 = {(2*kg)*2 + 0: 1, (2*kg)*2 + 1: 1}  # C[0,2kg], C[1,2kg]
            strassen_terms.append((U_m4, V_m4, W_m4))
            
            # M5 = (x00+x01)*y11
            # x00 = A[0, 2jg], x01 = A[0, 2jg+1]
            # y11 = B[2jg+1, 2kg+1]
            U_m5 = {0*4 + 2*jg: 1, 0*4 + (2*jg+1): 1}
            V_m5 = {(2*jg+1)*8 + (2*kg+1): 1}
            W_m5 = {(2*kg)*2 + 0: -1, (2*kg+1)*2 + 0: 1}  # C[0,2kg], C[0,2kg+1]
            strassen_terms.append((U_m5, V_m5, W_m5))
            
            # M6 = (x10-x00)*(y00+y10)
            # x10 = A[1, 2jg], x00 = A[0, 2jg]
            # y00 = B[2jg, 2kg], y10 = B[2jg+1, 2kg]
            U_m6 = {1*4 + 2*jg: 1, 0*4 + 2*jg: -1}
            V_m6 = {2*jg*8 + 2*kg: 1, (2*jg+1)*8 + 2*kg: 1}
            W_m6 = {(2*kg+1)*2 + 1: 1}  # C[1,2kg+1]
            strassen_terms.append((U_m6, V_m6, W_m6))
            
            # M7 = (x01-x11)*(y01+y11)
            # x01 = A[0, 2jg+1], x11 = A[1, 2jg+1]
            # y01 = B[2jg, 2kg+1], y11 = B[2jg+1, 2kg+1]
            U_m7 = {0*4 + (2*jg+1): 1, 1*4 + (2*jg+1): -1}
            V_m7 = {2*jg*8 + (2*kg+1): 1, (2*jg+1)*8 + (2*kg+1): 1}
            W_m7 = {(2*kg)*2 + 0: 1}  # C[0,2kg]
            strassen_terms.append((U_m7, V_m7, W_m7))
            
            for U_dict, V_dict, W_dict in strassen_terms:
                for idx, coeff in U_dict.items():
                    U[idx][col] = coeff
                for idx, coeff in V_dict.items():
                    V[idx][col] = coeff
                for idx, coeff in W_dict.items():
                    W[idx][col] = coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
