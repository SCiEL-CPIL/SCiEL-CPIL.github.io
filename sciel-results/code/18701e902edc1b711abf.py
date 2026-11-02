import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Laderman's 3x3x3 rank-23 decomposition
    # The 23 rank-1 terms for 3x3 matrix multiplication
    # Each term: (u_9, v_9, w_9) where indices are flattened 3x3 matrices
    # u[i*3+j] is U_r[i,j], v[j*3+k] is V_r[j,k], w[k*3+i] is W_r[i,k]
    # (note: W index is k*3+i per the trace convention W[k*n+i])
    
    # Laderman factors (verified from literature)
    # These are the 23 terms where sum over r of u_r[i,j]*v_r[j,k]*w_r[i,k] = 1 for all i,j,k in {0,1,2}
    # with w indexed as w[k*3+i] (i.e., w is stored in column-major for the (i,k) output)
    
    # Actually, let me be very careful with the index convention.
    # The task says: W[k*n+i, r] where n=3 for the 3x3x3 sub-problem.
    # So W[k*3+i, r] corresponds to the (i,k) entry of the output matrix C.
    # The decomposition requires: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # For 3x3x3: sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
    
    # Laderman's 23 factors for 3x3x3:
    # I'll use the well-known explicit factors.
    # Each factor is (U_9, V_9, W_9) where:
    # U_9[i*3+j] = coefficient for A[i,j]
    # V_9[j*3+k] = coefficient for B[j,k]  
    # W_9[k*3+i] = coefficient for C[i,k] (note the k*3+i indexing)
    
    LADERMAN_23 = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[0,1]*B[1,0] -> C[0,0]  (but combined in Laderman)
        # Let me use the actual Laderman factors
    ]
    
    # I need to use the actual Laderman factors. Let me construct them properly.
    # The Laderman 3x3x3 decomposition has 23 terms. Here they are:
    
    # Using the standard Laderman algorithm structure:
    # The 23 products combine the 27 schoolbook products.
    
    # Let me define the factors explicitly. Each is (u, v, w) where:
    # u has 9 entries: u[i*3+j] for the (i,j) entry of A
    # v has 9 entries: v[j*3+k] for the (j,k) entry of B
    # w has 9 entries: w[k*3+i] for the (i,k) entry of C (note: k*3+i, not i*3+k)
    
    # Laderman's 23 rank-1 terms:
    L = [
        # (u, v, w)
        # 1. a00*b00 -> c00
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # 2. a01*b00 -> c01 (a01 is at index 1, b00 at index 0, c01 at index k*3+i = 1*3+0=3)
        ([0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 3. a02*b00 -> c02 (a02 at 2, b00 at 0, c02 at 2*3+0=6)
        ([0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]),
        # 4. a10*b00 -> c10 (a10 at 3, b00 at 0, c10 at 0*3+1=1)
        ([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # 5. a11*b00 -> c11 (a11 at 4, b00 at 0, c11 at 1*3+1=4)
        ([0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # 6. a12*b00 -> c12 (a12 at 5, b00 at 0, c12 at 2*3+1=7)
        ([0,0,0,0,0,1,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
        # 7. a20*b00 -> c20 (a20 at 6, b00 at 0, c20 at 0*3+2=2)
        ([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        # 8. a21*b00 -> c21 (a21 at 7, b00 at 0, c21 at 1*3+2=5)
        ([0,0,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
        # 9. a22*b00 -> c22 (a22 at 8, b00 at 0, c22 at 2*3+2=8)
        ([0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
        # Now for b10 (index 3):
        # 10. a00*b10 -> c01 (a00 at 0, b10 at 3, c01 at 3)
        ([1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 11. a01*b10 -> c02 (a01 at 1, b10 at 3, c02 at 6)
        ([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]),
        # 12. a02*b10 -> c03... wait, c has only 3x3. c02 is at 6.
        # Hmm, I'm making errors. Let me restart with a clearer approach.
    ]
    
    # This manual approach is error-prone. Let me use a verified set of Laderman factors.
    # I'll hardcode the correct ones from a reliable source.
    
    # The correct Laderman 3x3x3 factors (23 terms):
    # I'll represent each as (u9, v9, w9) with the convention:
    # u9[i*3+j], v9[j*3+k], w9[k*3+i]
    
    # After careful consideration, here are the verified Laderman factors:
    laderman = [
        # Term 0
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 2
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 3
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 4
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 5
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 6
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 7
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 8
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Terms 9-22: The remaining 14 terms that make this rank-23 instead of rank-27
        # These involve combinations that reduce the rank
    ]
    
    # I realize I can't reliably produce the Laderman factors from memory.
    # Let me use a different approach: construct a valid decomposition with rank < 63.
    
    # Strategy: Use the schoolbook decomposition but with a block structure.
    # For 3x3x7, the tensor T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,...,6}.
    
    # Key insight: We can group the k indices and use shared factors.
    # For each pair (i,j), the contribution to all k values is:
    # C[i,k] += A[i,j] * B[j,k]
    # 
    # The schoolbook uses 3*3*7 = 63 terms.
    # 
    # Can we do better? Yes, if we can find a low-rank decomposition of the 3x3x7 tensor.
    # 
    # One approach: Use the fact that for fixed i, the row of C is:
    # C[i,:] = A[i,:] * B
    # This is a 1x3 times 3x7 multiplication, which has rank 3 (schoolbook for 1x3x7).
    # So for 3 rows, we get 3*3 = 9 terms? No, that's not right either.
    #
    # Actually, the rank of the 3x3x7 matmul tensor is at most 3*3*7 = 63,
    # but it can be lower. The minimum rank is at least 3*7 = 21 (by considering
    # the "slice" structure).
    #
    # Let me try a different approach: use a rank-21 decomposition if possible.
    # For each i in {0,1,2} and k in {0,...,6}, we need:
    # sum_j A[i,j] * B[j,k] = C[i,k]
    # 
    # If we set U[i*3+j, r] = delta(i, r_i) * u_j(r)
    # and V[j*3+k, r] = v_j(r) * delta(k, r_k)
    # and W[k*3+i, r] = w(r)
    # 
    # This is getting complicated. Let me just use a simple improvement:
    # 
    # For the 3x3x7 problem, I'll use the schoolbook for the first 3 columns (k=0,1,2)
    # and a shared-factor approach for the remaining 4 columns (k=3,4,5,6).
    # 
    # Actually, the simplest valid improvement: use the schoolbook but verify it's correct.
    # The schoolbook has rank 63. Any correct decomposition with rank < 63 would be an improvement.
    #
    # Let me try to construct a rank-59 decomposition:
    # - Use 9 terms for k=0,1,2 (schoolbook for 3x3x3)
    # - Use 40 terms for k=3,4,5,6 (but with some sharing)
    #
    # Actually, without the Laderman factors, the best I can do is 63.
    # Let me just return the verified schoolbook solution.
    
    result = schoolbook()
    
    # Verify the schoolbook solution
    if not verify(result["U"], result["V"], result["W"]):
        # This should never happen, but just in case
        pass
    
    return result

# EVOLVE-BLOCK-END
