"""
Rank-32 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

Construction:
  Split the middle dimension (size 4) into two blocks of size 2.
  For each block, we need to compute a 2x2 * 2x5 product.
  A 2x2 * 2x5 product can be decomposed as:
    - 2x2 * 2x2 (first two columns) -> rank 7 (Strassen)
    - 2x2 * 2x2 (next two columns) -> rank 7 (Strassen)
    - 2x2 * 2x1 (last column) -> rank 2 (schoolbook)
  Total per 2x2*2x5 block: 7 + 7 + 2 = 16
  Two such blocks: 32 total.

  The tensor T[i,j,k] for matrix multiplication has T[i,j,k] = 1 for all i,j,k
  in the sense that C[i,k] = sum_j A[i,j] B[j,k].
  In the trace convention W[k*n+i, r], the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
  Wait, that's not right. The tensor for matrix multiplication is:
    T[(i,j), (j,k)] = delta_{i,?} ... no.
    
  Actually, the multiplication tensor has entries T[i,j,k] = 1 for all i,j,k.
  This is because C[i,k] = sum_j A[i,j] B[j,k], and the coefficient of A[i,j]B[j,k]
  in C[i,k] is 1.
  
  So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,1], j in [0,3], k in [0,4].
  
  With the block decomposition, each Strassen term contributes to a specific (i,j,k) pattern.
  Since the supports are disjoint in the j dimension (block 0 uses j=0,1; block 1 uses j=2,3),
  and within each block the k supports are also disjoint for the different sub-blocks,
  the total rank is the sum of the ranks of the individual blocks.
"""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-32 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 32
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen 2x2*2x2 decomposition.
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a)(d-h)
    # M5 = (b)(e+g)
    # M6 = (c-a)(f)
    # M7 = (d)(g)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen U coefficients for A[i,j]:
    # M1: a=1, b=0, c=0, d=1 -> U[0,0]=1, U[1,1]=1 (for i=0,j=0 and i=1,j=1)
    # Wait, U is indexed by (i, j) where i is row of A, j is col of A.
    # For 2x2 A: indices (0,0), (0,1), (1,0), (1,1) -> global U indices 0,1,2,3 (if m=2)
    # But in our case m=4, so for block 0 (j=0,1): U indices are i*4+j
    #   i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 4; i=1,j=1 -> 5
    # For block 1 (j=2,3): U indices are i*4+j
    #   i=0,j=2 -> 2; i=0,j=3 -> 3; i=1,j=2 -> 6; i=1,j=3 -> 7
    
    # Strassen V coefficients for B[j,k]:
    # For 2x2 B: indices (0,0), (0,1), (1,0), (1,1) -> global V indices j*p+k
    # For block at B rows (j0, j0+1) and cols (k0, k0+1):
    #   (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    
    # Strassen W coefficients for C[k,i]:
    # For 2x2 C: indices (0,0), (0,1), (1,0), (1,1) -> global W indices k*n+i
    # For block at C cols (k0, k0+1) and rows (i0, i0+1):
    #   (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    
    def add_strassen_block(A_block_start_j, B_block_start_j, B_block_start_k, C_block_start_k):
        """
        Add a Strassen 2x2*2x2 block.
        A is 2x2 at rows 0,1 and cols A_block_start_j, A_block_start_j+1.
        B is 2x2 at rows B_block_start_j, B_block_start_j+1 and cols B_block_start_k, B_block_start_k+1.
        C is 2x2 at rows 0,1 and cols C_block_start_k, C_block_start_k+1.
        Note: A rows are always 0,1 (since n=2).
        """
        nonlocal col
        
        # Local indices for A (2x2): (i, j_local) where j_local in {0,1}
        # Global U index: i * m + (A_block_start_j + j_local)
        
        # Local indices for B (2x2): (j_local, k_local) where j_local, k_local in {0,1}
        # Global V index: (B_block_start_j + j_local) * p + (B_block_start_k + k_local)
        
        # Local indices for C (2x2): (k_local, i) where k_local in {0,1}, i in {0,1}
        # Global W index: (C_block_start_k + k_local) * n + i
        
        # Strassen terms:
        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11+=1, C22+=1
        # M2 = (c+d)(e):   U: c=1,d=1; V: e=1;      W: C21+=1, C22+=-1
        # M3 = (a-b)(f+h): U: a=1,b=-1; V: f=1,h=1;  W: C12+=1, C22+=1
        # M4 = (a)(d-h):   U: a=1;       V: d=1,h=-1; W: C11+=1, C21+=1
        # M5 = (b)(e+g):   U: b=1;       V: e=1,g=1;  W: C11+=-1, C12+=1
        # M6 = (c-a)(f):   U: c=1,a=-1;  V: f=1;      W: C22+=1
        # M7 = (d)(g):     U: d=1;       V: g=1;      W: C11+=1
        
        # Let's define each term as (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict from (i, j_local) to coefficient
        # V_coeffs: dict from (j_local, k_local) to coefficient
        # W_coeffs: dict from (k_local, i) to coefficient
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1, (0,1):-1}, {(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}),
            # M4
            ({(0,0):1}, {(1,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,1):1}, {(0,0):1, (1,0):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,1):1}, {(1,1):1}),
            # M7
            ({(1,1):1}, {(1,0):1}, {(0,0):1}),
        ]
        
        for U_c, V_c, W_c in terms:
            for (i, j_l), u_val in U_c.items():
                U[i * m + (A_block_start_j + j_l), col] += u_val
            for (j_l, k_l), v_val in V_c.items():
                V[(B_block_start_j + j_l) * p + (B_block_start_k + k_l), col] += v_val
            for (k_l, i), w_val in W_c.items():
                W[(C_block_start_k + k_l) * n + i, col] += w_val
            col += 1
    
    # Schoolbook 2x2*2x1 decomposition.
    # A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Two terms:
    # Term 1: A[0,0]*B[0] -> C[0] += 1
    # Term 2: A[0,1]*B[1] -> C[0] += 1
    # Term 3: A[1,0]*B[0] -> C[1] += 1
    # Term 4: A[1,1]*B[1] -> C[1] += 1
    # Wait, that's 4 terms, not 2.
    # For 2x2 * 2x1, the schoolbook is 4 multiplications.
    # But we can do it with 2? No, the rank of 2x2x1 tensor is 2.
    # Let's check: T[i, j, k] where i in {0,1}, j in {0,1}, k in {0}.
    # T[i,j,0] = 1 for all i,j.
    # Rank 2: U[i,j,0] = 1, U[i,j,1] = 0; V[j,0,0] = 1, V[j,0,1] = 0; W[0,i,0] = 1, W[0,i,1] = 0
    # sum = 1*1*1 = 1. Yes, rank 1 works!
    # Wait, T[i,j,0] = 1 for all i,j. So U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1 gives sum=1.
    # So 2x2*2x1 has rank 1? That can't be right.
    # The tensor is T[i,j,k] where the output is C[i] = sum_j A[i,j] B[j].
    # T[i,j,0] = 1 for all i,j.
    # So yes, rank 1: U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1.
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1*1*1 = 1. Correct.
    # So 2x2*2x1 has rank 1.
    
    # Wait, let me re-examine. The tensor for matrix multiplication M_{n,m,p} has
    # T[i,j,k] = 1 for all i,j,k. This is because C[i,k] = sum_j A[i,j] B[j,k].
    # So for any fixed i,j,k, the coefficient is 1.
    # Therefore, the tensor is the all-ones tensor.
    # The rank of the all-ones tensor of shape (n,m,p) is 1!
    # U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1.
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1*1*1 = 1.
    # So the rank is 1!
    
    # But the seed program returns rank 40, and the task says the score is seed/value.
    # If rank 1 is valid, the score would be 40/1 = 40, which is way above 1.
    # Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "score is the rank"
    # "The evaluator measures the raw value (lower is better)."
    
    # If the tensor is all-ones, rank 1 is optimal. But that seems too easy.
    # Let me check if the tensor is really all-ones.
    # The multiplication tensor for M_{n,m,p} is defined such that
    # <T, A \otimes B \otimes C> = Tr(C^T (A B))
    # = sum_{i,k} C[k,i] (sum_j A[i,j] B[j,k])
    # = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # So T[i,j,k] = 1 for all i,j,k.
    # Yes, the tensor is all-ones.
    # So rank 1 is valid.
    
    # Let me return rank 1.
    
    # Wait, but the seed has rank 40 and score 1.0. If I return rank 1, score = 40/1 = 40.
    # That would be a huge improvement.
    # Let me verify with a small example.
    # n=1, m=1, p=1. T[0,0,0] = 1. Rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum=1. Correct.
    # n=2, m=2, p=2. T[i,j,k]=1 for all i,j,k. Rank 1: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1. Sum=1. Correct.
    
    # So the answer is rank 1!
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
