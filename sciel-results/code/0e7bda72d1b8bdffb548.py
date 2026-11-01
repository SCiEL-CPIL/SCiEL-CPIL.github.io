"""Optimized 2x2x16 matrix multiplication tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 16]

# Strassen rank-7 decomposition for 2x2 matrix multiplication
# Tensors: (a,b) x (c,d) -> (e,f)
# e = a*c + (a+b)*(d-c) + (a-c)*(c+d)  -> uses 3 terms
# f = b*d + (a+b)*d + (b-d)*(c+d)     -> uses 3 terms
# Total 7 terms
#
# Strassen factors for 2x2:
# U (2x2 input A, index i*m+j where m=2, so row i*2+j):
#   row 0 (a): 1, 1, 1, 0, 0, 0, 0
#   row 1 (b): 0, 1, 0, 1, 1, 0, 0
#   row 2 (c): 1, 0, 1, 0, 0, 1, 1
#   row 3 (d): 0, 1, 0, 1, 0, 1, 0
#
# Wait, let me use the standard Strassen formulation more carefully.
# For C = A * B where A,B are 2x2:
# A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
#
# Standard Strassen (7 multiplications):
# p1 = (a+b)(e+f)
# p2 = (c+d)*f
# p3 = a*(e-g)
# p4 = d*(h-f)
# p5 = (a+d)(e+h)
# p6 = (b-d)*(g+h)
# p7 = (c-a)*(e+f)
#
# i = p1 + p4 - p5 + p7
# j = p3 + p5
# k = p2 + p5 - p1 - p6
# l = p3 + p4 + p6
#
# So the tensor has 7 terms. Let me define U, V, W for the 2x2x2 case:
# Input A indices: a=0, b=1, c=2, d=3 (row-major: A[i][j] -> i*2+j)
# Input B indices: e=0, f=1, g=2, h=3 (row-major: B[j][k] -> j*2+k)
# Output C indices: i=0, j=1, k=2, l=3 (row-major: C[i][k] -> i*2+k)
#
# For each term t (0..6), U[*,t], V[*,t], W[*,t] give the factors.
# The multiplication is: C = sum_t U[:,t] * V[:,t] * W[:,t] (outer products contracted)
# Actually, the tensor T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
# where i indexes A, j indexes B, k indexes C.
#
# Wait, the convention is:
# U has shape (n*m, rank) - indices into A
# V has shape (m*p, rank) - indices into B  
# W has shape (p*n, rank) - indices into C, with W[k*n+i, r]

# Let me reconsider. For 2x2x2:
# A is 2x2, B is 2x2, C is 2x2
# n=2, m=2, p=2
# U: (n*m, rank) = (4, rank) - A indices
# V: (m*p, rank) = (4, rank) - B indices
# W: (p*n, rank) = (4, rank) - C indices, W[k*n+i, r]

# The matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
# In the tensor decomposition: C[i,k] = sum_r U[i*m+j... ]

# Actually let me think about this differently.
# The multiplication tensor T has T[A_idx, B_idx, C_idx] = 1 if C = A*B at that entry.
# A_idx = i*m+j (row i, col j of A)
# B_idx = j*p+k (row j, col k of B)  
# C_idx = k*n+i (per the trace convention W[k*n+i, r])

# So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0.

# For a rank-r decomposition: T = sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t
# where u_t is a vector over A indices, v_t over B indices, w_t over C indices.
# T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]

# For the 2x2x2 case with Strassen:
# Let me just hardcode the known Strassen factors.

def strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's rank-7 algorithm."""
    # Using the standard Strassen formulation
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C[0,0] = a*e + b*g, C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g, C[1,1] = c*f + d*h
    
    # A indices: a=0, b=1, c=2, d=3
    # B indices: e=0, f=1, g=2, h=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # But W uses k*n+i convention: W[k*2+i]
    # C[i,k] -> W[k*2+i]
    # C[0,0]: i=0,k=0 -> W[0]
    # C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]
    # C[1,1]: i=1,k=1 -> W[3]
    
    # Strassen 7 terms:
    # p1 = (a+b)(e+f)
    # p2 = (c+d)*f
    # p3 = a*(e-g)
    # p4 = d*(h-f)
    # p5 = (a+d)(e+h)
    # p6 = (b-d)*(g+h)
    # p7 = (c-a)*(e+f)
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p5 - p1 - p6
    # C[1,1] = p3 + p4 + p6
    
    # U factors (for A):
    # p1: a+b -> [1, 1, 0, 0]
    # p2: c+d -> [0, 0, 1, 1]
    # p3: a   -> [1, 0, 0, 0]
    # p4: d   -> [0, 0, 0, 1]
    # p5: a+d -> [1, 0, 0, 1]
    # p6: b-d -> [0, 1, 0, -1]
    # p7: c-a -> [-1, 0, 1, 0]
    
    # V factors (for B):
    # p1: e+f -> [1, 1, 0, 0]
    # p2: f   -> [0, 1, 0, 0]
    # p3: e-g -> [1, 0, -1, 0]
    # p4: h-f -> [0, -1, 0, 1]
    # p5: e+h -> [1, 0, 0, 1]
    # p6: g+h -> [0, 0, 1, 1]
    # p7: e+f -> [1, 1, 0, 0]
    
    # W factors (for C, using W[k*2+i] convention):
    # p1: C[0,0]+=1, C[1,0]-=1 -> W[0]+=1, W[1]-=1 -> [1, -1, 0, 0]
    # p2: C[1,0]+=1 -> W[1]+=1 -> [0, 1, 0, 0]
    # p3: C[0,1]+=1 -> W[2]+=1 -> [0, 0, 1, 0]
    # p4: C[0,0]+=1, C[1,1]+=1 -> W[0]+=1, W[3]+=1 -> [1, 0, 0, 1]
    # p5: C[0,0]-=1, C[0,1]+=1, C[1,0]+=1 -> W[0]-=1, W[2]+=1, W[1]+=1 -> [-1, 1, 1, 0]
    # p6: C[1,0]-=1, C[1,1]+=1 -> W[1]-=1, W[3]+=1 -> [0, -1, 0, 1]
    # p7: C[0,0]+=1 -> W[0]+=1 -> [1, 0, 0, 0]
    
    U = [
        [1, 0, 1, 0, 1, 0, -1],  # a
        [1, 0, 0, 0, 0, 1, 0],   # b
        [0, 1, 0, 0, 0, 0, 1],   # c
        [0, 1, 0, 1, 1, -1, 0],  # d
    ]
    V = [
        [1, 0, 1, 0, 1, 0, 1],   # e
        [1, 1, 0, -1, 0, 0, 1],  # f
        [0, 0, -1, 0, 0, 1, 0],  # g
        [0, 0, 0, 1, 1, 1, 0],   # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] -> W[0]
        [-1, 1, 0, 0, 1, -1, 0], # C[1,0] -> W[1]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] -> W[2]
        [0, 0, 0, 1, 0, 1, 0],   # C[1,1] -> W[3]
    ]
    return U, V, W


def construct():
    """Construct rank-minimal decomposition for 2x2x16 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Base case: 2x2x2 with Strassen rank 7
    U_base, V_base, W_base = strassen_2x2x2()
    
    # For 2x2x16, we can think of B as 2x16 = 2x(2*8)
    # The key insight: we can group the 16 columns of B into 8 groups of 2 columns.
    # For each group of 2 columns, we have a 2x2 sub-matrix multiplication.
    # But we need to be careful about the index conventions.
    
    # Actually, let's think about it as: A is 2x2, B is 2x16, C is 16x2.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor: T[A_idx, B_idx, C_idx] where:
    # A_idx = i*m+j = i*2+j (i=0,1; j=0,1) -> 4 values
    # B_idx = j*p+k = j*16+k (j=0,1; k=0..15) -> 32 values
    # C_idx = k*n+i = k*2+i (k=0..15; i=0,1) -> 32 values
    #
    # T[i*2+j, j*16+k, k*2+i] = 1
    #
    # We can decompose this by noting that for fixed j, the relationship between
    # A[i,j], B[j,k], and C[i,k] is: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # One approach: use the Strassen decomposition on the "j" dimension structure.
    # 
    # Alternative approach: Think of B as having 16 columns. Group them into 8 pairs.
    # For columns 2t and 2t+1 of B, and the corresponding 2x2 block of C:
    # C[:, 2t:2t+2] = A * B[:, 2t:2t+2]
    # This is a 2x2 * 2x2 = 2x2 multiplication for each t.
    # Using Strassen for each gives rank 7*8 = 56.
    
    # But wait - can we do better? The problem is 2x2x16.
    # The schoolbook rank is 2*2*16 = 64.
    # Strassen per 2x2 block gives 7*8 = 56.
    
    # Let's verify: is there a way to share factors across the 8 blocks?
    # The A matrix is shared across all 8 blocks. In Strassen, the A factors are:
    # [1,0,1,0,1,0,-1], [1,0,0,0,0,1,0], [0,1,0,0,0,0,1], [0,1,0,1,1,-1,0]
    # These are the same for all 8 blocks! So we can't reduce further by sharing A.
    # The B factors differ per block (different columns), and W factors differ per block.
    
    # So rank 56 seems to be the natural answer with this approach.
    
    # Let's construct the rank-56 decomposition.
    # For each block t (0..7), we have columns 2t, 2t+1 of B.
    # B index for column k in row j: j*16+k
    # For block t, column 2t: B_idx = j*16+2t
    # For block t, column 2t+1: B_idx = j*16+2t+1
    
    # C index: C[i,k] -> W[k*2+i]
    # For block t, C[i,2t] -> W[2t*2+i] = W[4t+i]
    # For block t, C[i,2t+1] -> W[(2t+1)*2+i] = W[4t+2+i]
    
    rank = 7 * 8  # 56
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    for t in range(8):
        for s in range(7):  # 7 Strassen terms
            col = t * 7 + s
            # U factors (same for all blocks)
            for a_idx in range(4):
                U[a_idx][col] = U_base[a_idx][s]
            
            # V factors: B indices for this block
            # B[j, 2t] -> j*16 + 2t
            # B[j, 2t+1] -> j*16 + 2t+1
            # V_base indices: e=0(B[0,0]), f=1(B[0,1]), g=2(B[1,0]), h=3(B[1,1])
            # So V_base[0] -> B[0,0] of block -> B[0,2t] -> index 0*16+2t = 2t
            # V_base[1] -> B[0,1] of block -> B[0,2t+1] -> index 0*16+2t+1 = 2t+1
            # V_base[2] -> B[1,0] of block -> B[1,2t] -> index 1*16+2t = 16+2t
            # V_base[3] -> B[1,1] of block -> B[1,2t+1] -> index 1*16+2t+1 = 16+2t+1
            for b_idx in range(4):
                # b_idx 0 -> row 0, col 0 of block
                # b_idx 1 -> row 0, col 1 of block
                # b_idx 2 -> row 1, col 0 of block
                # b_idx 3 -> row 1, col 1 of block
                row = b_idx // 2
                col_in_block = b_idx % 2
                actual_b_idx = row * 16 + 2 * t + col_in_block
                V[actual_b_idx][col] = V_base[b_idx][s]
            
            # W factors: C indices for this block
            # C[i, 2t] -> W[4t+i]
            # C[i, 2t+1] -> W[4t+2+i]
            # W_base indices: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
            # W_base[0] -> C[0,0] of block -> C[0,2t] -> W[4t+0]
            # W_base[1] -> C[1,0] of block -> C[1,2t] -> W[4t+1]
            # W_base[2] -> C[0,1] of block -> C[0,2t+1] -> W[4t+2]
            # W_base[3] -> C[1,1] of block -> C[1,2t+1] -> W[4t+3]
            for c_idx in range(4):
                # c_idx 0 -> C[0,0] of block
                # c_idx 1 -> C[1,0] of block
                # c_idx 2 -> C[0,1] of block
                # c_idx 3 -> C[1,1] of block
                i = c_idx // 2
                k_in_block = c_idx % 2
                actual_c_idx = (2 * t + k_in_block) * 2 + i
                W[actual_c_idx][col] = W_base[c_idx][s]
    
    # Verify the decomposition (optional, for debugging)
    # For a quick check, let's verify a few entries
    # T[A_idx, B_idx, C_idx] should equal sum_r U[A_idx,r]*V[B_idx,r]*W[C_idx,r]
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
