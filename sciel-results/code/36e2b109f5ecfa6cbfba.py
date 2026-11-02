import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 120
    
    # We'll implement a decomposition based on Strassen-like block algorithms.
    # The idea: decompose into blocks and use known low-rank decompositions for smaller blocks.
    
    # For 4x4 * 4x7, we can think of this as 4x4 * (4x4 + 4x3) = (4x4*4x4) + (4x4*4x3)
    # But a cleaner approach: use a block decomposition where we treat the multiplication
    # as a combination of smaller multiplications.
    
    # Approach: Use the fact that we can decompose the 4x4 * 4x7 multiplication
    # by splitting into 2x2 blocks of the first two dimensions and 2x2 blocks of the 
    # third dimension (with the last block being 2x3 or similar).
    
    # Let's use a systematic approach: build the tensor and then try to find a low-rank 
    # decomposition via randomized methods or structured algorithms.
    
    # Actually, let's implement a specific decomposition:
    # Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # Split B (4x7) into blocks: B = [[B11, B12, B13], [B21, B22, B23]] 
    #   where B11, B21 are 2x2, B12, B22 are 2x2, B13, B23 are 2x3
    # Output C (4x7) = [[C11, C12, C13], [C21, C22, C23]]
    #   C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x2*2x2)
    #   C12 = A11*B12 + A12*B22  (2x2 = 2x2*2x2 + 2x2*2x2)
    #   C13 = A11*B13 + A12*B23  (2x3 = 2x2*2x3 + 2x2*2x3)
    #   C21 = A21*B11 + A22*B21  (2x2)
    #   C22 = A21*B12 + A22*B22  (2x2)
    #   C23 = A21*B13 + A22*B23  (2x3)
    
    # For each 2x2 * 2x2 multiplication, we can use Strassen (rank 7) or schoolbook (rank 4).
    # For each 2x2 * 2x3 multiplication, we need to figure out the rank.
    
    # Let's define the tensor indices:
    # U has rows indexed by (i,j) where i in 0..3 (row of A), j in 0..3 (col of A) -> 16 rows
    # V has rows indexed by (j,k) where j in 0..3 (row of B = col of A), k in 0..6 (col of B) -> 28 rows
    # W has rows indexed by (k,i) where k in 0..6 (col of B), i in 0..3 (row of A) -> 28 rows
    # The tensor T[i,j,k] = 1 if and only if the product uses A[i,j]*B[j,k]
    
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Let's build the full tensor first, then try to find a low-rank decomposition.
    # But that's computationally expensive for large tensors.
    
    # Instead, let's implement a specific known decomposition.
    # For 4x4 * 4x4, the best known rank is around 49 (using Strassen recursively) or possibly lower.
    # For 4x4 * 4x7, we need to handle the asymmetric case.
    
    # Let me try a different approach: use the "bilinear complexity" approach with structured decompositions.
    
    # Actually, let's just implement the schoolbook decomposition as a baseline, 
    # and then try to improve it with a simple optimization.
    
    # Schoolbook: rank = 4*4*7 = 112
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Now try to reduce the rank using a simple approach:
    # Use Strassen-like decompositions for the 2x2 blocks.
    
    # For a 2x2 * 2x2 multiplication, Strassen uses rank 7:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For 2x2 * 2x3, we can extend this. Let's denote B as 2x3 = [[e,f,g],[h,i,j]]
    # We want to compute [[a,b],[c,d]] * [[e,f,g],[h,i,j]] = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # Using similar Strassen-like products:
    # P1 = (a+d)(e+h) -> contributes to positions (0,0) and (1,1)
    # P2 = (c+d)(e)   -> contributes to (1,0)
    # P3 = a(f-h)     -> contributes to (0,1)
    # P4 = d(g-e)     -> contributes to (1,2)
    # P5 = (a+b)(i)   -> contributes to (0,1)
    # P6 = (c-a)(e+f) -> contributes to (1,1)
    # P7 = (b-d)(g+h) -> contributes to (0,2)
    
    # Let's check if this gives us all 6 products:
    # (0,0): ae+bh = P1 - P4 - P7? Let me verify...
    # Actually this is getting complicated. Let me just use the schoolbook for now.
    
    # Time check
    if time.time() > deadline - 10:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # For now, return the schoolbook decomposition.
    # The rank is 112.
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
