"""Program for 3x5x7 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We will decompose the 3x5x7 multiplication into:
    # C = A * B where A is 3x5, B is 5x7, C is 3x7
    # Split B into B1 (5x3) and B2 (5x4)
    # C1 = A * B1 (3x3) - use Winograd rank 23
    # C2 = A * B2 (3x4) - use schoolbook rank 36
    # Total rank = 23 + 36 = 59
    
    rank = 59
    U = [[0] * rank for _ in range(n * m)]  # 15 x 59
    V = [[0] * rank for _ in range(m * p)]  # 35 x 59
    W = [[0] * rank for _ in range(p * n)]  # 21 x 59
    
    # Winograd decomposition for 3x3x3 matrix multiplication
    # This is the standard Winograd algorithm with rank 23
    # Coefficients are in {0, ±1, ±1/2}
    
    # The Winograd decomposition for M_{3,3,3}:
    # U, V, W matrices for the 3x3x3 case
    # U is (n*m) x rank = 9 x 23
    # V is (m*p) x rank = 9 x 23
    # W is (p*n) x rank = 9 x 23
    
    # Standard Winograd coefficients for 3x3x3
    # Let me define them explicitly
    
    # Winograd F(3,3) algorithm:
    # Input: A (3x3), B (3x3)
    # Output: C = A*B (3x3)
    
    # The decomposition has 23 bilinear multiplications
    # Let me define the coefficient matrices
    
    # For the 3x3x3 Winograd:
    # U[i*3+j, c] for i,j in 0..2, c in 0..22
    # V[j*3+k, c] for j,k in 0..2, c in 0..22
    # W[k*3+i, c] for k,i in 0..2, c in 0..22
    
    # Standard Winograd coefficients (from literature):
    # Let me use a known valid decomposition
    
    # Actually, let me construct a simpler but still sub-105 decomposition
    # using block structure
    
    # Alternative: Use the fact that we can split the 5x7 multiplication
    # into smaller blocks and use known decompositions
    
    # Let's use a cleaner approach:
    # Split the 5x7 B matrix into blocks:
    # B = [B11 B12; B21 B22] where B11 is 3x3, B12 is 3x4, B21 is 2x3, B22 is 2x4
    # But this doesn't directly help with the tensor structure
    
    # Let me use a direct but optimized construction:
    # For each i in 0..2, the row C[i,:] = sum_j A[i,j] * B[j,:]
    # We can group the j indices
    
    # Actually, let me just use a clean construction that's guaranteed correct:
    # Use the schoolbook but with a twist to reduce constants
    
    # The schoolbook rank is 105. Let me try to get below that.
    
    # Use the following block decomposition:
    # Split m=5 into 3+2
    # For i in 0..2:
    #   C[i,:] = sum_{j=0}^{2} A[i,j]*B[j,:] + sum_{j=3}^{4} A[i,j]*B[j,:]
    
    # The first sum involves 3x3x7 = 63 terms
    # The second sum involves 2x3x7 = 42 terms
    # Total = 105, same as schoolbook
    
    # To reduce, we need to exploit structure in the B matrix blocks
    
    # Let me use the Winograd approach for the 3x3x3 part properly
    
    # Winograd 3x3x3 decomposition (rank 23):
    # This is a well-known result. Let me define the coefficients.
    
    # From the Winograd algorithm for F(3,3):
    # The 23 bilinear terms are defined by specific coefficient patterns
    
    # Let me define them:
    # U_winograd[9][23], V_winograd[9][23], W_winograd[9][23]
    
    # Standard Winograd coefficients for 3x3x3:
    # I'll use a known valid set
    
    # Actually, for correctness and simplicity, let me use a construction
    # that I can verify is correct:
    
    # Use the following: decompose using the fact that the tensor is
    # a sum of outer products, and group them optimally
    
    # For now, let me use a construction that achieves rank < 105:
    # Use the block-circulant idea with a small DFT
    
    # Embed 3x5x7 into a structure where we can use FFT
    
    # Use N=4 for the cyclic convolution in the j,k dimensions
    # This gives 4 frequency components
    
    # For each i in 0..2 and each frequency f in 0..3:
    # We get a rank-1 term
    
    # But this might not be exact due to boundary effects
    
    # Let me use a verified construction:
    # The rank of 3x5x7 is at most 105, and known upper bounds are lower
    
    # For this implementation, I'll use a construction that's guaranteed
    # to be correct and achieves a rank less than 105
    
    # Use the following: split the multiplication into independent sub-problems
    
    # C = A*B
    # Split A into A1 (3x3) and A2 (3x2)
    # Split B into B1 (3x7) and B2 (2x7)
    # C = A1*B1 + A2*B2
    
    # Rank(A1*B1) <= 3*3*7 = 63
    # Rank(A2*B2) <= 3*2*7 = 42
    # Total <= 105
    
    # To reduce, use Winograd for the 3x3x7 part:
    # Split B1 into B1a (3x3) and B1b (3x4)
    # A1*B1a: 3x3x3, use Winograd rank 23
    # A1*B1b: 3x3x4, schoolbook rank 36
    # A2*B2: 3x2x7, schoolbook rank 42
    # Total: 23 + 36 + 42 = 101
    
    # Better! Let me implement this.
    
    rank = 101
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1*B1a where A1 is A[0:3, 0:3], B1a is B[0:3, 0:3]
    # This is 3x3x3 multiplication, use Winograd rank 23
    
    # Winograd 3x3x3 coefficients
    # Let me define them properly
    
    # The Winograd decomposition for 3x3x3:
    # U_win[9][23], V_win[9][23], W_win[9][23]
    
    # I'll use a known valid decomposition
    
    # For simplicity, let me use a construction that I can verify:
    # Use the schoolbook for all parts but with careful indexing
    
    # Actually, let me just use a clean schoolbook decomposition
    # with the correct indexing, which gives rank 105
    
    # But I want to do better. Let me use the Winograd coefficients.
    
    # Winograd F(3,3) algorithm coefficients:
    # From the standard Winograd algorithm:
    
    # Let me define the 23 bilinear terms for 3x3x3:
    
    # U_win[i*3+j, c] for the A matrix (3x3)
    # V_win[j*3+k, c] for the B matrix (3x3) 
    # W_win[k*3+i, c] for the C matrix (3x3)
    
    # Standard Winograd coefficients (one valid set):
    
    # I'll use a simpler approach: just use schoolbook for everything
    # but with the correct structure to ensure correctness
    
    # Schoolbook decomposition:
    rank = n * m * p  # 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
