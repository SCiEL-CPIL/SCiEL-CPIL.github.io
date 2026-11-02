"""Program for matrix multiplication 2,2,10 using Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for
    multiplying a 2x2 matrix by a 2x10 matrix.
    
    Approach:
    - Split the 2x10 matrix into five 2x2 blocks: B = [B1, B2, B3, B4, B5]
    - Compute C = A * B = A*B1 + A*B2 + A*B3 + A*B4 + A*B5
    - Each A*Bj is a 2x2 * 2x2 multiplication, which can be done with Strassen (rank 7)
    - Total rank would be 5 * 7 = 35 if independent
    
    However, we can potentially share computations or use a smarter approach.
    
    Let's use the standard schoolbook for each 2x2 block but with Strassen's 7 multiplications per block.
    
    Strassen's algorithm for 2x2 matrices:
    Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Result:
    C11 = M1 - M4 + M5 + M6
    C12 = M3 + M4
    C21 = M2 + M4
    C22 = M1 - M3 + M5 - M7
    
    For our case, A is fixed 2x2, and B is 2x10 split into five 2x2 blocks.
    Each block multiplication uses 7 scalar multiplications.
    Total rank = 5 * 7 = 35.
    
    But wait - we can do better. Let's think about this differently.
    
    Actually, let's just implement the Strassen-based approach with rank 35 first,
    and see if we can optimize further.
    
    The tensor is T[i, j, k] where i is row of output (0..1), j is column of A (0..1), k is column of B (0..9).
    T[i,j,k] = sum over l of A[i,l] * B[l,k]
    
    Using Strassen on each 2x2 block:
    For block t (t=0..4), columns 2t and 2t+1 of B:
    B_t = [[B[0,2t], B[0,2t+1]], [B[1,2t], B[1,2t+1]]]
    
    A = [[a00, a01], [a10, a11]]
    
    M1_t = (a00+a11)(B[0,2t]+B[1,2t+1])
    M2_t = (a10+a11)B[0,2t]
    M3_t = a00(B[0,2t+1]-B[1,2t+1])
    M4_t = a11(B[1,2t]-B[0,2t])
    M5_t = (a00+a01)B[1,2t+1]
    M6_t = (a10-a00)(B[0,2t]+B[0,2t+1])
    M7_t = (a01-a11)(B[1,2t]+B[1,2t+1])
    
    C[0,2t] = M1_t - M4_t + M5_t + M6_t
    C[0,2t+1] = M3_t + M4_t
    C[1,2t] = M2_t + M4_t
    C[1,2t+1] = M1_t - M3_t + M5_t - M7_t
    
    Each M*_t involves one scalar multiplication (product of a linear combination of A entries and a linear combination of B entries).
    So 7 multiplications per block, 5 blocks = 35 total.
    
    Now let's construct the U, V, W matrices.
    
    U has shape (n*m, rank) = (4, 35)
    V has shape (m*p, rank) = (20, 35)  
    W has shape (p*n, rank) = (20, 35)
    
    Indexing:
    U[i*m+j, c] where i in 0..1, j in 0..1 (row, col of A)
    V[j*p+k, c] where j in 0..1, k in 0..9 (row of B, col of B)
    W[k*n+i, c] where k in 0..9, i in 0..1 (col of output, row of output) - trace convention W[k*n+i, r]
    
    For each block t and each Strassen multiplication m (0..6):
    Column c = t*7 + m
    
    Let's define the Strassen coefficients:
    For M1: A part = a00+a11, B part = B[0,2t]+B[1,2t+1]
    For M2: A part = a10+a11, B part = B[0,2t]
    For M3: A part = a00, B part = B[0,2t+1]-B[1,2t+1]
    For M4: A part = a11, B part = B[1,2t]-B[0,2t]
    For M5: A part = a00+a01, B part = B[1,2t+1]
    For M6: A part = a10-a00, B part = B[0,2t]+B[0,2t+1]
    For M7: A part = a01-a11, B part = B[1,2t]+B[1,2t+1]
    
    Output contributions:
    C[0,2t] += M1 - M4 + M5 + M6
    C[0,2t+1] += M3 + M4
    C[1,2t] += M2 + M4
    C[1,2t+1] += M1 - M3 + M5 - M7
    
    So for each (t, m), we have:
    U entries: the coefficient of each A[i,j] in the A-part of M_m
    V entries: the coefficient of each B[j,k] in the B-part of M_m
    W entries: the coefficient in the output C[i,k]
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Strassen parameters for each of the 7 multiplications
    # A coefficients: for each mult, list of (i, j, coeff) for A[i,j]
    # B coefficients: for each mult, list of (j, k_offset, coeff) for B[j, 2t+k_offset]
    # W coefficients: for each mult, list of (i_out, k_offset, coeff) for C[i_out, 2t+k_offset]
    
    strassen_A = [
        [(0,0,1), (1,1,1)],      # M1: a00+a11
        [(1,0,1), (1,1,1)],      # M2: a10+a11
        [(0,0,1)],                # M3: a00
        [(1,1,1)],                # M4: a11
        [(0,0,1), (0,1,1)],      # M5: a00+a01
        [(1,0,1), (0,0,-1)],     # M6: a10-a00
        [(0,1,1), (1,1,-1)],     # M7: a01-a11
    ]
    
    strassen_B = [
        [(0,0,1), (1,1,1)],      # M1: B[0,2t]+B[1,2t+1]
        [(0,0,1)],                # M2: B[0,2t]
        [(0,1,1), (1,1,-1)],     # M3: B[0,2t+1]-B[1,2t+1]
        [(1,0,1), (0,0,-1)],     # M4: B[1,2t]-B[0,2t]
        [(1,1,1)],                # M5: B[1,2t+1]
        [(0,0,1), (0,1,1)],      # M6: B[0,2t]+B[0,2t+1]
        [(1,0,1), (1,1,1)],      # M7: B[1,2t]+B[1,2t+1]
    ]
    
    strassen_W = [
        [(0,0,1), (0,1,-1), (1,0,1), (1,1,1)],  # M1: C[0,2t]+=1, C[0,2t+1]-=1, C[1,2t]+=1, C[1,2t+1]+=1
        [(1,0,1)],                                 # M2: C[1,2t]+=1
        [(0,1,1), (1,1,-1)],                       # M3: C[0,2t+1]+=1, C[1,2t+1]-=1
        [(0,0,-1), (0,1,1), (1,0,1)],             # M4: C[0,2t]-=1, C[0,2t+1]+=1, C[1,2t]+=1
        [(0,0,1), (1,1,1)],                        # M5: C[0,2t]+=1, C[1,2t+1]+=1
        [(0,0,1)],                                 # M6: C[0,2t]+=1
        [(1,1,-1)],                                # M7: C[1,2t+1]-=1
    ]
    
    rank = 5 * 7  # 35
    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)
    
    for t in range(5):
        for sm in range(7):
            col = t * 7 + sm
            
            # Fill U: A coefficients
            for (i, j, coeff) in strassen_A[sm]:
                U[i * m + j][col] += coeff
            
            # Fill V: B coefficients
            for (j, k_off, coeff) in strassen_B[sm]:
                k = 2 * t + k_off
                V[j * p + k][col] += coeff
            
            # Fill W: output coefficients
            for (i_out, k_off, coeff) in strassen_W[sm]:
                k = 2 * t + k_off
                W[k * n + i_out][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
