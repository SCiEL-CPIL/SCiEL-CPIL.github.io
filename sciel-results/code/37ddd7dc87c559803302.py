# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    This decomposition is rank-minimal (rank 7 is the known lower bound).
    The output follows the official trace convention for W.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    # Rank 7 is the minimum rank for 2x2 matrix multiplication
    rank = 7
    
    # Initialize matrices with zeros
    # U: (4, 7) for A (indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1])
    # V: (4, 7) for B (indices 0..3 correspond to B[0,0], B[0,1], B[1,0], B[1,1])
    # W: (4, 7) for C (indices 0..3 correspond to C[0,0], C[0,1], C[1,0], C[1,1] in trace convention k*n+i)
    # Note on W indexing: The task specifies W[k*n+i, r].
    # For 2x2, n=2. So W[k*2+i, r].
    # C[i,k] is the standard matrix multiplication output.
    # The "official trace convention" usually implies the output index is mapped to a specific linear index.
    # Let's verify the indexing in the current seed:
    # Seed comment: "W[k*n+i][r] corresponds to C[i,k] output"
    # Wait, standard C[i,k] = sum_j A[i,j] B[j,k].
    # If W index is k*2+i, then:
    # W[0] -> k=0, i=0 -> C[0,0]
    # W[1] -> k=0, i=1 -> C[1,0]
    # W[2] -> k=1, i=0 -> C[0,1]
    # W[3] -> k=1, i=1 -> C[1,1]
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to map these to U, V, W such that:
    # C[i,k] = sum_r W[k*2+i, r] * (sum_j U[i*2+j, r] A[i,j]) * (sum_j V[j*2+k, r] B[j,k])
    # Actually, the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Where T_{i,j,k} is 1 if it's a term in C[i,k] = sum_j A[i,j]B[j,k].
    # The output C[i,k] accumulates terms.
    
    # Let's stick to the verified decomposition from the seed, which is known to work.
    # The seed uses:
    # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
    # V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
    # W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11 (based on k*2+i mapping)
    
    # r=0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
