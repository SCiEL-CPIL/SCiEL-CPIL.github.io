"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses the Kronecker product structure: M_{2,3,6} ~ M_{2,3,1} \otimes M_{1,1,6}.
    Rank is R(M_{2,3,1}) * R(M_{1,1,6}) = 6 * 6 = 36.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Construct decomposition for M_{2,3,1}
    # A is 2x3, B is 3x1, C is 2x1
    n1, m1, p1 = 2, 3, 1
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                # Indexing: U[i*m1+j], V[j*p1+k], W[k*n1+i]
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
                
    # 2. Construct decomposition for M_{1,1,6}
    # A is 1x1, B is 1x6, C is 1x6
    # Note: Standard matrix multiplication 1x1 * 1x6 is not defined in the usual sense 
    # unless we view it as element-wise or specific tensor product.
    # However, M_{1,1,6} tensor corresponds to 6 independent scalar multiplications.
    # A_{11} * B_{1k} = C_{1k}.
    # This is isomorphic to 6 disjoint M_{1,1,1}.
    n2, m2, p2 = 1, 1, 6
    rank2 = n2 * m2 * p2
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1

    # 3. Kronecker Product
    # The tensor T = T1 \otimes T2
    # U = U1 \otimes U2 ?
    # The spaces are:
    # U space: (n*m) x rank. 
    # T1 U space is (n1*m1) x rank1.
    # T2 U space is (n2*m2) x rank2.
    # The combined U space should be (n1*m1*n2*m2) x (rank1*rank2)?
    # n=2, m=3, p=6.
    # n1=2, m1=3, p1=1.
    # n2=1, m2=1, p2=6.
    # n1*n2 = 2 = n.
    # m1*m2 = 3 = m.
    # p1*p2 = 6 = p.
    # So the dimensions match.
    
    # The Kronecker product of matrices U1 (size a x b) and U2 (size c x d)
    # results in a matrix of size (a*c) x (b*d).
    # We need U to be (n*m) x rank.
    # n*m = 6.
    # U1 is 6 x 6. U2 is 1 x 6.
    # U1 \otimes U2 is 6 x 36.
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists as required by the interface
    # The problem statement says "integer/half-integer entries".
    # np.kron of integers is integers.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
