"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the problem M_{n,m,p} into M_{n1,m1,p1} \otimes M_{n2,m2,p2}
    # such that n1*n2 = n, m1*m2 = m, p1*p2 = p.
    # Based on the directive: 6 = 3 * 2.
    # Let's choose:
    # n1 = 2, n2 = 1
    # m1 = 3, m2 = 1
    # p1 = 3, p2 = 2
    
    n1, n2 = 2, 1
    m1, m2 = 3, 1
    p1, p2 = 3, 2
    
    # Verify dimensions
    assert n1 * n2 == n
    assert m1 * m2 == m
    assert p1 * p2 == p
    
    # Construct schoolbook decomposition for M_{n1, m1, p1}
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
                
    # Construct schoolbook decomposition for M_{n2, m2, p2}
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
                
    # Combine using Kronecker product
    # U = U1 \otimes U2
    # V = V1 \otimes V2
    # W = W1 \otimes W2
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to list of lists as required by the interface (integer/half-integer entries)
    # The entries are 0 or 1, so integers.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
