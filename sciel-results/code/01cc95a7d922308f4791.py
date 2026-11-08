"""Program for rank-minimal tensor decomposition of 2x3x10 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a rank-30 decomposition based on the diagonal structure of the middle dimension.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    rank = m * p  # 3 * 10 = 30
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U: indices (i, j_prime) -> i in {0,1}, j_prime in {0,1,2}
            # U[(i, j_prime), r] = 1 if j_prime == j
            for i in range(n):
                for jp in range(m):
                    if jp == j:
                        U[i * m + jp, r] = 1
                        
            # V: indices (j_double_prime, k_prime) -> j_dp in {0,1,2}, k_p in {0..9}
            # V[(j_dp, k_p), r] = 1 if j_dp == j and k_p == k
            for jdp in range(m):
                for kp in range(p):
                    if jdp == j and kp == k:
                        V[jdp * p + kp, r] = 1
                        
            # W: indices (k_double_prime, i_prime) -> k_dp in {0..9}, i_p in {0,1}
            # W[(k_dp, i_p), r] = 1 if k_dp == k
            for kdp in range(p):
                for ip in range(n):
                    if kdp == k:
                        W[kdp * n + ip, r] = 1
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
