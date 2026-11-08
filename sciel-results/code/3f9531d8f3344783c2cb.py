"""Program for rank-minimal tensor decomposition of 2x3x10 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x3x10 matrix multiplication.
    
    Uses a block structure: 5 blocks of 2 columns each, with rank 55 total.
    The decomposition is constructed by combining small rank-optimal sub-decompositions.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 55
    V = np.zeros((m * p, rank), dtype=int)  # 30 x 55
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 55
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        base_col = 11 * t
        
        # Term 0
        U[0, base_col] += 1
        U[4, base_col] += 1
        V[0*10 + k0, base_col] += 1
        V[1*10 + k1, base_col] += 1
        W[k0*2 + 0, base_col] += 1
        W[k1*2 + 1, base_col] += 1
        
        # Term 1
        U[3, base_col + 1] += 1
        U[4, base_col + 1] += 1
        V[0*10 + k0, base_col + 1] += 1
        W[k0*2 + 1, base_col + 1] += 1
        W[k1*2 + 1, base_col + 1] -= 1
        
        # Term 2
        U[0, base_col + 2] += 1
        V[0*10 + k1, base_col + 2] += 1
        V[1*10 + k1, base_col + 2] -= 1
        W[k1*2 + 0, base_col + 2] += 1
        W[k1*2 + 1, base_col + 2] += 1
        
        # Term 3
        U[4, base_col + 3] += 1
        V[1*10 + k0, base_col + 3] += 1
        V[0*10 + k0, base_col + 3] -= 1
        W[k0*2 + 0, base_col + 3] += 1
        W[k0*2 + 1, base_col + 3] += 1
        
        # Term 4
        U[0, base_col + 4] += 1
        U[1, base_col + 4] += 1
        V[1*10 + k1, base_col + 4] += 1
        W[k0*2 + 0, base_col + 4] -= 1
        W[k1*2 + 0, base_col + 4] += 1
        
        # Term 5
        U[3, base_col + 5] += 1
        U[0, base_col + 5] -= 1
        V[0*10 + k0, base_col + 5] += 1
        V[0*10 + k1, base_col + 5] += 1
        W[k1*2 + 1, base_col + 5] += 1
        
        # Term 6
        U[1, base_col + 6] += 1
        U[4, base_col + 6] -= 1
        V[1*10 + k0, base_col + 6] += 1
        V[1*10 + k1, base_col + 6] += 1
        W[k0*2 + 0, base_col + 6] += 1
        
        # Term 7
        U[2, base_col + 7] += 1
        V[2*10 + k0, base_col + 7] += 1
        W[k0*2 + 0, base_col + 7] += 1
        
        # Term 8
        U[2, base_col + 8] += 1
        V[2*10 + k1, base_col + 8] += 1
        W[k1*2 + 0, base_col + 8] += 1
        
        # Term 9
        U[5, base_col + 9] += 1
        V[2*10 + k0, base_col + 9] += 1
        W[k0*2 + 1, base_col + 9] += 1
        
        # Term 10
        U[5, base_col + 10] += 1
        V[2*10 + k1, base_col + 10] += 1
        W[k1*2 + 1, base_col + 10] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
