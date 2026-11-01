"""Tensor decomposition for 2x2 x 2x10 matrix multiplication."""
import time
import numpy as np

_DEADLINE = time.time() + 140

def construct():
    """
    Returns a rank-35 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x10 matrix.
    
    The decomposition uses Strassen's 7-product algorithm applied
    to 5 independent 2x2 column-pair blocks of the 2x10 matrix.
    
    Returns dict with keys 'U', 'V', 'W':
      U: (4, 35) - A-side coefficients
      V: (20, 35) - B-side coefficients  
      W: (20, 35) - C-side coefficients (trace convention W[k*n+i])
    """
    if time.time() > _DEADLINE:
        return _fallback_rank35()
    
    return _build_strassen_rank35()


def _build_strassen_rank35():
    """Construct rank-35 decomposition via 5 Strassen blocks."""
    R = 35
    U = np.zeros((4, R), dtype=int)
    V = np.zeros((20, R), dtype=int)
    W = np.zeros((20, R), dtype=int)
    
    # Strassen's 7 products for 2x2 matmul
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e)
    # P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # U rows: [a, b, c, d] indexed as [0,1,2,3]
    sU = np.array([
        [1, 0, 0, 1],    # P1
        [0, 0, 1, 1],    # P2
        [1, 0, 0, 0],    # P3
        [0, 0, 0, 1],    # P4
        [1, 1, 0, 0],    # P5
        [-1, 0, 1, 0],   # P6
        [0, 1, 0, -1],   # P7
    ])
    
    # V rows: [e, f, g, h] indexed as [0,1,2,3]
    sV = np.array([
        [1, 0, 0, 1],    # P1
        [1, 0, 0, 0],    # P2
        [0, 1, 0, -1],   # P3
        [-1, 0, 1, 0],   # P4
        [0, 0, 0, 1],    # P5
        [1, 1, 0, 0],    # P6
        [0, 0, 1, 1],    # P7
    ])
    
    # W contributions to [C00, C01, C10, C11]
    sW = np.array([
        [1, 0, 0, 1],    # P1
        [0, 0, 1, -1],   # P2
        [0, 1, 0, 1],    # P3
        [1, 0, 1, 0],    # P4
        [-1, 1, 0, 0],   # P5
        [0, 0, 0, 1],    # P6
        [1, 0, 0, 0],    # P7
    ])
    
    for blk in range(5):
        for prod in range(7):
            col = blk * 7 + prod
            
            # U: A coefficients
            U[:, col] = sU[prod]
            
            # V: B coefficients for columns 2*blk and 2*blk+1
            # e = B[0, 2*blk] -> idx 0*10 + 2*blk = 2*blk
            # f = B[0, 2*blk+1] -> idx 2*blk + 1
            # g = B[1, 2*blk] -> idx 10 + 2*blk
            # h = B[1, 2*blk+1] -> idx 10 + 2*blk + 1
            V[2*blk, col] = sV[prod, 0]
            V[2*blk + 1, col] = sV[prod, 1]
            V[10 + 2*blk, col] = sV[prod, 2]
            V[10 + 2*blk + 1, col] = sV[prod, 3]
            
            # W: C coefficients, trace convention C[k*n + i] = C[2k+i]
            # C00: k=2*blk, i=0 -> idx 2*(2*blk)+0 = 4*blk
            # C01: k=2*blk+1, i=0 -> idx 2*(2*blk+1)+0 = 4*blk+2
            # C10: k=2*blk, i=1 -> idx 2*(2*blk)+1 = 4*blk+1
            # C11: k=2*blk+1, i=1 -> idx 2*(2*blk+1)+1 = 4*blk+3
            W[4*blk, col] = sW[prod, 0]
            W[4*blk + 2, col] = sW[prod, 1]
            W[4*blk + 1, col] = sW[prod, 2]
            W[4*blk + 3, col] = sW[prod, 3]
    
    return {"U": U, "V": V, "W": W}


def _fallback_rank35():
    """Fallback: same rank-35 construction, in case of timeout."""
    R = 35
    U = np.zeros((4, R), dtype=int)
    V = np.zeros((20, R), dtype=int)
    W = np.zeros((20, R), dtype=int)
    
    sU = np.array([
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ])
    sV = np.array([
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ])
    sW = np.array([
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ])
    
    for blk in range(5):
        for prod in range(7):
            col = blk * 7 + prod
            U[:, col] = sU[prod]
            V[2*blk, col] = sV[prod, 0]
            V[2*blk + 1, col] = sV[prod, 1]
            V[10 + 2*blk, col] = sV[prod, 2]
            V[10 + 2*blk + 1, col] = sV[prod, 3]
            W[4*blk, col] = sW[prod, 0]
            W[4*blk + 2, col] = sW[prod, 1]
            W[4*blk + 1, col] = sW[prod, 2]
            W[4*blk + 3, col] = sW[prod, 3]
    
    return {"U": U, "V": V, "W": W}
