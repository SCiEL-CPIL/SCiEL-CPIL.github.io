"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    # Check that the tensor decomposition is exact
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    # Check orthogonality: for non-matching indices, sum should be 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            for kp2 in range(p):
                                if i == ip and j == jp and k == kp:
                                    continue
                                s = 0
                                for r in range(rank):
                                    s += U[i * m + j][r] * V[jp * p + k][r] * W[kp2 * n + ip][r]
                                if abs(s) > 1e-9:
                                    return False
    return True

def _is_half_integer(val):
    return abs(val * 2 - round(val * 2)) < 1e-9

def _snap_to_half_integer(val):
    return round(val * 2) / 2.0

def _winograd_2x2_block(n, m, p, i0, j0, k0, U, V, W, offset):
    """Apply Winograd 2x2 scheme to block starting at (i0, j0, k0).
    
    For 2x2 matrix multiplication:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = A*B where C[i,j] = sum_k A[i,k]*B[k,j]
    
    Winograd uses 7 multiplications:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b21-b11)
    m4 = a22(b22-b12)
    m5 = (a11+a12)b12
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    C11 = m1 + m3 - m4 - m6
    C12 = m5 - m3 + m6 - m7
    C21 = m2 + m4 - m5 + m7
    C22 = m1 - m2 + m4 + m5
    """
    # We need at least 2x2 block in each dimension
    if i0 + 2 > n or j0 + 2 > m or k0 + 2 > p:
        return offset
    
    # The 7 Winograd terms
    # Each term corresponds to a rank-1 component
    # U indices: (i, j) where i in {i0, i0+1}, j in {j0, j0+1}
    # V indices: (j, k) where j in {j0, j0+1}, k in {k0, k0+1}
    # W indices: (k, i) where k in {k0, k0+1}, i in {i0, i0+1}
    
    winograd_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # m1 = (a11+a22)(b11+b22)
        # U: a11=1, a22=1 => U[i0,j0]=1, U[i0+1,j0+1]=1
        # V: b11=1, b22=1 => V[j0,k0]=1, V[j0+1,k0+1]=1
        # W: c11=1, c22=1 => W[k0,i0]=1, W[k0+1,i0+1]=1
        ([((i0, j0), 1), ((i0+1, j0+1), 1)], 
         [((j0, k0), 1), ((j0+1, k0+1), 1)], 
         [((k0, i0), 1), ((k0+1, i0+1), 1)]),
        # m2 = (a21+a22)b11
        # U: a21=1, a22=1 => U[i0+1,j0]=1, U[i0+1,j0+1]=1
        # V: b11=1 => V[j0,k0]=1
        # W: c21=1 => W[k0,i0+1]=1
        ([((i0+1, j0), 1), ((i0+1, j0+1), 1)], 
         [((j0, k0), 1)], 
         [((k0, i0+1), 1)]),
        # m3 = a11(b21-b11)
        # U: a11=1 => U[i0,j0]=1
        # V: b21=1, b11=-1 => V[j0+1,k0]=1, V[j0,k0]=-1
        # W: c11=1 => W[k0,i0]=1
        ([((i0, j0), 1)], 
         [((j0+1, k0), 1), ((j0, k0), -1)], 
         [((k0, i0), 1)]),
        # m4 = a22(b22-b12)
        # U: a22=1 => U[i0+1,j0+1]=1
        # V: b22=1, b12=-1 => V[j0+1,k0+1]=1, V[j0,k0+1]=-1
        # W: c22=1 => W[k0+1,i0+1]=1
        ([((i0+1, j0+1), 1)], 
         [((j0+1, k0+1), 1), ((j0, k0+1), -1)], 
         [((k0+1, i0+1), 1)]),
        # m5 = (a11+a12)b12
        # U: a11=1, a12=1 => U[i0,j0]=1, U[i0,j0+1]=1
        # V: b12=1 => V[j0,k0+1]=1
        # W: c12=1 => W[k0+1,i0]=1
        ([((i0, j0), 1), ((i0, j0+1), 1)], 
         [((j0, k0+1), 1)], 
         [((k0+1, i0), 1)]),
        # m6 = (a21-a11)(b11+b12)
        # U: a21=1, a11=-1 => U[i0+1,j0]=1, U[i0,j0]=-1
        # V: b11=1, b12=1 => V[j0,k0]=1, V[j0,k0+1]=1
        # W: c21=1 => W[k0,i0+1]=1
        ([((i0+1, j0), 1), ((i0, j0), -1)], 
         [((j0, k0), 1), ((j0, k0+1), 1)], 
         [((k0, i0+1), 1)]),
        # m7 = (a12-a22)(b21+b22)
        # U: a12=1, a22=-1 => U[i0,j0+1]=1, U[i0+1,j0+1]=-1
        # V: b21=1, b22=1 => V[j0+1,k0]=1, V[j0+1,k0+1]=1
        # W: c12=1 => W[k0+1,i0]=1
        ([((i0, j0+1), 1), ((i0+1, j0+1), -1)], 
         [((j0+1, k0), 1), ((j0+1, k0+1), 1)], 
         [((k0+1, i0), 1)]),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in winograd_terms:
        for (u_idx, u_val) in u_coeffs:
            U[u_idx[0] * m + u_idx[1]][offset] += u_val
        for (v_idx, v_val) in v_coeffs:
            V[v_idx[0] * p + v_idx[1]][offset] += v_val
        for (w_idx, w_val) in w_coeffs:
            W[w_idx[0] * n + w_idx[1]][offset] += w_val
        offset += 1
    
    return offset

def _winograd_decomposition(n, m, p):
    """Build a Winograd-based decomposition for n x m times m x p matrix mult."""
    # Start with schoolbook rank
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    # Try to apply Winograd to overlapping 2x2 blocks
    # For 3x4 x 4x4, we can try blocks at various positions
    
    # Process all possible 2x2 blocks
    for i0 in range(0, n - 1):
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                offset = _winograd_2x2_block(n, m, p, i0, j0, k0, U, V, W, offset)
    
    # Fill remaining with schoolbook for uncovered positions
    # Actually, let's just use schoolbook for the full decomposition
    # and see if Winograd helps
    
    return {"U": U, "V": V, "W": W}, offset

def construct():
    deadline = time.time() + 110  # 110s deadline, 10s safety margin
    n, m, p = DIMENSIONS
    
    # Return schoolbook decomposition (rank 48)
    result = _schoolbook(n, m, p)
    return result
# EVOLVE-BLOCK-END
