import numpy as np
import time

def construct():
    """
    Returns U, V, W for a 3x3x8 matrix multiplication decomposition.
    U shape (9, rank), V shape (24, rank), W shape (24, rank).
    Convention: A is 3x3 (indices i,k -> i*3+k), B is 3x8 (indices k,l -> k*8+l), C is 3x8 (indices i,l -> l*3+i).
    """
    n = 3
    m = 3
    p = 8
    
    U = np.zeros((n*m, 72))
    V = np.zeros((m*p, 72))
    W = np.zeros((p*n, 72))
    
    col = 0
    for i in range(n):
        for k in range(m):
            for l in range(p):
                # A index: i*m + k
                u_idx = i * m + k
                # B index: k*p + l
                v_idx = k * p + l
                # C index: l*n + i (based on task description W[k*n+i] where first index is column l of B/row of C? 
                # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
                # Usually C is n x p. C[i, l] = sum_k A[i,k] B[k,l].
                # If W is indexed by [row, col] of C, and C is 3x8, row is i (0-2), col is l (0-7).
                # But the prompt says "W[k*n+i,r]". This notation is slightly ambiguous.
                # Let's look at standard conventions for these tasks.
                # Often W is size (p*n, rank).
                # If W index is k*n + i, then the first part is k (0..p-1) and second is i (0..n-1).
                # This corresponds to C[i, k] if we transpose? Or C[k, i]?
                # Let's assume the standard flattening for the output matrix C of size n x p.
                # Usually the output tensor is defined such that sum_r U[i,k,r] V[k,l,r] W[i,l,r] = C[i,l].
                # If W has shape (p*n, rank), and the index is w, then w = l*n + i or i*p + l?
                # The prompt says "W[k*n+i,r]". Here 'k' is likely the row index of the output (0..p-1?) and 'i' is the column?
                # Wait, B is 3x8. A is 3x3. C is 3x8.
                # C has 3 rows, 8 columns.
                # If the index is k*n + i, and n=3, then k goes 0..7 (p-1) and i goes 0..2 (n-1).
                # So the index is l*n + i. This implies the output is stored column-major with respect to l (the column of C).
                # So W index = l * n + i.
                
                w_idx = l * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {'U': U, 'V': V, 'W': W}
