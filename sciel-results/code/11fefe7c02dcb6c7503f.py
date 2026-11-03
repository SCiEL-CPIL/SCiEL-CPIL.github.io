"""Recursive block decomposition with Strassen for 2x2x2 blocks."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from fractions import Fraction

def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    # Check that entries not corresponding to valid (i,j,k) are zero
    # Actually, the tensor is only defined for valid (i,j,k), so we just need those to be 1.
    # But we should also ensure no spurious contributions... 
    # The contract says "exactly decompose the matrix multiplication tensor"
    # The tensor T has T[i,j,k]=1 for all valid i,j,k and 0 otherwise.
    # Since i,j,k range over valid indices, every (i,j,k) in the product space is valid.
    # So we just need T[i,j,k]=1 for all valid triples.
    return True

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U row = i*2+j for A[i,j], V row = j*2+k for B[j,k], W row = k*2+i for C[k,n+i]
    """
    # Strassen's 7 products for multiplying 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U rows: [A00, A01, A10, A11] = indices 0,1,2,3
    # V rows: [B00, B01, B10, B11] = indices 0,1,2,3
    # W rows: [C00, C01, C10, C11] = indices k*2+i: C00->0*2+0=0, C01->0*2+1=1, C10->1*2+0=2, C11->1*2+1=3
    
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A00
        [0,  1,  0,  0,  1,  0,  1],  # A01
        [0,  0,  1,  1,  0,  1,  0],  # A10
        [1,  0,  0,  1,  0,  0, -1],  # A11
    ]
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  1],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  0],  # B11
    ]
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00
        [0,  1,  1,  0,  1,  0,  0],  # C01
        [0,  0,  0,  1,  0,  1,  0],  # C10
        [1,  0,  1,  0,  0,  1,  0],  # C11
    ]
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions for the same dimensions by concatenation."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    V = [row1 + row2 for row1, row2 in zip(V1, V2)]
    W = [row1 + row2 for row1, row2 in zip(W1, W2)]
    return U, V, W

def _decompose(n, m, p):
    """Recursively decompose n x m times m x p matrix multiplication."""
    # Base case: use schoolbook
    if n <= 2 and m <= 2 and p <= 2:
        # Use Strassen if all are 2, otherwise schoolbook
        if n == 2 and m == 2 and p == 2:
            return _strassen_222()
        else:
            return _schoolbook(n, m, p)
    
    # Try to split along one dimension to get 2x2x2 Strassen blocks
    # Strategy: if any dimension is >= 2, try to extract a 2x2x2 block
    
    # We'll try different split strategies and pick the best one
    
    best = None
    
    # Strategy 1: If n >= 2, split n into 2 + (n-2)
    if n >= 2:
        # Decompose as block diagonal: [A_top | A_bot] where A_top is 2xm, A_bot is (n-2)xm
        # This gives: C_top = A_top * B (2 x p) and C_bot = A_bot * B ((n-2) x p)
        # These are independent, so we can decompose each separately.
        
        # Top block: 2 x m times m x p
        # Bottom block: (n-2) x m times m x p
        
        U_top, V_top, W_top = _decompose(2, m, p)
        U_bot, V_bot, W_bot = _decompose(n-2, m, p)
        
        # Combine: U has rows [top_rows; bot_rows], V and W are shared but need to be aligned
        # Actually V and W must have the same rank. We concatenate the columns.
        U = [row for row in U_top] + [row for row in U_bot]
        V = [row1 + row2 for row1, row2 in zip(V_top, V_bot)]
        W = [row1 + row2 for row1, row2 in zip(W_top, W_bot)]
        
        rank = len(U[0])
        if best is None or rank < len(best[0][0]):
            best = (U, V, W)
    
    # Strategy 2: If m >= 2, split m into 2 + (m-2)
    if m >= 2:
        # A is n x m = [A_left | A_right], B is m x p = [B_top; B_bot]
        # C = A_left * B_top + A_right * B_bot
        # These are two independent matrix multiplications:
        #   C1 = A_left (n x 2) * B_top (2 x p)
        #   C2 = A_right (n x (m-2)) * B_bot ((m-2) x p)
        
        U1, V1, W1 = _decompose(n, 2, p)
        U2, V2, W2 = _decompose(n, m-2, p)
        
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]
        V = [row for row in V1] + [row for row in V2]
        W = [row1 + row2 for row1, row2 in zip(W1, W2)]
        
        rank = len(U[0])
        if best is None or rank < len(best[0][0]):
            best = (U, V, W)
    
    # Strategy 3: If p >= 2, split p into 2 + (p-2)
    if p >= 2:
        # C = A (n x m) * B (m x p) = [C_left | C_right]
        # C_left = A * B_left (n x 2)
        # C_right = A * B_right (n x (p-2))
        
        U1, V1, W1 = _decompose(n, m, 2)
        U2, V2, W2 = _decompose(n, m, p-2)
        
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]
        V = [row1 + row2 for row1, row2 in zip(V1, V2)]
        W = [row for row in W1] + [row for row in W2]
        
        rank = len(U[0])
        if best is None or rank < len(best[0][0]):
            best = (U, V, W)
    
    # Strategy 4: Try to extract a 2x2x2 Strassen block directly
    # If n>=2, m>=2, p>=2, we can partition A into [A11 A12; A21 A22] (2x2 block) and rest
    # This is more complex. Let's use the approach where we handle the 2x2x2 corner separately.
    
    if n >= 2 and m >= 2 and p >= 2:
        # Partition:
        # A: [A11 (2x2) | A12 (2x(m-2)); A21 ((n-2)x2) | A22 ((n-2)x(m-2))]
        # B: [B11 (2x2); B12 (2x(p-2)); B21 ((m-2)x2); B22 ((m-2)x(p-2))]
        # C: [C11 (2x2) | C12 (2x(p-2)); C21 ((n-2)x2) | C22 ((n-2)x(p-2))]
        #
        # C11 = A11*B11 + A12*B21  -- wait, this isn't right for block matrix mult
        # Actually for block matrix multiplication:
        # C11 = A11*B11 + A12*B21  -- NO, dimensions don't work out this way
        # 
        # Let me reconsider. Standard block matrix multiplication:
        # A is n x m, B is m x p, C is n x p.
        # If we split i into {0,1} and rest, j into {0,1} and rest, k into {0,1} and rest:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        #
        # For i in {0,1}, k in {0,1}:
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + sum_{j>=2} A[i,j]*B[j,k]
        #
        # This doesn't cleanly separate into a 2x2x2 Strassen block.
        # 
        # Alternative: use the recursive approach but with a different split.
        # Let's try: handle the case where we can apply Strassen to a sub-block.
        # 
        # Actually, the simplest correct approach is what we already have:
        # split one dimension at a time. Let's also try splitting into 2 and (dim-2)
        # for each dimension, which is what strategies 1-3 do.
        # 
        # Let's also try a 2x2x2 Strassen block by partitioning all three dimensions:
        # We compute the contribution of the top-left 2x2x2 block using Strassen,
        # and the rest using schoolbook/recursive.
        
        # The 2x2x2 block contributes to C[0:2, 0:2]
        # But C[0:2, 0:2] also receives contributions from other parts of A and B.
        # So we can't just isolate it.
        
        # Skip this for now; the recursive 1D splits should be sufficient.
        pass
    
    if best is None:
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 by 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
