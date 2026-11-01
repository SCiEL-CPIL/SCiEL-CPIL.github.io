"""Matrix multiplication 2x2 by 2x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 4]

# Deadline guard: the evaluator kills after 150s; keep a safety margin.
_START = time.time()
_DEADLINE = 135.0  # seconds

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta for correct entries.
    Here the multiplication is A (n x m) * B (m x p) = C (n x p).
    The tensor is T[i,j,k] = delta_{i,j,k} in the sense that C[i,k] = sum_j A[i,j] B[j,k].
    The decomposition requires: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triple? 
    Actually the multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The rank-r decomposition means T[i,j,k] = sum_{r=1}^r U[i,j,r] V[j,k,r] W[k,i,r].
    """
    n, m, p = n, m, p
    rank = len(U[0])
    # Build the target tensor
    # T[i,j,k] = 1 for all valid i,j,k
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n=2, m=2, p=4):
    """Exact verification using fractions."""
    n, m, p = n, m, p
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook rank-16 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 16
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
    return {"U": U, "V": V, "W": W}

def _try_strassen_style():
    """
    Try to find a rank < 16 decomposition using structured search.
    
    For 2x2 * 2x4, we can think of it as two 2x2 * 2x2 multiplications side by side.
    Each 2x2*2x2 has rank at least 7 (Strassen). So the combined rank is at least 7+7=14?
    But they share the middle dimension j, so there might be some sharing.
    
    Actually, the tensor for 2x2*2x4 is a sum of two 2x2*2x2 tensors (one for k=0,1 and one for k=2,3).
    The rank of a sum is at most the sum of ranks, but could be less if there's sharing.
    
    Let's try rank 14 first, then 13, etc.
    """
    n, m, p = 2, 2, 4
    
    # Try ranks from 13 up to 15 (16 is schoolbook)
    for target_rank in [13, 14, 15]:
        if _time_left() < 5.0:
            break
        if _try_rank(n, m, p, target_rank):
            return True
    return False

def _try_rank(n, m, p, target_rank):
    """Try to find a rank-target_rank decomposition using randomized search with local optimization."""
    if _time_left() < 3.0:
        return False
    
    # We'll use a simple approach: start from schoolbook and try to merge columns
    # Or use random initialization and gradient-free search
    
    # For now, let's try a structured approach:
    # The 2x2*2x4 multiplication can be seen as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # This is two independent 2x2*2x2 multiplications (for k=0,1 and k=2,3).
    # Each has rank >= 7, so total rank >= 14.
    # Can we achieve rank 14? If the two 2x2*2x2 multiplications are independent, rank = 7+7 = 14.
    # But they share the A matrix indices, so maybe we can do better?
    # Actually no - the B indices are different (k=0,1 vs k=2,3), so the tensors are on disjoint supports in the k dimension.
    # The rank of a direct sum is the sum of ranks. So rank >= 14.
    # And rank 14 is achievable by using Strassen on each 2x2*2x2 block.
    
    if target_rank >= 14:
        # Try to construct rank 14 using Strassen on each 2x2*2x2 block
        result = _strassen_2x2x2x4()
        if result is not None:
            return result
        return False
    else:
        # rank < 14 is theoretically impossible (rank >= 14)
        return False

def _strassen_2x2x2x4():
    """
    Construct a rank-14 decomposition by applying Strassen's algorithm to each 2x2*2x2 block.
    
    For the first block (k=0,1): multiply A (2x2) by B1 (2x2) where B1 = B[:, 0:2]
    For the second block (k=2,3): multiply A (2x2) by B2 (2x2) where B2 = B[:, 2:4]
    
    Strassen's algorithm for 2x2*2x2 uses 7 multiplications.
    We apply it separately to each block, giving 7+7=14 total.
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen's 7 products for 2x2*2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e, b*f], [c*g, d*h]] + ... no, Strassen computes:
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For block 1 (k=0,1): B1 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # For block 2 (k=2,3): B2 = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
    
    # A = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # Indices: A[i,j] -> U[i*m+j] = U[i*2+j]
    # B[j,k] -> V[j*p+k] = V[j*4+k]
    # C[i,k] -> W[k*n+i] = W[k*2+i]
    
    # Block 1: k=0,1
    # B1[0,0] = B[0,0] -> V[0*4+0] = V[0]
    # B1[0,1] = B[0,1] -> V[0*4+1] = V[1]
    # B1[1,0] = B[1,0] -> V[1*4+0] = V[4]
    # B1[1,1] = B[1,1] -> V[1*4+1] = V[5]
    
    # Block 2: k=2,3
    # B2[0,0] = B[0,2] -> V[0*4+2] = V[2]
    # B2[0,1] = B[0,3] -> V[0*4+3] = V[3]
    # B2[1,0] = B[1,2] -> V[1*4+2] = V[6]
    # B2[1,1] = B[1,3] -> V[1*4+3] = V[7]
    
    # A indices:
    # A[0,0] -> U[0], A[0,1] -> U[1]
    # A[1,0] -> U[2], A[1,1] -> U[3]
    
    # C indices for block 1:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # C indices for block 2:
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    
    # Now fill in the 14 columns (7 for block 1, 7 for block 2)
    
    # Block 1 (columns 0-6):
    # M1 = (A[0,0]+A[1,1]) * (B1[0,0]+B1[1,1])
    # U: A[0,0]=1, A[1,1]=1 -> U[0][0]=1, U[3][0]=1
    # V: B1[0,0]=1, B1[1,1]=1 -> V[0][0]=1, V[5][0]=1
    # W: C[0,0] += M1, C[0,1] += 0, C[1,0] += 0, C[1,1] += M1
    # So W[0][0]=1, W[3][0]=1
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A[1,0]+A[1,1]) * B1[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1  # C[1,0] += M2, C[1,1] -= M2
    
    # M3 = A[0,0] * (B1[0,1]-B1[1,1])
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C[0,1] += M3, C[1,1] += M3
    
    # M4 = A[1,1] * (B1[1,0]-B1[0,0])
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C[0,0] += M4, C[1,0] += M4
    
    # M5 = (A[0,0]+A[0,1]) * B1[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C[0,0] -= M5, C[0,1] += M5
    
    # M6 = (A[1,0]-A[0,0]) * (B1[0,0]+B1[0,1])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C[1,1] += M6
    
    # M7 = (A[0,1]-A[1,1]) * (B1[1,0]+B1[1,1])
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1; W[2][6] = -1  # C[0,0] += M7, C[0,1] -= M7
    
    # Block 2 (columns 7-13):
    # M1' = (A[0,0]+A[1,1]) * (B2[0,0]+B2[1,1])
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1  # C[0,2] += M1', C[1,3] += M1'
    
    # M2' = (A[1,0]+A[1,1]) * B2[0,0]
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1  # C[1,2] += M2', C[1,3] -= M2'
    
    # M3' = A[0,0] * (B2[0,1]-B2[1,1])
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1  # C[0,3] += M3', C[1,3] += M3'
    
    # M4' = A[1,1] * (B2[1,0]-B2[0,0])
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1  # C[0,2] += M4', C[1,2] += M4'
    
    # M5' = (A[0,0]+A[0,1]) * B2[1,1]
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1  # C[0,2] -= M5', C[0,3] += M5'
    
    # M6' = (A[1,0]-A[0,0]) * (B2[0,0]+B2[0,1])
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1  # C[1,3] += M6'
    
    # M7' = (A[0,1]-A[1,1]) * (B2[1,0]+B2[1,1])
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1; W[6][13] = -1  # C[0,2] += M7', C[0,3] -= M7'
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_fraction(U, V, W):
        return result
    else:
        # Debug: print which entries fail
        n, m, p = 2, 2, 4
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(14):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        print(f"FAIL: i={i}, j={j}, k={k}, sum={s}")
        return None

def construct():
    """
    Return the rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
    """
    # First try to get rank 14 using Strassen
    result = _strassen_2x2x2x4()
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 16
    return _schoolbook()
