"""Tensor rank minimization for 4x4x5 matrix multiplication via structured Strassen-based search."""
DIMENSIONS = [4, 4, 5]
import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check exact MM decomposition with trace convention W[k*n+i]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    # Reconstruct and check
    for t in range(rank):
        T -= np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    return np.max(np.abs(T)) < 1e-9

def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen's 7 products:
    p1 = (a+d)(e+h)
    p2 = (a+b)e  -> wait, let me use the standard form:
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c+d)g
    p7 = (a-c)(e+f)
    
    C11 = p1 + p4 - p5 + p7
    C12 = p3 + p5
    C21 = p2 + p6
    C22 = p1 - p2 + p3 + p6
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices (trace convention k*n+i): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Wait: W[k*n+i] so for n=2: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # p1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W contributes to C11,C22
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p6
    # C22 = p1 - p2 + p3 + p6
    
    # p1: U[0]=1, U[3]=1; V[0]=1, V[3]=1; W[0]=1, W[3]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # p2 = (c+d)e: U[2]=1, U[3]=1; V[0]=1; W[2]=1 (C21)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    
    # p3 = a(f-h): U[0]=1; V[1]=1, V[3]=-1; W[1]=1, W[3]=1 (C12, C22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 and C22
    
    # p4 = d(g-e): U[3]=1; V[2]=1, V[0]=-1; W[0]=1 (C11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    
    # p5 = (a+b)h: U[0]=1, U[1]=1; V[3]=1; W[0]=-1, W[1]=1 (C11 neg, C12)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 gets -p5, C12 gets +p5
    
    # p6 = (c+d)g: U[2]=1, U[3]=1; V[2]=1; W[2]=1, W[3]=1 (C21, C22)
    U[2, 5] = 1; U[3, 5] = 1
    V[2, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1  # C21 and C22
    
    # p7 = (a-c)(e+f): U[0]=1, U[2]=-1; V[0]=1, V[1]=1; W[0]=1 (C11)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C11
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _block_decompose_4x4x4():
    """
    Decompose 4x4x4 MM using 2x2 block Strassen.
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each 2x2x2 has rank 7. We need to combine these.
    For C11: 7 terms for A11*B11 + 7 terms for A12*B21 = 14 terms
    Similarly for others. Total = 4*14 = 56 terms? No, that's not right either.
    
    Actually, the standard approach: each block multiplication is independent.
    C11 needs rank 7 for A11*B11 plus rank 7 for A12*B21, but these are separate
    products so we just concatenate: 14 terms.
    Total for 4x4x4: 4 blocks * 14 = 56 terms.
    
    But wait, we can do better by using the full Strassen structure on 4x4 directly,
    which gives rank 49. Let me implement that.
    
    For now, let's use the 56-term block decomposition.
    """
    n = m = p = 4
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block structure for 4x4:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # For each block multiplication, we use Strassen's 7 products.
    # We'll create a helper to add a 2x2x2 Strassen decomposition with offsets.
    
    def add_strassen_block(U, V, W, rank_start, i_off, j_off, k_off, a_i, a_j, b_i, b_j, c_i, c_j):
        """Add Strassen 2x2x2 terms for A[a_i:a_i+2, a_j:a_j+2] * B[b_i:b_i+2, b_j:b_j+2] -> C[c_i:c_i+2, c_j:c_j+2]"""
        # A indices in flat form: (a_i+r)*m + (a_j+s) for r,s in 0,1
        # B indices in flat form: (b_i+r)*p + (b_j+s) for r,s in 0,1
        # C indices in flat form (trace convention k*n+i): (c_j+s)*n + (c_i+r) for r,s in 0,1
        
        for t in range(7):
            col = rank_start + t
            # Get the Strassen pattern for product t
            # p1: U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=1
            # p2: U[2]=1,U[3]=1; V[0]=1; W[2]=1
            # p3: U[0]=1; V[1]=1,V[3]=-1; W[1]=1,W[3]=1
            # p4: U[3]=1; V[2]=1,V[0]=-1; W[0]=1
            # p5: U[0]=1,U[1]=1; V[3]=1; W[0]=-1,W[1]=1
            # p6: U[2]=1,U[3]=1; V[2]=1; W[2]=1,W[3]=1
            # p7: U[0]=1,U[2]=-1; V[0]=1,V[1]=1; W[0]=1
            
            if t == 0:  # p1
                U[(a_i)*m + a_j, col] = 1
                U[(a_i+1)*m + (a_j+1), col] = 1
                V[(b_i)*p + b_j, col] = 1
                V[(b_i+1)*p + (b_j+1), col] = 1
                W[(c_j)*n + c_i, col] = 1
                W[(c_j+1)*n + (c_i+1), col] = 1
            elif t == 1:  # p2
                U[(a_i+1)*m + a_j, col] = 1
                U[(a_i+1)*m + (a_j+1), col] = 1
                V[(b_i)*p + b_j, col] = 1
                W[(c_j)*n + (c_i+1), col] = 1
            elif t == 2:  # p3
                U[(a_i)*m + a_j, col] = 1
                V[(b_i)*p + (b_j+1), col] = 1
                V[(b_i+1)*p + (b_j+1), col] = -1
                W[(c_j)*n + (c_i+1), col] = 1
                W[(c_j+1)*n + (c_i+1), col] = 1
            elif t == 3:  # p4
                U[(a_i+1)*m + (a_j+1), col] = 1
                V[(b_i+1)*p + b_j, col] = 1
                V[(b_i)*p + b_j, col] = -1
                W[(c_j)*n + c_i, col] = 1
            elif t == 4:  # p5
                U[(a_i)*m + a_j, col] = 1
                U[(a_i)*m + (a_j+1), col] = 1
                V[(b_i+1)*p + (b_j+1), col] = 1
                W[(c_j)*n + c_i, col] = -1
                W[(c_j)*n + (c_i+1), col] = 1
            elif t == 5:  # p6
                U[(a_i+1)*m + a_j, col] = 1
                U[(a_i+1)*m + (a_j+1), col] = 1
                V[(b_i+1)*p + b_j, col] = 1
                W[(c_j)*n + (c_i+1), col] = 1
                W[(c_j+1)*n + (c_i+1), col] = 1
            elif t == 6:  # p7
                U[(a_i)*m + a_j, col] = 1
                U[(a_i+1)*m + a_j, col] = -1
                V[(b_i)*p + b_j, col] = 1
                V[(b_i)*p + (b_j+1), col] = 1
                W[(c_j)*n + c_i, col] = 1
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: a_i=0, a_j=0, b_i=0, b_j=0, c_i=0, c_j=0
    add_strassen_block(U, V, W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # A12*B21: a_i=0, a_j=2, b_i=2, b_j=0, c_i=0, c_j=0
    add_strassen_block(U, V, W, 7, 0, 2, 0, 0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: a_i=0, a_j=0, b_i=0, b_j=2, c_i=0, c_j=2
    add_strassen_block(U, V, W, 14, 0, 0, 2, 0, 0, 0, 2, 0, 2)
    # A12*B22: a_i=0, a_j=2, b_i=2, b_j=2, c_i=0, c_j=2
    add_strassen_block(U, V, W, 21, 0, 2, 2, 0, 2, 2, 2, 0, 2)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: a_i=2, a_j=0, b_i=0, b_j=0, c_i=2, c_j=0
    add_strassen_block(U, V, W, 28, 2, 0, 0, 2, 0, 0, 0, 2, 0)
    # A22*B21: a_i=2, a_j=2, b_i=2, b_j=0, c_i=2, c_j=0
    add_strassen_block(U, V, W, 35, 2, 2, 0, 2, 2, 2, 0, 2, 0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: a_i=2, a_j=0, b_i=0, b_j=2, c_i=2, c_j=2
    add_strassen_block(U, V, W, 42, 2, 0, 2, 2, 0, 0, 2, 2, 2)
    # A22*B22: a_i=2, a_j=2, b_i=2, b_j=2, c_i=2, c_j=2
    add_strassen_block(U, V, W, 49, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _extend_to_4x4x5(U444, V444, W444):
    """
    Extend a 4x4x4 decomposition to 4x4x5 by adding 16 terms for the 5th column.
    The 5th column of C: C[i,4] = sum_j A[i,j] * B[j,4]
    This is a 4x4x1 multiplication, which has rank 16 (schoolbook).
    """
    n = m = 4
    p_old = 4
    p_new = 5
    rank_old = U444.shape[1]
    rank_new = rank_old + 16
    
    U = np.zeros((n*m, rank_new))
    V = np.zeros((m*p_new, rank_new))
    W = np.zeros((p_new*n, rank_new))
    
    # Copy the 4x4x4 part
    U[:, :rank_old] = U444
    V[:, :rank_old] = V444
    W[:, :rank_old] = W444
    
    # Add 16 terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            col = rank_old + i * m + j
            U[i * m + j, col] = 1
            V[j * p_new + 4, col] = 1  # B[j,4]
            W[4 * n + i, col] = 1  # C[i,4] with trace convention: k*n+i = 4*4+i
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Strategy: Use block Strassen for 4x4x4 (rank 56) + 16 terms for 5th column = rank 72
    # This should be valid and better than schoolbook rank 80.
    
    # First, get the 4x4x4 block Strassen decomposition
    dec444 = _block_decompose_4x4x4()
    U444 = np.array(dec444["U"])
    V444 = np.array(dec444["V"])
    W444 = np.array(dec444["W"])
    
    # Verify 4x4x4
    if not _verify(U444, V444, W444, 4, 4, 4):
        # Fall back to schoolbook for 4x4x4
        sb444 = _schoolbook()
        # Need to extract just the 4x4x4 part from schoolbook 4x4x5
        # Actually, let's just use schoolbook for everything
        sb = _schoolbook()
        if _verify(sb["U"], sb["V"], sb["W"], n, m, p):
            return sb
        raise ValueError("Schoolbook verification failed")
    
    # Extend to 4x4x5
    dec445 = _extend_to_4x4x5(U444, V444, W444)
    
    # Verify 4x4x5
    U445 = np.array(dec445["U"])
    V445 = np.array(dec445["V"])
    W445 = np.array(dec445["W"])
    
    if _verify(U445, V445, W445, n, m, p):
        return dec445
    
    # If verification fails, fall back to schoolbook
    sb = _schoolbook()
    if _verify(sb["U"], sb["V"], sb["W"], n, m, p):
        return sb
    
    raise ValueError("No valid decomposition found")
# EVOLVE-BLOCK-END
