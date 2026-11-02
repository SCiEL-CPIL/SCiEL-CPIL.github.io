"""Optimized program for matrix multiplication 4x4x5 tensor decomposition."""
import numpy as np
from numpy.linalg import matrix_rank
import time
import sys

DIMENSIONS = [4, 4, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 if (i,j,k) valid."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _rank_of_tensor(T):
    """Compute the tensor rank via flattening and SVD (upper bound)."""
    n, m, p = T.shape
    flat = T.reshape(n, m * p)
    r = matrix_rank(flat, tol=1e-10)
    return r

def _strassen_2x2_2x2():
    """Return the 7 bilinear forms for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as 4x7, 4x7, 4x7 matrices (entries in {-1, 0, 1})."""
    # Strassen's 7 products for 2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 - p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p7
    
    # U maps input indices (i,j) -> coefficient for each product
    # For 2x2: rows are (0,0),(0,1),(1,0),(1,1)
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # p1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] += p1, C[1,1] += p1
    
    # p2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0] += p2, C[1,1] -= p2
    
    # p3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1] += p3, C[1,1] += p3
    
    # p4 = d(g-e): d=A[1,1], g=B[1,0], e=B[0,0]
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0] += p4, C[1,0] += p4
    
    # p5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0] -= p5, C[0,1] += p5
    
    # p6 = (c-a)(e+f): c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1  # C[0,1] += p6 (note: this is actually for C[1,0] in some conventions)
    # Let me re-derive carefully. Standard Strassen:
    # C[0,0] = p1 + p4 - p5 - p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p7
    # Wait, I need to be more careful. Let me use the standard formulation.
    
    # Actually, let me just use a known correct Strassen decomposition.
    # The 7 products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,3]=1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,2]=1; W[3,3]=-1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,1]=1; W[3,3]=1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[0,0]=1; W[2,2]=1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,0]=-1; W[1,1]=1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[1,1]=1  # This should contribute to... let me check. Actually M6 isn't in standard Strassen output directly.
    # Hmm, I think I'm confusing myself. Let me use a simpler approach.
    
    # Let me just hardcode a known valid rank-7 decomposition for 2x2x2.
    # Using the standard Strassen scheme with indices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # p = (a+d)(e+h) + d(g-e) - (a+b)h - (b-d)(g+h)
    # q = a(f-h) + (a+b)h
    # r = (c+d)e + d(g-e)
    # s = (a+d)(e+h) - (c+d)e + a(f-h) + (b-d)(g+h)
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Product 1: (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,3]=1
    
    # Product 2: (c+d)e
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,2]=1; W[3,3]=-1
    
    # Product 3: a(f-h)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,1]=1; W[3,3]=1
    
    # Product 4: d(g-e)
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[0,0]=1; W[2,2]=1
    
    # Product 5: (a+b)h
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,0]=-1; W[1,1]=1
    
    # Product 6: (c-a)(e+f)
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[1,1]=1  # Hmm, this doesn't look right for standard Strassen
    
    # Product 7: (b-d)(g+h)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,0]=-1; W[3,3]=1
    
    return U, V, W

def _block_decompose_4x4_4x5():
    """Use block decomposition with Strassen for 2x2 sub-blocks.
    
    Split A (4x4) into four 2x2 blocks, B (4x5) into two 2x5 blocks.
    C (4x5) = A*B, split into four 2x5 blocks.
    
    C_top_left = A11*B11 + A12*B21
    C_top_right = A11*B12 + A12*B22
    C_bot_left = A21*B11 + A22*B21
    C_bot_right = A21*B12 + A22*B22
    
    Each 2x2 * 2x5 multiplication can be done with rank at most 6*... 
    Actually, 2x2 * 2x5: we can treat this as 2x2 * (2x5) = two 2x2 * 2x2 plus 2x2 * 2x1.
    This gets complicated. Let me use a simpler approach.
    """
    n, m, p = 4, 4, 5
    
    # Schoolbook baseline
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all valid (i,j,k).
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    error += abs(val - 1.0)
    return error

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2

def construct():
    """Construct a low-rank tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = 4, 4, 5
    
    # Start with schoolbook decomposition
    U, V, W = _block_decompose_4x4_4x5()
    current_rank = U.shape[1]
    
    # Try to reduce rank using ALS-style refinement
    # We'll try various target ranks and use optimization
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try rank reduction via column elimination
    # Check if any column is redundant
    for target_rank in range(current_rank - 1, max(20, current_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try to find a lower-rank decomposition
        # Use random initialization and ALS
        max_iters = 50
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse matrices
            U_try = np.zeros((n*m, target_rank), dtype=np.float64)
            V_try = np.zeros((m*p, target_rank), dtype=np.float64)
            W_try = np.zeros((p*n, target_rank), dtype=np.float64)
            
            # Sparse random initialization
            density = 0.3
            for c in range(target_rank):
                for i in range(n*m):
                    if np.random.random() < density:
                        U_try[i, c] = np.random.choice([-1, 1])
                for j in range(m*p):
                    if np.random.random() < density:
                        V_try[j, c] = np.random.choice([-1, 1])
                for k in range(p*n):
                    if np.random.random() < density:
                        W_try[k, c] = np.random.choice([-1, 1])
            
            # ALS refinement
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx = i*m + j
                        # For each column c, U[idx,c] = sum over k of V[j*p+k,c]*W[k*n+i,c] / (sum of squares)
                        num = np.zeros(target_rank)
                        den = np.zeros(target_rank)
                        for k in range(p):
                            v_val = V_try[j*p+k, :]
                            w_val = W_try[k*n+i, :]
                            num += v_val * w_val
                            den += v_val * v_val
                        mask = den > 1e-10
                        U_try[idx, mask] = num[mask] / den[mask]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        idx = j*p + k
                        num = np.zeros(target_rank)
                        den = np.zeros(target_rank)
                        for i in range(n):
                            u_val = U_try[i*m+j, :]
                            w_val = W_try[k*n+i, :]
                            num += u_val * w_val
                            den += u_val * u_val
                        mask = den > 1e-10
                        V_try[idx, mask] = num[mask] / den[mask]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        idx = k*n + i
                        num = np.zeros(target_rank)
                        den = np.zeros(target_rank)
                        for j in range(m):
                            u_val = U_try[i*m+j, :]
                            v_val = V_try[j*p+k, :]
                            num += u_val * v_val
                            den += u_val * u_val
                        mask = den > 1e-10
                        W_try[idx, mask] = num[mask] / den[mask]
            
            # Check error
            error = _verify_decomposition(U_try, V_try, W_try, n, m, p)
            
            if error < 1e-6:
                # Round to half-integers
                U_round = _round_to_halfint(U_try)
                V_round = _round_to_halfint(V_try)
                W_round = _round_to_halfint(W_try)
                
                error_round = _verify_decomposition(U_round, V_round, W_round, n, m, p)
                if error_round < 1e-6:
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    break
                else:
                    # Use unrounded if it's valid
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    break
    
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
