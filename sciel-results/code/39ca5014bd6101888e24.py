import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if len(U) > 0 else 0
    if rank == 0:
        return True
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    # Reshape to check all entries
    # Build target
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    # Compute decomposition
    # T_decomp[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    # This is a tensor of order 3, rank decomposition
    # Efficient: for each c, add outer product
    T_decomp = np.zeros((nm, mp, pn))
    for c in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    return np.allclose(T_decomp, target, atol=1e-9)

def _strassen_4x4():
    """
    Build a rank-7 decomposition of 4x4x4 matrix multiplication using Strassen.
    Returns (U, V, W) where U is (16,7), V is (16,7), W is (16,7).
    T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c] for u=i*4+j, v=j*4+k, w=k*4+i
    """
    # Strassen for 2x2: 7 products
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, split into 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # Each Ci = sum of products of 2x2 submatrices
    
    # The 4x4x4 tensor can be decomposed using 4*7=28 terms if we do Strassen on each block pair
    # But we can do better: use the structure more carefully.
    
    # Actually, for 4x4x4, the known minimal rank is 19.
    # Let's build a concrete rank-19 decomposition.
    
    # Strategy: Use the fact that 4x4 = 2x2 blocks.
    # The multiplication tensor for 4x4x4 has rank at most 2*7*2 + ... 
    # Let me use a different approach: build from known rank-19 decomposition.
    
    # I'll construct it using the "lattice method" or known algorithms.
    # For now, let's build a valid (possibly non-minimal) decomposition and then optimize.
    
    # Simple approach: 4x4 multiplication = sum over i,j,k of A[i,j]*B[j,k]*e[i,k]
    # That's rank 64 (schoolbook).
    
    # Better: Use Strassen-like recursion.
    # 4x4x4: split into 2x2 blocks, apply Strassen to get 7 "super-products"
    # Each super-product is a 2x2*2x2 multiplication = rank 4 (schoolbook) or rank 7 (Strassen)
    # Using schoolbook for inner: 7*4 = 28
    # Using Strassen for inner: 7*7 = 49 (worse)
    # So 28 is better for this approach.
    
    # Even better: there are known rank-19 decompositions for 4x4x4.
    # Let me try to construct one.
    
    # Actually, let me just build a rank-28 decomposition using Strassen + schoolbook inner,
    # then try to reduce it.
    
    # Strassen 4x4x4 with schoolbook inner products:
    # 7 Strassen products, each requiring 4 scalar multiplications = 28 total
    
    # Let me define the indices:
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x4: B[j,k] for j,k in 0..3
    # C is 4x4: C[i,k] for i,k in 0..3
    
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # Strassen products (each is a 2x2 matrix product):
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pm is a 2x2 matrix = sum of 4 scalar products
    # So total 7*4 = 28 scalar products
    
    # Let me build the U, V, W matrices for this rank-28 decomposition.
    # U has 16 rows (4*4), V has 16 rows, W has 16 rows, 28 columns.
    
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(16)]
    W = [[0]*28 for _ in range(16)]
    
    col = 0
    for s in range(7):  # 7 Strassen products
        for a_i in range(2):  # 2x2 inner
            for a_j in range(2):
                for b_j in range(2):
                    for b_k in range(2):
                        # This scalar product contributes to P_s[a_i, b_k]
                        # Which A element? Depends on s and (a_i, a_j)
                        # Which B element? Depends on s and (b_j, b_k)
                        
                        # A index in 4x4:
                        # For P1: (A11+A22) -> A[(a_i + 0 or 2), (a_j + 0 or 2)]
                        # Let me define the A and B contributions for each Strassen product
                        
                        # A contributions (row, col in 4x4):
                        if s == 0:  # P1 = (A11+A22)(B11+B22)
                            a_row = a_i  # 0 or 1
                            a_col = a_j  # 0 or 1
                            # Also A22 part: row = a_i+2, col = a_j+2
                            # We need two terms, but in a rank-1 decomposition each term is one product
                            # So this approach doesn't directly work for rank-1 terms.
                            pass
                        
                        col += 1
    
    # This is getting complicated. Let me use a different approach.
    # I'll directly construct the rank-28 decomposition by enumerating all scalar products.
    
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(16)]
    W = [[0]*28 for _ in range(16)]
    
    col = 0
    for s in range(7):
        # Define A and B coefficient vectors for this Strassen product
        # A_s is a 4x4 matrix of coefficients
        # B_s is a 4x4 matrix of coefficients
        # P_s = A_s * B_s (matrix product)
        # C += coeff_s * P_s
        
        if s == 0:  # P1 = (A11+A22)(B11+B22), coeff +1 for C11, +1 for C22
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A11 part
            A_s[0,0] = 1; A_s[0,1] = 1
            A_s[1,0] = 1; A_s[1,1] = 1
            # A22 part
            A_s[2,2] = 1; A_s[2,3] = 1
            A_s[3,2] = 1; A_s[3,3] = 1
            # B11 part
            B_s[0,0] = 1; B_s[0,1] = 1
            B_s[1,0] = 1; B_s[1,1] = 1
            # B22 part
            B_s[2,2] = 1; B_s[2,3] = 1
            B_s[3,2] = 1; B_s[3,3] = 1
            # C contributions: C11 += P1, C22 += P1
            C_s = np.zeros((4,4))
            C_s[0,0] = 1; C_s[0,1] = 1
            C_s[1,0] = 1; C_s[1,1] = 1
            C_s[2,2] = 1; C_s[2,3] = 1
            C_s[3,2] = 1; C_s[3,3] = 1
        elif s == 1:  # P2 = (A21+A22)B11, coeff +1 for C21, -1 for C22
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A21 part
            A_s[2,0] = 1; A_s[2,1] = 1
            A_s[3,0] = 1; A_s[3,1] = 1
            # A22 part
            A_s[2,2] = 1; A_s[2,3] = 1
            A_s[3,2] = 1; A_s[3,3] = 1
            # B11 part
            B_s[0,0] = 1; B_s[0,1] = 1
            B_s[1,0] = 1; B_s[1,1] = 1
            # C contributions: C21 += P2, C22 -= P2
            C_s = np.zeros((4,4))
            C_s[2,0] = 1; C_s[2,1] = 1
            C_s[3,0] = 1; C_s[3,1] = 1
            C_s[2,2] = -1; C_s[2,3] = -1
            C_s[3,2] = -1; C_s[3,3] = -1
        elif s == 2:  # P3 = A11(B12-B22), coeff +1 for C12, +1 for C22
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A11 part
            A_s[0,0] = 1; A_s[0,1] = 1
            A_s[1,0] = 1; A_s[1,1] = 1
            # B12 part (positive)
            B_s[0,2] = 1; B_s[0,3] = 1
            B_s[1,2] = 1; B_s[1,3] = 1
            # B22 part (negative)
            B_s[2,2] = -1; B_s[2,3] = -1
            B_s[3,2] = -1; B_s[3,3] = -1
            # C contributions: C12 += P3, C22 += P3
            C_s = np.zeros((4,4))
            C_s[0,2] = 1; C_s[0,3] = 1
            C_s[1,2] = 1; C_s[1,3] = 1
            C_s[2,2] = 1; C_s[2,3] = 1
            C_s[3,2] = 1; C_s[3,3] = 1
        elif s == 3:  # P4 = A22(B21-B11), coeff +1 for C11, +1 for C21
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A22 part
            A_s[2,2] = 1; A_s[2,3] = 1
            A_s[3,2] = 1; A_s[3,3] = 1
            # B21 part (positive)
            B_s[2,0] = 1; B_s[2,1] = 1
            B_s[3,0] = 1; B_s[3,1] = 1
            # B11 part (negative)
            B_s[0,0] = -1; B_s[0,1] = -1
            B_s[1,0] = -1; B_s[1,1] = -1
            # C contributions: C11 += P4, C21 += P4
            C_s = np.zeros((4,4))
            C_s[0,0] = 1; C_s[0,1] = 1
            C_s[1,0] = 1; C_s[1,1] = 1
            C_s[2,0] = 1; C_s[2,1] = 1
            C_s[3,0] = 1; C_s[3,1] = 1
        elif s == 4:  # P5 = (A11+A12)B22, coeff -1 for C11, +1 for C12
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A11 part
            A_s[0,0] = 1; A_s[0,1] = 1
            A_s[1,0] = 1; A_s[1,1] = 1
            # A12 part
            A_s[0,2] = 1; A_s[0,3] = 1
            A_s[1,2] = 1; A_s[1,3] = 1
            # B22 part
            B_s[2,2] = 1; B_s[2,3] = 1
            B_s[3,2] = 1; B_s[3,3] = 1
            # C contributions: C11 -= P5, C12 += P5
            C_s = np.zeros((4,4))
            C_s[0,0] = -1; C_s[0,1] = -1
            C_s[1,0] = -1; C_s[1,1] = -1
            C_s[0,2] = 1; C_s[0,3] = 1
            C_s[1,2] = 1; C_s[1,3] = 1
        elif s == 5:  # P6 = (A21-A11)(B11+B12), coeff +1 for C22
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A21 part (positive)
            A_s[2,0] = 1; A_s[2,1] = 1
            A_s[3,0] = 1; A_s[3,1] = 1
            # A11 part (negative)
            A_s[0,0] = -1; A_s[0,1] = -1
            A_s[1,0] = -1; A_s[1,1] = -1
            # B11 part
            B_s[0,0] = 1; B_s[0,1] = 1
            B_s[1,0] = 1; B_s[1,1] = 1
            # B12 part
            B_s[0,2] = 1; B_s[0,3] = 1
            B_s[1,2] = 1; B_s[1,3] = 1
            # C contributions: C22 += P6
            C_s = np.zeros((4,4))
            C_s[2,2] = 1; C_s[2,3] = 1
            C_s[3,2] = 1; C_s[3,3] = 1
        elif s == 6:  # P7 = (A12-A22)(B21+B22), coeff +1 for C11
            A_s = np.zeros((4,4))
            B_s = np.zeros((4,4))
            # A12 part (positive)
            A_s[0,2] = 1; A_s[0,3] = 1
            A_s[1,2] = 1; A_s[1,3] = 1
            # A22 part (negative)
            A_s[2,2] = -1; A_s[2,3] = -1
            A_s[3,2] = -1; A_s[3,3] = -1
            # B21 part
            B_s[2,0] = 1; B_s[2,1] = 1
            B_s[3,0] = 1; B_s[3,1] = 1
            # B22 part
            B_s[2,2] = 1; B_s[2,3] = 1
            B_s[3,2] = 1; B_s[3,3] = 1
            # C contributions: C11 += P7
            C_s = np.zeros((4,4))
            C_s[0,0] = 1; C_s[0,1] = 1
            C_s[1,0] = 1; C_s[1,1] = 1
        
        # Now decompose P_s = A_s * B_s into rank-1 terms
        # P_s[i,k] = sum_j A_s[i,j] * B_s[j,k]
        # Each (i,j,k) triple gives a rank-1 term with coefficient 1
        # But we can combine: for fixed s, the product A_s*B_s has rank at most 4
        # We decompose it as sum of at most 4 rank-1 terms
        
        # Actually, for the tensor decomposition, each scalar multiplication A[i,j]*B[j,k]
        # contributes to C[i,k]. In the Strassen framework, we have:
        # C[i,k] = sum_s coeff_s[i,k] * (A_s * B_s)[i,k]
        # = sum_s sum_j coeff_s[i,k] * A_s[i,j] * B_s[j,k]
        
        # Each term (s, i, j, k) where A_s[i,j] != 0 and B_s[j,k] != 0 and coeff_s[i,k] != 0
        # is a rank-1 term in the tensor decomposition.
        
        # But wait, that's not quite right. The tensor is:
        # T[u,v,w] where u = i*4+j (A index), v = j*4+k (B index), w = k*4+i (C index)
        # T[u,v,w] = 1 iff the (i,j,k) triple is valid.
        
        # In the Strassen decomposition:
        # C[i,k] = sum_s sum_j coeff_s[i,k] * A_s[i,j] * B_s[j,k]
        # = sum_s sum_j coeff_s[i,k] * A[i,j] * B[j,k] (when A_s[i,j]=1, B_s[j,k]=1)
        
        # So the tensor entry T[i*4+j, j*4+k, k*4+i] gets contribution:
        # sum_s coeff_s[i,k] * A_s[i,j] * B_s[j,k]
        # This should equal 1 for all valid (i,j,k).
        
        # Let me verify this works by constructing the full decomposition.
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a_val = A_s[i,j]
                    b_val = B_s[j,k]
                    c_val = C_s[i,k]
                    if a_val != 0 and b_val != 0 and c_val != 0:
                        U[i*4+j][col] = a_val
                        V[j*4+k][col] = b_val
                        W[k*4+i][col] = c_val
                        col += 1
    
    # col should be 28 (7 Strassen products * 4 scalar products each... but not exactly)
    # Actually, the number of non-zero (i,j,k) triples per Strassen product varies.
    # Let me just use col as the actual rank.
    
    rank = col
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return U, V, W, rank

def _extend_to_3x4x4(U4, V4, W4, rank4):
    """
    Extend a 4x4x4 decomposition to 3x4x4.
    The 3x4x4 tensor: T[u,v,w] where u=i*4+j (i in 0..2, j in 0..3),
    v=j*4+k (j in 0..3, k in 0..3), w=k*3+i (k in 0..3, i in 0..2)
    
    We can use the 4x4x4 decomposition and "project" the output from 4 to 3 dimensions.
    For the 3x4x4 case, C is 3x4, so w = k*3+i.
    The 4x4x4 has w = k*4+i.
    
    This doesn't directly work because the output dimension is different.
    Let me think differently.
    
    The 3x4x4 multiplication: C = A*B where A is 3x4, B is 4x4, C is 3x4.
    C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..3, k in 0..3.
    
    We can embed A into a 4x4 matrix by adding a zero row:
    A_ext = [[A[0,:]], [A[1,:]], [A[2,:]], [0,0,0,0]]
    Then C_ext = A_ext * B is 4x4, and C = C_ext[0:3, :].
    
    So we can use the 4x4x4 decomposition and just ignore the 4th row of C.
    The rank stays the same!
    
    U for 3x4x4: 12 rows (3*4), V: 16 rows (4*4), W: 12 rows (4*3)
    U4 has 16 rows, we take the first 12.
    W4 has 16 rows (w = k*4+i for k in 0..3, i in 0..3), we need w = k*3+i for k in 0..3, i in 0..2.
    This is a different indexing!
    
    W4[k*4+i] corresponds to C[i,k] in the 4x4 output.
    For 3x4x4, W[k*3+i] corresponds to C[i,k] in the 3x4 output.
    So W4[k*4+i] -> W[k*3+i] for i in 0..2 (drop i=3).
    """
    n, m, p = 3, 4, 4
    nm, mp, pn = 12, 16, 12
    
    U = [[0]*rank4 for _ in range(nm)]
    V = [[0]*rank4 for _ in range(mp)]
    W = [[0]*rank4 for _ in range(pn)]
    
    for c in range(rank4):
        # U: first 12 rows of U4 (i in 0..2, j in 0..3 -> u = i*4+j)
        for u in range(nm):
            U[u][c] = U4[u][c]
        # V: all 16 rows of V4
        for v in range(mp):
            V[v][c] = V4[v][c]
        # W: W4[k*4+i] -> W[k*3+i] for i in 0..2
        for k in range(4):
            for i in range(3):
                W[k*3+i][c] = W4[k*4+i][c]
    
    return U, V, W

def _schoolbook():
    """Return the trivial schoolbook decomposition with rank = n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, deadline):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if len(U) > 0 else 0
    
    if rank <= 1:
        return U, V, W, rank
    
    start_time = time.time()
    current_rank = rank
    
    # Try removing columns one by one
    for _ in range(50):
        if time.time() > deadline:
            break
        if current_rank <= 1:
            break
        
        # Try removing a random column
        idx = random.randint(0, current_rank - 1)
        
        new_U = [row[:idx] + row[idx+1:] for row in U]
        new_V = [row[:idx] + row[idx+1:] for row in V]
        new_W = [row[:idx] + row[idx+1:] for row in W]
        
        if _verify_fast(new_U, new_V, new_W):
            U, V, W = new_U, new_V, new_W
            current_rank -= 1
        else:
            # Try swapping with another column
            pass
    
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook()
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try Strassen-based 4x4x4 decomposition extended to 3x4x4
    try:
        U4, V4, W4, rank4 = _strassen_4x4()
        if rank4 > 0:
            U_ext, V_ext, W_ext = _extend_to_3x4x4(U4, V4, W4, rank4)
            if _verify_fast(U_ext, V_ext, W_ext):
                if rank4 < best_rank:
                    best_rank = rank4
                    best_U = U_ext
                    best_V = V_ext
                    best_W = W_ext
    except Exception:
        pass
    
    # Try to reduce rank further
    if best_rank < n * m * p and time.time() < deadline - 10:
        U_red, V_red, W_red, rank_red = _try_reduce_rank(best_U, best_V, best_W, deadline - 5)
        if rank_red < best_rank:
            best_rank = rank_red
            best_U = U_red
            best_V = V_red
            best_W = W_red
    
    return {"U": best_U, "V": best_V, "W": best_W}
