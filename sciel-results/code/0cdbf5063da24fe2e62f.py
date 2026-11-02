"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, simplify, expand

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_decompose(T, rank, n, m, p, max_iters=1000, tol=1e-12, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (n*m, m*p, p*n)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    # Normalize initial vectors to avoid scaling issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
            # Keep product of norms constant? Or just normalize?
            # Usually, we can scale U up and V down.
            # Let's just normalize to unit length for stability, then let ALS adjust.
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
            
    prev_error = float('inf')
    
    for it in range(max_iters):
        # Update U
        for a in range(dim_a):
            # Form matrix M (dim_b * dim_c) x rank
            # M[bc, r] = V[b, r] * W[c, r]
            M = np.einsum('br,cr->bc r', V, W)
            M_flat = M.reshape(dim_b * dim_c, rank)
            T_flat = T[a, :, :].reshape(-1)
            # Solve M_flat @ u_a = T_flat
            u_a, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        for b in range(dim_b):
            M = np.einsum('ar,cr->ac r', U, W)
            M_flat = M.reshape(dim_a * dim_c, rank)
            T_flat = T[:, b, :].reshape(-1)
            v_b, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        for c in range(dim_c):
            M = np.einsum('ar,br->ab r', U, V)
            M_flat = M.reshape(dim_a * dim_b, rank)
            T_flat = T[:, :, c].reshape(-1)
            w_c, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-6:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-6:
        return U, V, W
    return None

def _try_rank23(n, m, p):
    """
    Attempt to find a rank-23 decomposition using ALS.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Get schoolbook decomposition
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    
    # Strategy: Drop 4 terms from schoolbook and use ALS to recover
    # Try different sets of 4 terms to drop
    all_indices = list(range(total_terms))
    
    # To save time, we can try a few random subsets and some structured ones
    attempts = 0
    max_attempts = 50 # 50 attempts * 1000 iters * 27 lstsq is fast
    
    # Random shuffling of indices to drop
    random.seed(42)
    
    for _ in range(max_attempts):
        # Pick 4 indices to drop
        drop_indices = random.sample(all_indices, 4)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        # Run ALS
        result = _als_decompose(T, 23, n, m, p, max_iters=200, tol=1e-12, 
                                init_U=init_U, init_V=init_V, init_W=init_W)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
                
    return None

def _construct_rank26():
    """
    Construct a rank-26 decomposition by partitioning into 2x2 and 1x1 blocks
    and applying Strassen's algorithm to the 2x2 block.
    
    Partition:
    A = [[A11, A12], [A21, A22]] where A11, A12, A21, A22 are 2x2, 2x1, 1x2, 1x1
    B = [[B11, B12], [B21, B22]]
    
    Wait, the task is 3x3 * 3x3. Let's partition as:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    
    Product C = A*B:
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x1)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x1)
    
    For the 2x2 part (C11 = A11*B11), we can use Strassen with 7 multiplications.
    But A11 and B11 are 2x2 matrices.
    
    Let's define the blocks explicitly.
    A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    C = A*B
    c11 = a11*b11 + a12*b21 + a13*b31
    c12 = a11*b12 + a12*b22 + a13*b32
    c13 = a11*b13 + a12*b23 + a13*b33
    c21 = a21*b11 + a22*b21 + a23*b31
    c22 = a21*b12 + a22*b22 + a23*b32
    c23 = a21*b13 + a22*b23 + a23*b33
    c31 = a31*b11 + a32*b21 + a33*b31
    c32 = a31*b12 + a32*b22 + a33*b32
    c33 = a31*b13 + a32*b23 + a33*b33
    
    Schoolbook has 27 multiplications.
    
    Let's try to use Strassen on the top-left 2x2 block of A and B.
    Let A1 = [[a11, a12], [a21, a22]], B1 = [[b11, b12], [b21, b22]]
    Let A2 = [a13, a23] (2x1), B2 = [b13, b23] (2x1)
    Let A3 = [a31, a32] (1x2), B3 = [b31, b32] (1x2)
    Let a33, b33 be scalars.
    
    C11 = A1*B1 + A2*B3  (2x2)
    C12 = A1*B2 + a13*b33  (2x1)
    C21 = A3*B1 + a33*B2  (1x2)
    C22 = A3*B2 + a33*b33  (1x1)
    
    This doesn't seem to reduce the count directly because the cross terms are still there.
    
    Let's try a different approach. The known rank-26 decomposition for 3x3 matrix multiplication
    can be constructed by viewing the 3x3 matrix multiplication as a 2x2 block multiplication
    where one of the blocks is 1x1, and using Strassen for the 2x2 part.
    
    Actually, there is a known rank-26 algorithm. It's often derived by partitioning into
    2x2 and 1x1 blocks and using 7 multiplications for the 2x2 part plus some additional
    multiplications for the cross terms.
    
    Let's use the specific construction from literature.
    One such construction uses the following 26 products:
    
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 + A22)(B21 + B22)
    
    These are for the 2x2 block product.
    
    For the 3x3 case, we need to handle the interactions with the 3rd row and column.
    
    Let's try to implement a known rank-26 decomposition directly.
    
    Actually, let's try to use the ALS approach but with a better initialization.
    The current ALS approach might be getting stuck in local minima.
    
    Let's try to use the known rank-26 decomposition.
    """
    # Define the rank-26 decomposition coefficients
    # This is a known decomposition for 3x3 matrix multiplication
    
    # The decomposition can be represented as:
    # C[i,j] = sum_r U[i,r] * V[r,j] * W[j,r]
    # where U, V, W are the factor matrices
    
    # Let's construct it using the block structure
    # A = [[A11, a13], [a31, a33]] where A11 is 2x2, a13 is 2x1, a31 is 1x2, a33 is 1x1
    # B = [[B11, b13], [b31, b33]] where B11 is 2x2, b13 is 2x1, b31 is 1x2, b33 is 1x1
    
    # C = [[C11, c13], [c31, c33]]
    # C11 = A11*B11 + a13*b31
    # c13 = A11*b13 + a13*b33
    # c31 = a31*B11 + a33*b31
    # c33 = a31*b13 + a33*b33
    
    # For C11 = A11*B11, we can use Strassen with 7 multiplications
    # Let A11 = [[a11, a12], [a21, a22]], B11 = [[b11, b12], [b21, b22]]
    
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 + a22)(b21 + b22)
    
    # C11_11 = M1 - M5 + M7 + M4
    # C11_12 = M3 + M5
    # C11_21 = M2 + M4
    # C11_22 = M1 - M2 + M3 + M6
    
    # Now we need to handle the cross terms
    # c13_1 = a11*b13 + a12*b23 + a13*b33
    # c13_2 = a21*b13 + a22*b23 + a13*b33
    # c31_1 = a31*b11 + a32*b21 + a33*b31
    # c31_2 = a31*b12 + a32*b22 + a33*b31
    # c33 = a31*b13 + a32*b23 + a33*b33
    
    # This is getting complicated. Let's try a different approach.
    
    # Use the known rank-26 decomposition from the literature
    # One such decomposition uses the following 26 products:
    
    # Let's define the 26 products explicitly
    products = []
    
    # 7 products for the 2x2 block (Strassen)
    # M1 = (a11 + a22)(b11 + b22)
    products.append(((0,0), (2,2), (0,0), (2,2)))  # (a11+a22)(b11+b22)
    # M2 = (a21 + a22)b11
    products.append(((2,0), (2,1), (0,0), (0,0)))  # (a21+a22)b11
    # M3 = a11(b12 - b22)
    products.append(((0,0), (0,0), (0,1), (1,1)))  # a11(b12-b22)
    # M4 = a22(b21 - b11)
    products.append(((1,1), (1,1), (1,0), (0,0)))  # a22(b21-b11)
    # M5 = (a11 + a12)b22
    products.append(((0,0), (0,1), (1,1), (1,1)))  # (a11+a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    products.append(((2,0), (0,0), (0,0), (0,1)))  # (a21-a11)(b11+b12)
    # M7 = (a12 + a22)(b21 + b22)
    products.append(((0,1), (2,2), (1,0), (1,1)))  # (a12+a22)(b21+b22)
    
    # This approach is too complex to implement correctly without the exact coefficients.
    # Let's fall back to the ALS approach but with more attempts and better initialization.
    
    return None

def _try_rank26(n, m, p):
    """
    Attempt to find a rank-26 decomposition using ALS with better initialization.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Get schoolbook decomposition
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    
    # Strategy: Drop 1 term from schoolbook and use ALS to recover
    all_indices = list(range(total_terms))
    
    random.seed(42)
    
    for _ in range(100):
        # Pick 1 index to drop
        drop_indices = random.sample(all_indices, 1)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        # Run ALS
        result = _als_decompose(T, 26, n, m, p, max_iters=500, tol=1e-12, 
                                init_U=init_U, init_V=init_V, init_W=init_W)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Try to find rank 26
    if time.time() < deadline:
        result = _try_rank26(n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
