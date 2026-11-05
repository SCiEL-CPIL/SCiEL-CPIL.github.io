"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    rank = 54
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        i_base: base index for i (rows of A, rows of C)
        j_base: base index for j (cols of A, rows of B)
        k_base: base index for k (cols of B, cols of C)
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base, col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[j_base * p + k_base, col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base, col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] -= 1
        W[(k_base + 1) * n + i_base, col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[(j_base + 1) * p + k_base, col] += 1
        V[j_base * p + k_base, col] -= 1
        W[k_base * n + i_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base, col] += 1
        U[i_base * m + (j_base + 1), col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] -= 1
        W[(k_base + 1) * n + i_base, col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base, col] += 1
        U[i_base * m + j_base, col] -= 1
        V[j_base * p + k_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1), col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] -= 1
        V[(j_base + 1) * p + k_base, col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        i_base: base index for i
        j_base: single index for j
        k_base: base index for k
        """
        nonlocal col
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + i_base, col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # and k pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4 and k pairs (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != rank:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    
    if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
        # If verification fails, return schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction via linear dependency search
    # We look for a linear combination of columns that sums to zero.
    # Let X be a vector of coefficients. We want X^T U_r = 0, X^T V_r = 0, X^T W_r = 0 for all r?
    # No, we want sum_r X_r * (U[:,r] \otimes V[:,r] \otimes W[:,r]) = 0.
    # This is equivalent to finding a null vector of the tensor viewed as a matrix.
    # However, the tensor is 3-way. We can reshape it to a matrix of size (nm*mp) x pn or similar.
    # Let M be the matrix such that M[(a,b), c] = sum_r U[a,r] V[b,r] W[c,r].
    # This is exactly the tensor T_dec reshaped.
    # If there is a linear dependency, the matrix M has rank < 54?
    # No, the rank of the tensor is the minimum number of rank-1 terms.
    # If the 54 terms are linearly dependent, we can remove one.
    # The condition for linear dependence is that there exist coefficients c_r not all zero such that
    # sum_r c_r U[:,r] \otimes V[:,r] \otimes W[:,r] = 0.
    # This implies that for all a,b,c: sum_r c_r U[a,r] V[b,r] W[c,r] = 0.
    # This is a system of linear equations in c_r.
    # Number of equations: nm * mp * pn = 10 * 30 * 12 = 3600.
    # Number of unknowns: 54.
    # We can solve this using least squares or SVD.
    # Construct matrix A of size (nm*mp*pn) x 54.
    # A[index, r] = U[a,r] * V[b,r] * W[c,r] where index = (a*mp + b)*pn + c.
    
    if time.time() < deadline - 5.0:
        # Build the matrix A
        # A has 3600 rows and 54 cols
        # We can build it block by block to save memory/time
        # A is effectively the vectorized tensor T_dec.
        # T_dec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We want to find c such that sum_r c_r (U[:,r] \otimes V[:,r] \otimes W[:,r]) = 0
        # This is equivalent to finding the null space of the matrix formed by stacking the terms.
        
        # Let's form the matrix M where M[i, r] is the i-th element of the r-th rank-1 tensor.
        # i corresponds to (a, b, c).
        # M[i, r] = U[a, r] * V[b, r] * W[c, r]
        
        # We can compute this as:
        # M = np.einsum('ar,br,cr->abc r', U, V, W).reshape(nm*mp*pn, rank)
        # But this creates a 3600 x 54 array.
        
        M = np.einsum('ar,br,cr->abc r', U, V, W).reshape(nm*mp*pn, rank)
        
        # Find null space
        # Use SVD
        try:
            U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
            # The null space is spanned by the right singular vectors corresponding to zero singular values.
            # Since we are looking for exact dependencies (integer coefficients), we check for singular values close to 0.
            # However, due to floating point, we might not get exact zeros.
            # The rank of M is the number of non-zero singular values.
            # If rank(M) < 54, there is a dependency.
            
            tol = 1e-10
            r_rank = np.sum(s > tol)
            
            if r_rank < rank:
                # There is a null space.
                # The null space is spanned by Vh[r_rank:].T
                # We want to find an integer vector in this null space.
                # For now, let's just see if we can remove columns.
                # This is complex. A simpler heuristic:
                # If there is a dependency, we can remove one column and solve for the coefficients of the others?
                # Actually, if sum c_r T_r = 0, we can write T_1 = -sum_{r>1} (c_r/c_1) T_r.
                # So we can drop T_1 and adjust the others.
                
                # Let's try to find a simple integer dependency.
                # Project Vh onto integer lattice?
                # Or just check if any column is a linear combination of others.
                
                # For now, let's just return the 54 rank if no simple reduction is found.
                # The probability of finding a simple integer dependency in a random-looking null space is low.
                # But Strassen tensors often have structure.
                
                # Let's try to remove one column at a time and see if the remaining can still span the tensor.
                # This is expensive.
                
                pass
        except np.linalg.LinAlgError:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
