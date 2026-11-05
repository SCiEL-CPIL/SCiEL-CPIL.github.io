"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each term has form: sum of (coeff * A_entry) * (coeff * B_entry)
    # We represent each product as (U_coeffs, V_coeffs) where
    # U_coeffs is a dict from (i,j) to coeff, V_coeffs is dict from (j,k) to coeff
    
    terms = []
    
    # M1 = (a11+a22)*(b11+b22)
    terms.append((
        {(0,0): 1, (1,1): 1},  # U: a11 + a22
        {(0,0): 1, (0,1): 1}   # V: b11 + b22
    ))
    
    # M2 = (a21+a22)*b11
    terms.append((
        {(1,0): 1, (1,1): 1},  # U: a21 + a22
        {(0,0): 1}             # V: b11
    ))
    
    # M3 = a11*(b12-b22)
    terms.append((
        {(0,0): 1},            # U: a11
        {(0,1): 1, (1,1): -1}  # V: b12 - b22
    ))
    
    # M4 = a22*(b21-b11)
    terms.append((
        {(1,1): 1},            # U: a22
        {(1,0): 1, (0,0): -1}  # V: b21 - b11
    ))
    
    # M5 = (a11+a12)*b22
    terms.append((
        {(0,0): 1, (0,1): 1},  # U: a11 + a12
        {(1,1): 1}             # V: b22
    ))
    
    # M6 = (a21-a11)*(b11+b12)
    terms.append((
        {(1,0): 1, (0,0): -1}, # U: a21 - a11
        {(0,0): 1, (0,1): 1}   # V: b11 + b12
    ))
    
    # M7 = (a12-a22)*(b21+b22)
    terms.append((
        {(0,1): 1, (1,1): -1}, # U: a12 - a22
        {(1,0): 1, (1,1): 1}   # V: b21 + b22
    ))
    
    return terms

def _combine_blocks(block_decomp, n, m, p, block_n, block_m, block_p):
    """
    Combine block decomposition into full decomposition.
    
    block_decomp: list of (U_block, V_block, W_block) where each is a 2D array
                  representing a rank-1 contribution to the block multiplication
    n, m, p: full dimensions
    block_n, block_m, block_p: block dimensions
    
    Returns: (U, V, W) full matrices
    """
    # Number of blocks in each dimension
    blocks_n = n // block_n
    blocks_m = m // block_m
    blocks_p = p // block_p
    
    # Check if dimensions divide evenly
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    
    block_rank = len(block_decomp)
    total_rank = block_rank * blocks_n * blocks_m * blocks_p
    
    # Initialize full matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill in the block structure
    for bn in range(blocks_n):
        for bm in range(blocks_m):
            for bp in range(blocks_p):
                # Base column index for this specific block instance
                block_instance_idx = (bn * blocks_m + bm) * blocks_p + bp
                base_col = block_instance_idx * block_rank
                
                for r, (U_b, V_b, W_b) in enumerate(block_decomp):
                    global_col = base_col + r
                    
                    # A block (bn, bm)
                    for i in range(block_n):
                        for j in range(block_m):
                            a_idx = (bn * block_n + i) * m + (bm * block_m + j)
                            U[a_idx, global_col] = U_b[i, j]
                    
                    # B block (bm, bp)
                    for j in range(block_m):
                        for k in range(block_p):
                            b_idx = (bm * block_m + j) * p + (bp * block_p + k)
                            V[b_idx, global_col] = V_b[j, k]
                    
                    # C block (bp, bn)
                    for k in range(block_p):
                        for i in range(block_n):
                            c_idx = (bp * block_p + k) * n + (bn * block_n + i)
                            W[c_idx, global_col] = W_b[k, i]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using vectorized operations."""
    # The tensor T is defined such that T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We want T to be 1 at indices corresponding to valid matrix multiplications and 0 elsewhere.
    # Valid indices:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # for i in [0, n), j in [0, m), k in [0, p)
    
    # We can compute the full tensor T efficiently?
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This might be memory heavy for large dimensions, but 4x5x6 is small.
    # Dimensions: (20, 30, 24). 20*30*24 = 14400 entries. Very small.
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create a mask of valid positions
    # valid[a, b, c] is True if there exist i, j, k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # Instead of building a mask, we can just check the specific entries.
    # There are n*m*p = 120 valid entries.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
                
    # Check that all other entries are 0
    # We can zero out the valid entries and check if the rest is zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0.0
                
    if np.any(np.abs(T_check) > 1e-9):
        return False
        
    return True

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses block decomposition with Strassen's algorithm for sub-blocks.
    """
    n, m, p = DIMENSIONS
    
    # Try different block configurations
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as baseline
    
    # First, try schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_result = (U_school, V_school, W_school)
    
    # Try block decompositions
    # Possible block sizes that divide 4, 5, 6
    block_sizes = [
        (2, 2, 2),  # 2x2x2 Strassen
        (2, 1, 2),  # 2x1x2
        (1, 2, 2),  # 1x2x2
        (2, 2, 1),  # 2x2x1
        (1, 1, 2),  # 1x1x2
        (1, 2, 1),  # 1x2x1
        (2, 1, 1),  # 2x1x1
        (1, 1, 1),  # 1x1x1 (schoolbook)
    ]
    
    for bn, bm, bp in block_sizes:
        if n % bn != 0 or m % bm != 0 or p % bp != 0:
            continue
        
        # Get the block decomposition
        block_decomp = []
        
        if bn == 2 and bm == 2 and bp == 2:
            # Use Strassen's 7-term decomposition
            # We need to convert the list of (U_coeffs, V_coeffs) to (U_b, V_b, W_b)
            # Strassen's algorithm computes C = A*B where C is 2x2.
            # The output C entries are linear combinations of the 7 products.
            # However, the tensor decomposition requires that each rank-1 term
            # corresponds to a specific (U_b, V_b, W_b) such that
            # sum_r U_b[i,j] V_b[j,k] W_b[k,l] = C[l,i] (note index swap for trace convention?)
            # Wait, the trace convention in the task is W[k*n+i, r].
            # In the block context, W_b should be indexed [k_block, i_block] corresponding to output (i, k)?
            # Let's trace carefully.
            # Global: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is valid.
            # Block: U_b[i', j'] V_b[j', k'] W_b[k', i'] should sum to 1 for local (i', j', k').
            # So W_b[k', i'] is the coefficient for the output at local (i', k').
            
            # Let's derive the W_b for Strassen.
            # Standard Strassen:
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b12-b22)
            # M4 = a22(b21-b11)
            # M5 = (a11+a12)b22
            # M6 = (a21-a11)(b11+b12)
            # M7 = (a12-a22)(b21+b22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to decompose C into sum of rank-1 tensors.
            # Each term r has U_b, V_b, W_b.
            # The contribution to C[i, k] is sum_j U_b[i, j] V_b[j, k] ... wait.
            # The tensor T[i, j, k] = sum_r U[i, j] V[j, k] W[k, i].
            # This T[i, j, k] is 1 if the multiplication path is valid.
            # This is equivalent to saying that the matrix product C = A*B is computed.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # So we need sum_r U_b[i, j] V_b[j, k] W_b[k, i] = delta_{i, k?} no.
            # Actually, the decomposition defines the multiplication algorithm.
            # If we sum over r, we get the value of the "product" at (i, k) for inputs A, B?
            # No, the tensor T represents the bilinear form.
            # T[i, j, k] is the coefficient of A[i, j] B[j, k] in the output C[i, k]?
            # No, C[i, k] = sum_j A[i, j] B[j, k].
            # So the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
            # The coefficient of A[i, j] B[j, k] in C[i', k'] is 0 if (i, k) != (i', k').
            
            # The decomposition provides U, V, W such that:
            # sum_r U[i, j] V[j, k] W[k, i] = 1 if we are looking at the term for C[i, k]?
            # Wait, the index of W is k*n+i. So W is indexed by (k, i).
            # So the term is U[i, j] V[j, k] W[k, i].
            # This sums to 1 for the specific (i, j, k) triple.
            # This means the decomposition computes the diagonal of the "product" tensor?
            # Actually, this is the standard definition of the matrix multiplication tensor.
            
            # For Strassen, we define 7 products M_r.
            # M_r = sum_j U_r[i, j] A[i, j] * sum_k V_r[j, k] B[j, k]? No.
            # M_r = (sum_j U_r[i, j] A[i, j]) * (sum_k V_r[j, k] B[j, k])? No, U and V depend on r.
            # M_r = (linear combo of A) * (linear combo of B).
            # Let u_r be the vector of coefficients for A, v_r for B.
            # M_r = (sum_{i,j} U_r[i,j] A[i,j]) * (sum_{j,k} V_r[j,k] B[j,k]).
            # Then C[i, k] = sum_r W_r[k, i] M_r.
            
            # So for each term r:
            # U_b is the matrix of coefficients for A.
            # V_b is the matrix of coefficients for B.
            # W_b is the matrix of coefficients for C.
            
            # We have the formulas for M1..M7.
            # We need to extract U_b, V_b for each M_r.
            # And then determine W_b such that sum_r W_b[k, i] M_r = C[i, k].
            
            # Let's list U and V for each M:
            # M1: U={(0,0):1, (1,1):1}, V={(0,0):1, (0,1):1}
            # M2: U={(1,0):1, (1,1):1}, V={(0,0):1}
            # M3: U={(0,0):1}, V={(0,1):1, (1,1):-1}
            # M4: U={(1,1):1}, V={(1,0):1, (0,0):-1}
            # M5: U={(0,0):1, (0,1):1}, V={(1,1):1}
            # M6: U={(1,0):1, (0,0):-1}, V={(0,0):1, (0,1):1}
            # M7: U={(0,1):1, (1,1):-1}, V={(1,0):1, (1,1):1}
            
            # Now we need W_b for each.
            # C[0,0] = M1 + M4 - M5 + M7
            # C[0,1] = M3 + M5
            # C[1,0] = M2 + M4
            # C[1,1] = M1 - M2 + M3 + M6
            
            # W_b[k, i] is the coefficient of M_r in C[i, k].
            # Note indices: W is [k, i]. C is [i, k].
            
            # For r=1 (M1):
            # C[0,0] coeff: 1 -> W[0, 0] = 1
            # C[1,1] coeff: 1 -> W[1, 1] = 1
            # Others 0.
            
            # For r=2 (M2):
            # C[1,0] coeff: 1 -> W[0, 1] = 1
            # C[1,1] coeff: -1 -> W[1, 1] = -1
            
            # For r=3 (M3):
            # C[0,1] coeff: 1 -> W[1, 0] = 1
            # C[1,1] coeff: 1 -> W[1, 1] = 1
            
            # For r=4 (M4):
            # C[0,0] coeff: 1 -> W[0, 0] = 1
            # C[1,0] coeff: 1 -> W[0, 1] = 1
            
            # For r=5 (M5):
            # C[0,0] coeff: -1 -> W[0, 0] = -1
            # C[0,1] coeff: 1 -> W[1, 0] = 1
            
            # For r=6 (M6):
            # C[1,1] coeff: 1 -> W[1, 1] = 1
            
            # For r=7 (M7):
            # C[0,0] coeff: 1 -> W[0, 0] = 1
            
            # Let's construct these.
            strassen_terms = _strassen_2x2x2()
            for idx, (U_coeffs, V_coeffs) in enumerate(strassen_terms):
                U_b = np.zeros((2, 2))
                V_b = np.zeros((2, 2))
                W_b = np.zeros((2, 2))
                
                for (i, j), coeff in U_coeffs.items():
                    U_b[i, j] = coeff
                for (j, k), coeff in V_coeffs.items():
                    V_b[j, k] = coeff
                
                # Assign W_b based on derived coefficients
                r = idx + 1
                if r == 1:
                    W_b[0, 0] = 1
                    W_b[1, 1] = 1
                elif r == 2:
                    W_b[0, 1] = 1
                    W_b[1, 1] = -1
                elif r == 3:
                    W_b[1, 0] = 1
                    W_b[1, 1] = 1
                elif r == 4:
                    W_b[0, 0] = 1
                    W_b[0, 1] = 1
                elif r == 5:
                    W_b[0, 0] = -1
                    W_b[1, 0] = 1
                elif r == 6:
                    W_b[1, 1] = 1
                elif r == 7:
                    W_b[0, 0] = 1
                    
                block_decomp.append((U_b, V_b, W_b))
        else:
            # For smaller blocks, use schoolbook within each block
            for i in range(bn):
                for j in range(bm):
                    for k in range(bp):
                        U_b = np.zeros((bn, bm))
                        V_b = np.zeros((bm, bp))
                        W_b = np.zeros((bp, bn))
                        U_b[i, j] = 1
                        V_b[j, k] = 1
                        W_b[k, i] = 1
                        block_decomp.append((U_b, V_b, W_b))
        
        result = _combine_blocks(block_decomp, n, m, p, bn, bm, bp)
        if result is None:
            continue
        
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            current_rank = U.shape[1]
            if current_rank < best_rank:
                best_rank = current_rank
                best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
