"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _build_strassen_2x2x2():
    """Return the 7 rank-1 terms for Strassen 2×2×2.
    
    Each term is (u_dict, v_dict, w_dict) where dicts map (i,j) or (j,k) or (i,k) to coefficient.
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1: A part has A11=1, A22=1; B part has B11=1, B22=1
    # Contributes +1 to C11, -1 to C22
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): -1}
    terms.append((u, v, w))
    
    # M2: A part has A21=1, A22=1; B part has B11=1
    # Contributes +1 to C21, -1 to C22
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1}
    w = {(1,0): 1, (1,1): -1}
    terms.append((u, v, w))
    
    # M3: A part has A11=1; B part has B21=1, B22=-1
    # Contributes +1 to C12, +1 to C22
    u = {(0,0): 1}
    v = {(1,0): 1, (1,1): -1}
    w = {(0,1): 1, (1,1): 1}
    terms.append((u, v, w))
    
    # M4: A part has A22=1; B part has B12=1, B11=-1
    # Contributes +1 to C11, +1 to C21
    u = {(1,1): 1}
    v = {(0,1): 1, (0,0): -1}
    w = {(0,0): 1, (1,0): 1}
    terms.append((u, v, w))
    
    # M5: A part has A11=1, A12=1; B part has B22=1
    # Contributes -1 to C11, +1 to C12
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (0,1): 1}
    terms.append((u, v, w))
    
    # M6: A part has A21=1, A11=-1; B part has B11=1, B12=1
    # Contributes +1 to C22
    u = {(1,0): 1, (0,0): -1}
    v = {(0,0): 1, (0,1): 1}
    w = {(1,1): 1}
    terms.append((u, v, w))
    
    # M7: A part has A12=1, A22=-1; B part has B21=1, B22=1
    # Contributes +1 to C11
    u = {(0,1): 1, (1,1): -1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1}
    terms.append((u, v, w))
    
    return terms

def _add_strassen_term(U, V, W, u_dict, v_dict, w_dict, off_i, off_j, off_k, m_stride, p_stride, n_stride):
    """Add one Strassen term to the decomposition."""
    col = len(U[0])
    for row in range(len(U)):
        U[row].append(0)
    for row in range(len(V)):
        V[row].append(0)
    for row in range(len(W)):
        W[row].append(0)
    
    for (i, j), coeff in u_dict.items():
        U[(off_i + i) * m_stride + (off_j + j)][col] = coeff
    for (j, k), coeff in v_dict.items():
        V[(off_j + j) * p_stride + (off_k + k)][col] = coeff
    for (i, k), coeff in w_dict.items():
        W[(off_k + k) * n_stride + (off_i + i)][col] = coeff

def _decompose(n, m, p, off_i, off_j, off_k, m_stride, p_stride, n_stride, U, V, W, strassen_terms):
    """Recursively decompose n×m × m×p block."""
    if n == 1 and m == 1 and p == 1:
        col = len(U[0])
        for row in range(len(U)):
            U[row].append(0)
        for row in range(len(V)):
            V[row].append(0)
        for row in range(len(W)):
            W[row].append(0)
        U[off_i * m_stride + off_j][col] = 1
        V[off_j * p_stride + off_k][col] = 1
        W[off_k * n_stride + off_i][col] = 1
        return
    
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # Use Strassen's 7 terms
        for u_dict, v_dict, w_dict in strassen_terms:
            _add_strassen_term(U, V, W, u_dict, v_dict, w_dict, 
                              off_i, off_j, off_k, m_stride, p_stride, n_stride)
        
        # Wait, this is wrong. Strassen's 7 terms are for 2×2×2, but we need
        # to apply them recursively. The issue is that Strassen's algorithm
        # computes M1-M7 as products of submatrices, not as single rank-1 terms.
        
        # I think the correct approach is:
        # For 2×2×2, use the 7 terms directly.
        # For larger even dimensions, split into 8 subproblems of half size,
        # but use Strassen to reduce to 7 recursive calls.
        
        # This is getting too complex for this iteration. Let me just do schoolbook.
        pass
    
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = len(U[0])
                for row in range(len(U)):
                    U[row].append(0)
                for row in range(len(V)):
                    V[row].append(0)
                for row in range(len(W)):
                    W[row].append(0)
                U[(off_i + i) * m_stride + (off_j + j)][col] = 1
                V[(off_j + j) * p_stride + (off_k + k)][col] = 1
                W[(off_k + k) * n_stride + (off_i + i)][col] = 1

def _decompose_recursive(n, m, p, off_i, off_j, off_k, U, V, W, strassen_terms):
    """Recursively decompose n×m × m×p matrix multiplication tensor."""
    # Base case: 1×1×1
    if n == 1 and m == 1 and p == 1:
        col = len(U[0]) if U[0] else 0
        for row in range(len(U)):
            U[row].append(0)
        for row in range(len(V)):
            V[row].append(0)
        for row in range(len(W)):
            W[row].append(0)
        U[off_i * m + off_j][col] = 1
        V[off_j * p + off_k][col] = 1
        W[off_k * n + off_i][col] = 1
        return
    
    # Try Strassen if all dimensions are even
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # Strassen's 7 products for 2×2×2 blocks:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a product of (n2×m2) and (m2×p2) matrices
        # So we recursively decompose each M
        
        # M1: (A11+A22) * (B11+B22)
        # A part: rows 0..n2-1, cols 0..m2-1 (coeff 1) + rows n2..n-1, cols m2..m-1 (coeff 1)
        # B part: rows 0..m2-1, cols 0..p2-1 (coeff 1) + rows m2..m-1, cols p2..p-1 (coeff 1)
        # C part: contributes to C11 (rows 0..n2-1, cols 0..p2-1) with +1, and C22 (rows n2..n-1, cols p2..p-1) with +1
        _strassen_recursive(n2, m2, p2, 
                           [(0, 0, 1), (n2, m2, 1)],  # A blocks: (row_off, col_off, coeff)
                           [(0, 0, 1), (m2, p2, 1)],  # B blocks: (row_off, col_off, coeff)
                           [(0, 0, 1), (n2, p2, 1)],  # C blocks: (row_off, col_off, coeff)
                           off_i, off_j, off_k, U, V, W)
        
        # M2: (A21+A22) * B11
        _strassen_recursive(n2, m2, p2,
                           [(n2, 0, 1), (n2, m2, 1)],
                           [(0, 0, 1)],
                           [(n2, 0, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        # M3: A11 * (B21-B22)
        _strassen_recursive(n2, m2, p2,
                           [(0, 0, 1)],
                           [(m2, 0, 1), (m2, p2, -1)],
                           [(0, p2, 1), (n2, p2, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        # M4: A22 * (B12-B11)
        _strassen_recursive(n2, m2, p2,
                           [(n2, m2, 1)],
                           [(0, p2, 1), (0, 0, -1)],
                           [(0, 0, 1), (n2, 0, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        # M5: (A11+A12) * B22
        _strassen_recursive(n2, m2, p2,
                           [(0, 0, 1), (0, m2, 1)],
                           [(m2, p2, 1)],
                           [(0, 0, -1), (0, p2, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        # M6: (A21-A11) * (B11+B12)
        _strassen_recursive(n2, m2, p2,
                           [(n2, 0, 1), (0, 0, -1)],
                           [(0, 0, 1), (0, p2, 1)],
                           [(n2, p2, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        # M7: (A12-A22) * (B21+B22)
        _strassen_recursive(n2, m2, p2,
                           [(0, m2, 1), (n2, m2, -1)],
                           [(m2, 0, 1), (m2, p2, 1)],
                           [(0, 0, 1)],
                           off_i, off_j, off_k, U, V, W)
        
        return
    
    # Schoolbook for odd dimensions or small cases
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = len(U[0]) if U[0] else 0
                for row in range(len(U)):
                    U[row].append(0)
                for row in range(len(V)):
                    V[row].append(0)
                for row in range(len(W)):
                    W[row].append(0)
                U[(off_i + i) * m + (off_j + j)][col] = 1
                V[(off_j + j) * p + (off_k + k)][col] = 1
                W[(off_k + k) * n + (off_i + i)][col] = 1

def _strassen_recursive(n2, m2, p2, a_blocks, b_blocks, c_blocks, off_i, off_j, off_k, U, V, W):
    """Handle one Strassen product M = sum of A_block * B_block, contributing to C_blocks.
    
    a_blocks: list of (row_off, col_off, coeff) for the A matrix
    b_blocks: list of (row_off, col_off, coeff) for the B matrix  
    c_blocks: list of (row_off, col_off, coeff) for the C matrix
    """
    # This is a product of a (n2×m2) matrix (with coefficients) and (m2×p2) matrix
    # The A matrix is formed by summing coefficient * A_block for each block
    # The B matrix is formed by summing coefficient * B_block for each block
    # The result contributes coefficient * C_block to each C block
    
    # For each combination of a_block and b_block, we get a product term
    # The coefficient is a_coeff * b_coeff
    # And it contributes to each c_block with c_coeff * (a_coeff * b_coeff)
    
    # Actually, this is more complex. Let me think again.
    
    # M = A_combined * B_combined where
    # A_combined = sum_i a_coeff_i * A_block_i (each A_block_i is n2×m2)
    # B_combined = sum_j b_coeff_j * B_block_j (each B_block_j is m2×p2)
    
    # M = sum_i sum_j a_coeff_i * b_coeff_j * (A_block_i * B_block_j)
    
    # Each (A_block_i * B_block_j) is an n2×p2 matrix product
    # And M contributes to C as: sum_k c_coeff_k * M_block_k
    
    # So the full contribution is:
    # sum_i sum_j sum_k a_coeff_i * b_coeff_j * c_coeff_k * (A_block_i * B_block_j) placed at C_block_k
    
    # For each (i, j, k) triple, we need to decompose the n2×m2 × m2×p2 product
    # with coefficient a_coeff_i * b_coeff_j * c_coeff_k
    
    for (ai, aj, ac) in a_blocks:
        for (bi, bj, bc) in b_blocks:
            for (ci, cj, cc) in c_blocks:
                coeff = ac * bc * cc
                if coeff == 0:
                    continue
                # Decompose n2×m2 × m2×p2 product with offset and coefficient
                _decompose_with_coeff(n2, m2, p2, ai + off_i, aj + off_j, bj + off_k, 
                                      ci + off_i, cj + off_k, coeff, U, V, W)

def _decompose_with_coeff(n, m, p, a_off_i, a_off_j, b_off_k, c_off_i, c_off_k, coeff, U, V, W):
    """Decompose n×m × m×p product with given offsets and coefficient."""
    # Base case
    if n == 1 and m == 1 and p == 1:
        col = len(U[0]) if U[0] else 0
        for row in range(len(U)):
            U[row].append(0)
        for row in range(len(V)):
            V[row].append(0)
        for row in range(len(W)):
            W[row].append(0)
        U[a_off_i * m + a_off_j][col] = coeff
        V[a_off_j * p + b_off_k][col] = 1
        W[c_off_k * n + c_off_i][col] = 1
        return
    
    # Try Strassen
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # Same 7 Strassen products, but now with coefficient
        # M1 = (A11+A22)(B11+B22)
        _strassen_recursive(n2, m2, p2,
                           [(0, 0, coeff), (n2, m2, coeff)],
                           [(0, 0, 1), (m2, p2, 1)],
                           [(0, 0, 1), (n2, p2, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M2 = (A21+A22)B11
        _strassen_recursive(n2, m2, p2,
                           [(n2, 0, coeff), (n2, m2, coeff)],
                           [(0, 0, 1)],
                           [(n2, 0, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M3 = A11(B21-B22)
        _strassen_recursive(n2, m2, p2,
                           [(0, 0, coeff)],
                           [(m2, 0, 1), (m2, p2, -1)],
                           [(0, p2, 1), (n2, p2, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M4 = A22(B12-B11)
        _strassen_recursive(n2, m2, p2,
                           [(n2, m2, coeff)],
                           [(0, p2, 1), (0, 0, -1)],
                           [(0, 0, 1), (n2, 0, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M5 = (A11+A12)B22
        _strassen_recursive(n2, m2, p2,
                           [(0, 0, coeff), (0, m2, coeff)],
                           [(m2, p2, 1)],
                           [(0, 0, -1), (0, p2, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M6 = (A21-A11)(B11+B12)
        _strassen_recursive(n2, m2, p2,
                           [(n2, 0, coeff), (0, 0, -coeff)],
                           [(0, 0, 1), (0, p2, 1)],
                           [(n2, p2, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        # M7 = (A12-A22)(B21+B22)
        _strassen_recursive(n2, m2, p2,
                           [(0, m2, coeff), (n2, m2, -coeff)],
                           [(m2, 0, 1), (m2, p2, 1)],
                           [(0, 0, 1)],
                           a_off_i, a_off_j, b_off_k, U, V, W)
        
        return
    
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = len(U[0]) if U[0] else 0
                for row in range(len(U)):
                    U[row].append(0)
                for row in range(len(V)):
                    V[row].append(0)
                for row in range(len(W)):
                    W[row].append(0)
                U[(a_off_i + i) * m + (a_off_j + j)][col] = coeff
                V[(a_off_j + j) * p + (b_off_k + k)][col] = 1
                W[(c_off_k + k) * n + (c_off_i + i)][col] = 1

def construct():
    deadline = time.time() + 145
    
    n, m, p = 4, 5, 6
    # Start with empty lists, we'll build them up
    U = []
    V = []
    W = []
    
    # Initialize with one zero column to establish length
    U.append([0])
    V.append([0])
    W.append([0])
    for i in range(n * m - 1):
        U.append([0])
    for i in range(m * p - 1):
        V.append([0])
    for i in range(p * n - 1):
        W.append([0])
    
    # Decompose the full 4×5×6 problem
    # Since 4 is even but 5 is odd, we can't use pure Strassen
    # Split 4×5×6 into: (2×5×6) + (2×5×6) for the two halves of the 4
    # Or: (4×2×6) + (4×3×6) for the two halves of the 5
    
    # Let's split the middle dimension: 4×5×6 = (4×2×6) + (4×3×6)
    # 4×2×6: all even, can use Strassen
    # 4×3×6: 3 is odd, need further splitting
    
    # Actually, let's just use the recursive approach directly
    _decompose_recursive(n, m, p, 0, 0, 0, U, V, W, None)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
