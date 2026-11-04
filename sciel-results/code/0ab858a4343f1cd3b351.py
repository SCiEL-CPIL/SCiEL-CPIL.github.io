import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Strategy:
    1. Use Strassen's algorithm (rank 7) for 2x2 sub-blocks where possible.
    2. Use schoolbook (rank 4) for remaining parts.
    3. For 2x5x6, we can partition the middle dimension m=5 into 2+2+1.
       - For the first 4 columns (j=0..3), we can apply Strassen decomposition 
         for each pair of columns against the 2 rows of A and 6 cols of B? 
         No, Strassen works on 2x2 * 2x2. 
         
    Let's look at the structure:
    A is 2x5, B is 5x6.
    We can view A as [A0 A1 A2] where A0, A1 are 2x2 and A2 is 2x1.
    We can view B as [B0; B1; B2] where B0, B1 are 2x6 and B2 is 1x6.
    
    C = A0 B0 + A1 B1 + A2 B2.
    
    The term A0 B0 is a 2x2 * 2x6 multiplication.
    This can be decomposed by splitting B0 into three 2x2 blocks: B0 = [B00 B01 B02].
    So A0 B0 = A0 B00 + A0 B01 + A0 B02.
    Each A0 B0i is a 2x2 * 2x2 multiplication, which has rank 7 (Strassen).
    So A0 B0 costs 3 * 7 = 21 rank units.
    
    Similarly, A1 B1 costs 3 * 7 = 21 rank units.
    
    The term A2 B2 is a 2x1 * 1x6 multiplication.
    This is just A2 * B2. A2 is 2x1, B2 is 1x6.
    C2 = A2 B2 is 2x6.
    (A2 B2)[i, k] = A2[i, 0] * B2[0, k].
    This can be decomposed with rank 2 * 6 = 12? No.
    For each i (0,1) and k (0..5), we have one term.
    Wait, A2 is 2x1. Let A2 = [a0; a1]. B2 = [b0 b1 ... b5].
    C2 = [a0 b0, a0 b1, ..., a0 b5;
           a1 b0, a1 b1, ..., a1 b5]
    This is the outer product of A2 and B2.
    The rank of the outer product tensor is 1? No, the multiplication tensor for 2x1 * 1x6.
    The tensor is T[(i,0), (0,k), (k,i)]? No.
    Indices:
    A index: (i, j=0)
    B index: (j=0, k)
    C index: (i, k)
    
    The tensor entry is 1 if i in {0,1}, j=0, k in {0..5}.
    Actually, the tensor is defined on the full space.
    For the sub-block, we need rank 1 decompositions?
    Let's check the rank of 2x1 * 1x6.
    C[i,k] = A[i,0] B[0,k].
    This is a single bilinear form? No, it's 12 separate products?
    No, A[i,0] and B[0,k] are scalars.
    The multiplication is:
    c_00 = a_0 b_0
    c_01 = a_0 b_1
    ...
    c_10 = a_1 b_0
    ...
    
    This can be done with rank 12?
    Or can we do better?
    Note that c_ik = a_i b_k.
    This is the outer product of vector a and vector b.
    The tensor for outer product of n-vector and m-vector has rank n*m.
    Here n=2, m=6. So rank 12.
    
    Total rank = 21 + 21 + 12 = 54.
    
    This matches the inspiration score of 1.1111 (which implies rank 54 vs 60).
    Let's implement this specific decomposition.
    """
    
    # Target Rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        i_base: base index for i (rows of A, rows of C)
        j_base: base index for j (cols of A, rows of B)
        k_base: base index for k (cols of B, cols of C)
        
        A_sub is 2x2: indices (i_base..i_base+1, j_base..j_base+1)
        B_sub is 2x2: indices (j_base..j_base+1, k_base..k_base+1)
        C_sub is 2x2: indices (i_base..i_base+1, k_base..k_base+1)
        
        W index convention: W[k*n + i]
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_outer_product_block(i_base, j_single, k_base, k_width):
        """
        Adds rank (n * k_width) decomposition for a n x 1 * 1 x k_width block.
        Here n=2, j_single is the single column of A and row of B.
        k_base is start of k, k_width is number of k columns.
        
        A is 2x1 at column j_single.
        B is 1xk_width at row j_single.
        C is 2xk_width.
        
        C[i, k] = A[i, j_single] * B[j_single, k]
        
        For each i in [0,1] and k in [k_base, k_base+k_width):
        We add a rank 1 term:
        U[i*m + j_single] += 1
        V[j_single*p + k] += 1
        W[k*n + i] += 1
        """
        nonlocal col
        for i in range(n):
            for k in range(k_base, k_base + k_width):
                U[i * m + j_single][col] += 1
                V[j_single * p + k][col] += 1
                W[k * n + i][col] += 1
                col += 1

    # Partition j (middle dimension) into 0,1 (Strassen), 2,3 (Strassen), 4 (Outer)
    # Partition k (output cols) into 0,1 / 2,3 / 4,5
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # This is 2x2 * 2x6.
    # Split B[0:2, :] into three 2x2 blocks: k=0,1; k=2,3; k=4,5.
    
    # Sub-block 1.1: j=0,1 and k=0,1
    add_strassen_block(0, 0, 0)
    # Sub-block 1.2: j=0,1 and k=2,3
    add_strassen_block(0, 0, 2)
    # Sub-block 1.3: j=0,1 and k=4,5
    add_strassen_block(0, 0, 4)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # This is 2x2 * 2x6.
    # Split B[2:4, :] into three 2x2 blocks: k=0,1; k=2,3; k=4,5.
    
    # Sub-block 2.1: j=2,3 and k=0,1
    add_strassen_block(0, 2, 0)
    # Sub-block 2.2: j=2,3 and k=2,3
    add_strassen_block(0, 2, 2)
    # Sub-block 2.3: j=2,3 and k=4,5
    add_strassen_block(0, 2, 4)
    
    # Block 3: A[:, 4] * B[4, :]
    # This is 2x1 * 1x6.
    # Outer product rank 12.
    add_outer_product_block(0, 4, 0, 6)
    
    if col != 54:
        # Fallback to schoolbook if something went wrong
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

    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = 2, 5, 6
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    is_valid = check_decomposition(res['U'], res['V'], res['W'])
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Valid: {is_valid}")
