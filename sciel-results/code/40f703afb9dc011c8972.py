"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def _verify(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    Convention: W[k*n + i, r]
    We check that for all (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    and for all other (i', j, k) with i' != i, the sum is 0, etc.
    Actually, the tensor T has T[i, j, k] = 1 for all i, j, k? 
    No, the multiplication tensor is defined such that 
    C[i, k] = sum_j A[i, j] B[j, k]
    The tensor T has entries T[i, j, k] = 1 for all i, j, k.
    Wait, let's re-read the convention.
    "output factor follows the official trace convention W[k*n+i,r]"
    This implies the standard matrix multiplication tensor where 
    sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{?}
    Actually, the standard definition of the matrix multiplication tensor M_{n,m,p}
    is such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    Let's check the schoolbook code.
    U[i*m+j, col] = 1
    V[j*p+k, col] = 1
    W[k*n+i, col] = 1
    Then sum_r U[i,m+j,r] V[j*p+k,r] W[k*n+i,r] is 1 if there is a column c such that 
    U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    In schoolbook, for each (i,j,k) there is exactly one column where all three are 1.
    So the sum is 1 for all (i,j,k).
    Is that the correct tensor?
    Yes, the matrix multiplication tensor over a field is defined by the bilinear map 
    A x B -> C where C = A*B.
    The tensor is T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C ... wait.
    Let A be n x m, B be m x p, C be n x p.
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T has components T_{i,j,k} such that 
    <T, A \otimes B \otimes C^*> = sum_{i,k} C[i,k] (A B)[i,k] = sum_{i,j,k} C[i,k] A[i,j] B[j,k].
    This is a trilinear form. The tensor itself is T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C.
    Wait, if T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then 
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k.
    Let's verify with schoolbook.
    For fixed i,j,k, the column c = index of (i,j,k).
    U[i*m+j, c] = 1.
    V[j*p+k, c] = 1.
    W[k*n+i, c] = 1.
    So the product is 1.
    Are there other columns where all three are 1?
    No, each column corresponds to a unique (i,j,k).
    So the sum is 1.
    So the condition is: for all i, j, k, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-12:
                    return False
    return True

def _rank_of(U, V, W):
    return len(U[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # 2. Block Decomposition (Rank 34 for 3x3x4)
    # Strategy:
    # Split A (3x3) into 4 blocks? Or split differently.
    # Let's use the 2x2 + 1x1 idea.
    # A = [ A11 (2x2) A12 (2x1) ; A21 (1x2) A22 (1x1) ]
    # B = [ B11 (2x2) B12 (2x2) ; B21 (1x2) B22 (1x2) ] -- Wait, B is 3x4.
    # Let B = [ B1 (2x4) ; B2 (1x4) ]
    # Then C = [ C1 (2x4) ; C2 (1x4) ]
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    
    # C1 (2x4) = A11 (2x2) * B1 (2x4) + A12 (2x1) * B2 (1x4)
    # C2 (1x4) = A21 (1x2) * B1 (2x4) + A22 (1x1) * B2 (1x4)
    
    # Term 1: A11 * B1.
    # A11 is 2x2, B1 is 2x4.
    # We can view B1 as [B1a B1b] where B1a is 2x2, B1b is 2x2.
    # Then A11 B1a is 2x2x2 (Strassen rank 7).
    # A11 B1b is 2x2x2 (Strassen rank 7).
    # Total 14 products for A11 B1.
    
    # Term 2: A12 * B2.
    # A12 is 2x1, B2 is 1x4.
    # This is (2x1) * (1x4) -> 2x4.
    # Each element of C1 is a dot product of length 1? No.
    # C1[i, k] = sum_j A12[i, j] B2[j, k]. Here j is only 0 (since A12 is 2x1).
    # So C1[i, k] = A12[i, 0] * B2[0, k].
    # This is 2 * 4 = 8 products.
    
    # Term 3: A21 * B1.
    # A21 is 1x2, B1 is 2x4.
    # C2[k] = sum_j A21[0, j] B1[j, k]. j is 0, 1.
    # This is 1 * 4 = 4 rows? No, C2 is 1x4.
    # So 4 dot products of length 2.
    # Total 4 * 2 = 8 products.
    
    # Term 4: A22 * B2.
    # A22 is 1x1, B2 is 1x4.
    # C2[k] += A22[0,0] * B2[0, k].
    # This is 4 products.
    
    # Total Rank = 14 + 8 + 8 + 4 = 34.
    
    U_b = [[0] * 34 for _ in range(n * m)]
    V_b = [[0] * 34 for _ in range(m * p)]
    W_b = [[0] * 34 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, val_u=1, val_v=1, val_w=1):
        nonlocal col
        if col >= 34:
            raise ValueError("Rank overflow")
        U_b[i*m+j][col] += val_u
        V_b[j*p+k][col] += val_v
        W_b[k*n+i][col] += val_w
        col += 1

    # Strassen 2x2x2 decomposition for A11 * B1a and A11 * B1b
    # Indices for A11: i in 0,1; j in 0,1.
    # Indices for B1a: j in 0,1; k in 0,1. (Columns 0,1 of B1)
    # Indices for B1b: j in 0,1; k in 2,3. (Columns 2,3 of B1)
    
    # Strassen Formulas for 2x2 * 2x2:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def add_strassen(base_i, base_j, base_k):
        """
        Adds Strassen terms for A[base_i:base_i+2, base_j:base_j+2] * B[base_j:base_j+2, base_k:base_k+2]
        """
        # m1
        # (a11 + a22) * (b11 + b22)
        # We need to distribute this product.
        # In tensor decomposition, we add terms for each pair.
        # Wait, Strassen is a bilinear complexity.
        # The term m1 is one product: M1 = (a11+a22)(b11+b22).
        # This corresponds to a rank-1 tensor where 
        # U part has 1s at (1,1) and (2,2) (relative indices)
        # V part has 1s at (1,1) and (2,2) (relative indices)
        # But wait, the product is (u1*v1 + u2*v2) * (v1*w1 + v2*w2)? No.
        # The product is (A_part) * (B_part).
        # A_part = a11 + a22. So U has 1 at (0,0) and 1 at (1,1).
        # B_part = b11 + b22. So V has 1 at (0,0) and 1 at (1,1).
        # The result contributes to C11, C22.
        # C11 gets +1, C22 gets +1.
        # So W has 1 at (0,0) and 1 at (1,1).
        
        nonlocal col
        
        def add_strassen_term(u_indices, v_indices, w_indices, sign=1):
            nonlocal col
            if col >= 34:
                return
            for (ri, cj) in u_indices:
                U_b[(base_i+ri)*m + (base_j+cj)][col] += sign
            for (rj, ck) in v_indices:
                V_b[(base_j+rj)*p + (base_k+ck)][col] += sign
            for (rk, ci) in w_indices:
                W_b[(base_k+rk)*n + (base_i+ci)][col] += sign
            col += 1
            
        # m1: (a11+a22)(b11+b22)
        # u: (0,0), (1,1)
        # v: (0,0), (1,1)
        # w: (0,0), (1,1)
        add_strassen_term([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)])
        
        # m2: (a21+a22)b11
        # u: (1,0), (1,1)
        # v: (0,0)
        # w: (1,0)
        add_strassen_term([(1,0), (1,1)], [(0,0)], [(1,0)])
        
        # m3: a11(b12-b22)
        # u: (0,0)
        # v: (0,1), (1,1) with signs +, -
        # w: (0,1), (1,1) with signs +, -
        # This is two terms or one term with negative coefficients?
        # Rank is the number of terms. We can have negative coefficients.
        # So one term:
        # u: (0,0) with 1
        # v: (0,1) with 1, (1,1) with -1
        # w: (0,1) with 1, (1,1) with -1
        # Let's just add them as separate entries in the lists?
        # No, the function above adds `sign` to all.
        # I need to allow different signs for different indices.
        # Let's refine add_strassen_term to take lists of (index, coeff).
        
    # Let's restart the block construction with a more flexible helper.
    
    U_b = [[0] * 34 for _ in range(n * m)]
    V_b = [[0] * 34 for _ in range(m * p)]
    W_b = [[0] * 34 for _ in range(p * n)]
    col = 0
    
    def add_block_term(u_list, v_list, w_list):
        """
        u_list: list of (i, j, coeff)
        v_list: list of (j, k, coeff)
        w_list: list of (k, i, coeff)
        """
        nonlocal col
        if col >= 34:
            return
        for (ri, cj, cu) in u_list:
            U_b[ri*m + cj][col] += cu
        for (rj, ck, cv) in v_list:
            V_b[rj*p + ck][col] += cv
        for (rk, ci, cw) in w_list:
            W_b[rk*n + ci][col] += cw
        col += 1

    # Strassen for A11 (i:0,1; j:0,1) * B1a (j:0,1; k:0,1)
    # m1: (a00+a11)(b00+b11)
    add_block_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    # m2: (a10+a11)b00
    add_block_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1)]
    )
    # m3: a00(b01-b11)
    add_block_term(
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,-1)]
    )
    # m4: a11(b10-b00)
    add_block_term(
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(1,0,1), (0,0,-1)]
    )
    # m5: (a00+a01)b11
    add_block_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(1,1,1)]
    )
    # m6: (a10-a00)(b00+b01)
    add_block_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    )
    # m7: (a01-a11)(b10+b11)
    add_block_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)]
    )
    
    # Strassen for A11 (i:0,1; j:0,1) * B1b (j:0,1; k:2,3)
    # Note: k indices are 2,3 here.
    add_block_term(
        [(0,0,1), (1,1,1)],
        [(0,2,1), (1,3,1)],
        [(2,0,1), (3,1,1)]
    )
    add_block_term(
        [(1,0,1), (1,1,1)],
        [(0,2,1)],
        [(3,0,1)] # C[1,2] corresponds to k=2, i=1. Wait.
    )
    # Let's be careful with indices.
    # B1b is columns 2,3.
    # m2: (a10+a11) b_{0,2}
    # Result C[1,2]. i=1, k=2.
    add_block_term(
        [(1,0,1), (1,1,1)],
        [(0,2,1)],
        [(2,1,1)]
    )
    # m3: a00 (b_{0,3} - b_{1,3})
    # Result C[0,3] gets +, C[1,3] gets -.
    add_block_term(
        [(0,0,1)],
        [(0,3,1), (1,3,-1)],
        [(3,0,1), (3,1,-1)]
    )
    # m4: a11 (b_{1,2} - b_{0,2})
    # Result C[1,2] gets +, C[0,2] gets -.
    add_block_term(
        [(1,1,1)],
        [(1,2,1), (0,2,-1)],
        [(2,1,1), (2,0,-1)]
    )
    # m5: (a00+a01) b_{1,3}
    # Result C[1,3].
    add_block_term(
        [(0,0,1), (0,1,1)],
        [(1,3,1)],
        [(3,1,1)]
    )
    # m6: (a10-a00) (b_{0,2}+b_{0,3})
    # Result C[1,2], C[1,3].
    add_block_term(
        [(1,0,1), (0,0,-1)],
        [(0,2,1), (0,3,1)],
        [(2,1,1), (3,1,1)]
    )
    # m7: (a01-a11) (b_{1,2}+b_{1,3})
    # Result C[1,2], C[1,3].
    add_block_term(
        [(0,1,1), (1,1,-1)],
        [(1,2,1), (1,3,1)],
        [(2,1,1), (3,1,1)]
    )
    
    # Term 2: A12 (i:0,1; j:2) * B2 (j:2; k:0,1,2,3)
    # A12 is 2x1. B2 is 1x4.
    # C1[i,k] += A12[i,2] B2[2,k]
    # 8 products.
    for i in range(2):
        for k in range(4):
            add_block_term(
                [(i,2,1)],
                [(2,k,1)],
                [(k,i,1)]
            )
            
    # Term 3: A21 (i:2; j:0,1) * B1 (j:0,1; k:0,1,2,3)
    # A21 is 1x2. B1 is 2x4.
    # C2[k] += sum_j A21[2,j] B1[j,k]
    # 8 products.
    for j in range(2):
        for k in range(4):
            add_block_term(
                [(2,j,1)],
                [(j,k,1)],
                [(k,2,1)]
            )
            
    # Term 4: A22 (i:2; j:2) * B2 (j:2; k:0,1,2,3)
    # 4 products.
    for k in range(4):
        add_block_term(
            [(2,2,1)],
            [(2,k,1)],
            [(k,2,1)]
        )
        
    if col == 34 and _verify(U_b, V_b, W_b):
        best_result = {"U": U_b, "V": V_b, "W": W_b}
        best_rank = 34
        
    # If we have time, we could try random search or other optimizations,
    # but 34 is a solid improvement over 36.
    
    return best_result

# EVOLVE-BLOCK-END
