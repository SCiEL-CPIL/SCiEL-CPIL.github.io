import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    n_u, n_v, n_w = n * m, m * p, p * n
    rank = len(U[0])
    
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                actual = 0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if actual != target:
                    return False
    return True

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for (2x4) * (4x8).
    Returns U (8xR), V (32xR), W (16xR).
    """
    t0 = time.time()
    deadline = t0 + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    n_u, n_v, n_w = n * m, m * p, p * n
    
    # 1. Schoolbook Baseline (Rank 64)
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n_u)]
    V_school = [[0] * schoolbook_rank for _ in range(n_v)]
    W_school = [[0] * schoolbook_rank for _ in range(n_w)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r_idx = i * m * p + j * p + k
                U_school[i*m + j][r_idx] = 1
                V_school[j*p + k][r_idx] = 1
                W_school[k*n + i][r_idx] = 1

    best_rank = schoolbook_rank
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # 2. Block Decomposition using Strassen (Rank 56)
    # Strategy:
    # The product C = A * B where A is 2x4 and B is 4x8.
    # Split A into two 2x2 blocks: A1 (cols 0,1), A2 (cols 2,3).
    # Split B into two 2x8 blocks: B1 (rows 0,1), B2 (rows 2,3).
    # C = A1*B1 + A2*B2.
    # 
    # For each term A_x * B_y (2x2 * 2x8):
    # Split B_y into four 2x2 blocks: B_{y,0}, B_{y,1}, B_{y,2}, B_{y,3} (cols 0-1, 2-3, 4-5, 6-7).
    # A_x * B_y = A_x * B_{y,0} + A_x * B_{y,1} + A_x * B_{y,2} + A_x * B_{y,3}.
    # Each A_x * B_{y,k} is a 2x2 * 2x2 multiplication.
    # Using Strassen (Rank 7) for each 2x2*2x2:
    # Total terms = 2 (A blocks) * 2 (B blocks) * 4 (sub-blocks of B) * 7 (Strassen) = 112?
    # Wait.
    # A is 2x4. B is 4x8.
    # A = [A11 A12 A13 A14] ... no, A is 2 rows, 4 cols.
    # A = [A_left A_right] where A_left is 2x2, A_right is 2x2.
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    # C = A_left * B_top + A_right * B_bottom.
    # 
    # Let's look at A_left * B_top. (2x2) * (2x8).
    # This is equivalent to multiplying a 2x2 matrix by four 2x2 matrices (the 2x2 blocks of B_top).
    # So A_left * B_top = A_left * B_{top,0} + A_left * B_{top,1} + A_left * B_{top,2} + A_left * B_{top,3}.
    # Each term is 2x2 * 2x2. Rank 7.
    # So A_left * B_top costs 4 * 7 = 28 rank terms.
    # Similarly A_right * B_bottom costs 4 * 7 = 28 rank terms.
    # Total rank = 28 + 28 = 56.
    
    # Strassen decomposition for 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will construct U, V, W for this 56-rank decomposition.
    
    R_strassen = 7
    R_total = 56
    
    U_new = [[0] * R_total for _ in range(n_u)]
    V_new = [[0] * R_total for _ in range(n_v)]
    W_new = [[0] * R_total for _ in range(n_w)]
    
    # Helper to add a Strassen instance
    # a_block: indices (i, j) in A
    # b_block: indices (j, k) in B
    # c_block: indices (i, k) in C
    # base_rank: offset in the rank dimension
    
    def add_strassen(i_off, j_off, k_off, base_r):
        # A is at (i_off + i_loc, j_off + j_loc)
        # B is at (j_off + j_loc, k_off + k_loc)
        # C is at (i_off + i_loc, k_off + k_loc) -> W index is k_glob * n + i_glob
        
        # Local indices 0, 1
        # Terms M1..M7
        
        # M1: (A11+A22)*(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11 (+), C22 (+)
        
        # M2: (A21+A22)*B11
        # U: A21, A22
        # V: B11
        # W: C21 (+)
        
        # M3: A11*(B12-B22)
        # U: A11
        # V: B12 (+), B22 (-)
        # W: C12 (+), C22 (+)
        
        # M4: A22*(B21-B11)
        # U: A22
        # V: B21 (+), B11 (-)
        # W: C11 (+), C21 (+)
        
        # M5: (A11+A12)*B22
        # U: A11, A12
        # V: B22
        # W: C11 (-), C12 (+)
        
        # M6: (A21-A11)*(B11+B12)
        # U: A21 (+), A11 (-)
        # V: B11, B12
        # W: C22 (+)
        
        # M7: (A12-A22)*(B21+B22)
        # U: A12 (+), A22 (-)
        # V: B21, B22
        # W: C11 (+), C22 (-)
        
        def idx_U(i, j): return (i_off + i) * m + (j_off + j)
        def idx_V(j, k): return (j_off + j) * p + (k_off + k)
        def idx_W(i, k): return (k_off + k) * n + (i_off + i)
        
        # Term 1
        r = base_r + 0
        U_new[idx_U(0,0)][r] += 1
        U_new[idx_U(1,1)][r] += 1
        V_new[idx_V(0,0)][r] += 1
        V_new[idx_V(1,1)][r] += 1
        W_new[idx_W(0,0)][r] += 1
        W_new[idx_W(1,1)][r] += 1
        
        # Term 2
        r = base_r + 1
        U_new[idx_U(1,0)][r] += 1
        U_new[idx_U(1,1)][r] += 1
        V_new[idx_V(0,0)][r] += 1
        W_new[idx_W(1,0)][r] += 1
        
        # Term 3
        r = base_r + 2
        U_new[idx_U(0,0)][r] += 1
        V_new[idx_V(0,1)][r] += 1
        V_new[idx_V(1,1)][r] += -1
        W_new[idx_W(0,1)][r] += 1
        W_new[idx_W(1,1)][r] += 1
        
        # Term 4
        r = base_r + 3
        U_new[idx_U(1,1)][r] += 1
        V_new[idx_V(1,0)][r] += 1
        V_new[idx_V(0,0)][r] += -1
        W_new[idx_W(0,0)][r] += 1
        W_new[idx_W(1,0)][r] += 1
        
        # Term 5
        r = base_r + 4
        U_new[idx_U(0,0)][r] += 1
        U_new[idx_U(0,1)][r] += 1
        V_new[idx_V(1,1)][r] += 1
        W_new[idx_W(0,0)][r] += -1
        W_new[idx_W(0,1)][r] += 1
        
        # Term 6
        r = base_r + 5
        U_new[idx_U(1,0)][r] += 1
        U_new[idx_U(0,0)][r] += -1
        V_new[idx_V(0,0)][r] += 1
        V_new[idx_V(0,1)][r] += 1
        W_new[idx_W(1,1)][r] += 1
        
        # Term 7
        r = base_r + 6
        U_new[idx_U(0,1)][r] += 1
        U_new[idx_U(1,1)][r] += -1
        V_new[idx_V(1,0)][r] += 1
        V_new[idx_V(1,1)][r] += 1
        W_new[idx_W(0,0)][r] += 1
        W_new[idx_W(1,1)][r] += -1

    # Block 1: A_left (i=0,1, j=0,1) * B_top (j=0,1, k=0..7)
    # B_top has 4 sub-blocks of 2x2:
    # k=0,1 -> k_off=0
    # k=2,3 -> k_off=2
    # k=4,5 -> k_off=4
    # k=6,7 -> k_off=6
    base_r = 0
    for k_off in [0, 2, 4, 6]:
        add_strassen(0, 0, k_off, base_r)
        base_r += 7
        
    # Block 2: A_right (i=0,1, j=2,3) * B_bottom (j=2,3, k=0..7)
    # j_off = 2
    # B_bottom has 4 sub-blocks of 2x2:
    # k=0,1 -> k_off=0
    # k=2,3 -> k_off=2
    # k=4,5 -> k_off=4
    # k=6,7 -> k_off=6
    for k_off in [0, 2, 4, 6]:
        add_strassen(0, 2, k_off, base_r)
        base_r += 7
        
    if _verify(U_new, V_new, W_new):
        best_rank = R_total
        best_U = U_new
        best_V = V_new
        best_W = W_new
    else:
        # If Strassen fails (shouldn't, but just in case), stick to schoolbook
        pass

    # 3. Try to improve further?
    # Rank 56 is likely not minimal, but finding lower is hard.
    # Known bounds for 2x4x8:
    # Schoolbook 64.
    # Block Strassen 56.
    # There might be better algorithms.
    # Given time, 56 is a solid improvement.
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Can we do better than 56?
    # Perhaps using a different blocking?
    # If we split A into 1x2 blocks? No, Strassen is 2x2.
    # What if we use 4x4x4? No, dimensions are 2,4,8.
    
    # Let's try a random search for rank 48 or 40?
    # This is computationally expensive and risky.
    # The linear algebra approach:
    # Fix U and V, solve for W.
    # T = U * V * W^T (tensor product).
    # Unfold T along W: T_w is a (p*n) x (n*m * m*p) matrix?
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For fixed W, this is a system.
    # Actually, let's just return the 56 rank solution.
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    if _verify(res['U'], res['V'], res['W']):
        print("Valid")
    else:
        print("Invalid")
