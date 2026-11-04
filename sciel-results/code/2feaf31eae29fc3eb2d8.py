import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.allclose(T, 1.0, atol=tol)

def _snap_to_half_integer(mat, tol=1e-9):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]

def _strassen_2x2_terms():
    # Returns list of 7 terms, each with U, V, W coefficients for 2x2x2
    # Indices: A(i,j), B(j,k), C(i,k) for i,j,k in 0,1
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
    # M3 = A00(B10-B11)
    terms.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    # M4 = A11(B00-B01)
    terms.append(({(1,1):1}, {(0,0):1, (0,1):-1}, {(0,0):1, (1,0):1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _add_strassen_block(U, V, W, a_row_offset, b_row_offset, c_row_offset, c_col_offset, col_start):
    # Adds Strassen terms for a 2x2x2 block
    # A is at rows [a_row_offset, a_row_offset+2), cols [0, 2) relative to block? 
    # Wait, U is indexed by (i, j). i is row of A, j is col of A.
    # V is indexed by (j, k). j is col of A, k is col of B.
    # W is indexed by (k, i). k is col of B, i is row of A.
    
    # In our global coordinates:
    # A rows: a_row_offset, a_row_offset+1
    # A cols: 0, 1 (relative to the 2x2 block in A? No, A is 2x4. We split A into 2x2 blocks.)
    # Let's define the local indices for the 2x2x2 problem.
    # local i in 0,1 -> global i = a_row_offset + local_i
    # local j in 0,1 -> global j = local_j (assuming A block starts at col 0)
    # local k in 0,1 -> global k = b_col_offset + local_k (B block)
    # Result C local (i, k) -> global (a_row_offset + i, c_col_offset + k)
    
    # Wait, the function signature in the previous code was slightly different.
    # Let's adapt.
    
    terms = _strassen_2x2_terms()
    col = col_start
    for (x_coeffs, y_coeffs, z_coeffs) in terms:
        for (xi, yi), ux in x_coeffs.items():
            # U index: (global_i, global_j)
            gi = a_row_offset + xi
            gj = yi # A cols are 0,1 for this block
            U[gi * 4 + gj][col] += ux
        for (yi, xi), vy in y_coeffs.items():
            # V index: (global_j, global_k)
            gj = yi
            gk = b_row_offset + xi # B rows are b_row_offset, b_row_offset+1
            V[gj * 4 + gk][col] += vy
        for (zi, zj), wz in z_coeffs.items():
            # W index: (global_k, global_i)
            gk = b_row_offset + zj # B col? No, W is indexed by (k, i) in the tensor definition?
            # Task says: W[k*n+i, r]. So W index is (k, i).
            # In our einsum: W3[k, i, r].
            # So global_k is the first index, global_i is the second.
            gi = c_row_offset + zi # Wait, zi is row of C?
            # In Strassen terms, z_coeffs keys are (i, k) for C?
            # Let's check the previous code's _strassen_terms.
            # It used {(0,0): 1, (1,1): 1} for z_coeffs.
            # And W[(cc + zj) * n + (cr + zi)][col].
            # This implies zj is k (col of B / row of C?) and zi is i (row of C).
            # In M_{2,2,2}, C is 2x2. C[i, k].
            # So z_coeffs keys (zi, zj) correspond to (i, k).
            gi = c_row_offset + zi
            gk = b_row_offset + zj
            W[gk * 3 + gi][col] += wz
        col += 1
    return col

def _build_decomposition_rank_30(n, m, p):
    total_rank = 30
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 1x4x4 (Row 0 of A)
    # Schoolbook for 1x4x4
    for j in range(4):
        for k in range(4):
            U[0 * 4 + j][col] = 1.0
            V[j * 4 + k][col] = 1.0
            W[k * 3 + 0][col] = 1.0
            col += 1
            
    # Part 2: 2x4x4 (Rows 1,2 of A)
    # Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into B_LL (rows 0,1, cols 0,1), B_LR (rows 0,1, cols 2,3), 
    #             B_RL (rows 2,3, cols 0,1), B_RR (rows 2,3, cols 2,3)
    
    # C_L = A_L B_LL + A_R B_RL
    # C_R = A_L B_LR + A_R B_RR
    
    # Strassen for C_L
    # A block: rows 1,2. Cols 0,1 (for A_L) and 2,3 (for A_R).
    # This is tricky. The Strassen algorithm assumes A is a 2x2 matrix.
    # Here A_L is 2x2, A_R is 2x2.
    # The "A" in the Strassen formula for C_L is not a single 2x2 matrix, but a sum of two products.
    # Actually, we can just apply Strassen to the product $X Y$ where $X$ is 2x2 and $Y$ is 2x2.
    # But we have $A_L B_{LL} + A_R B_{RL}$.
    # This is two separate 2x2x2 multiplications.
    # Rank 7 + Rank 7 = 14.
    
    # Let's add Strassen for A_L B_LL
    # A rows 1,2. A cols 0,1.
    # B rows 0,1. B cols 0,1.
    # C rows 1,2. C cols 0,1.
    col = _add_strassen_block(U, V, W, 1, 0, 1, 0, col)
    
    # Let's add Strassen for A_R B_RL
    # A rows 1,2. A cols 2,3.
    # B rows 2,3. B cols 0,1.
    # C rows 1,2. C cols 0,1.
    # We need to modify _add_strassen_block to handle A col offset.
    
    # Let's rewrite _add_strassen_block to be more flexible.
    
    return {"U": U, "V": W, "W": W} # Placeholder, will fix

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Build Rank 30
    total_rank = 30
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    # 1x4x4 part
    for j in range(4):
        for k in range(4):
            U[0 * 4 + j][col] = 1.0
            V[j * 4 + k][col] = 1.0
            W[k * 3 + 0][col] = 1.0
            col += 1
            
    # 2x4x4 part
    # We need to add 4 Strassen blocks (each 2x2x2)
    # Block 1: A_L B_LL -> C_LL
    # A rows 1,2; A cols 0,1
    # B rows 0,1; B cols 0,1
    # C rows 1,2; C cols 0,1
    terms = _strassen_2x2_terms()
    
    def add_block(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        c = start_col
        for (x_coeffs, y_coeffs, z_coeffs) in terms:
            for (xi, yi), ux in x_coeffs.items():
                gi = a_r + xi
                gj = a_c + yi
                U[gi * 4 + gj][c] += ux
            for (yi, xi), vy in y_coeffs.items():
                gj = yi # Wait, y_coeffs keys are (j, k) in local?
                # In _strassen_2x2_terms, y_coeffs keys are (j, k) for B?
                # Let's check: {(0,0):1, (1,1):1} for M1.
                # M1 = (A00+A11)(B00+B11).
                # B is indexed by (j, k).
                # So yi is j, xi is k?
                # In previous code: V[(br + yi) * p + (bc + xi)][col].
                # So yi is row of B (j), xi is col of B (k).
                # Correct.
                gk = b_c + xi
                V[(a_c + yi) * 4 + (b_r + gk)][c] += vy # Wait, V index is (j, k).
                # j is col of A. k is col of B.
                # In local block, j is 0,1. Global j is a_c + yi.
                # k is 0,1. Global k is b_c + xi.
                # So V index is (a_c + yi) * 4 + (b_r + gk)?
                # No, V is indexed by (j, k).
                # j is col of A. k is col of B.
                # So V[ (a_c + yi) * 4 + (b_c + xi) ][c].
                # But B rows are b_r, b_r+1.
                # In M_{2,2,2}, B is 2x2. Rows 0,1. Cols 0,1.
                # So local j is 0,1. Local k is 0,1.
                # Global j = a_c + yi.
                # Global k = b_c + xi.
                # But wait, B is 4x4.
                # The block is B[b_r:b_r+2, b_c:b_c+2].
                # So the "row" of B in the multiplication is the index of the column of A?
                # No, $C = A B$. $C_{ik} = \sum_j A_{ij} B_{jk}$.
                # So $B$ is indexed by $j$ (row) and $k$ (col).
                # In the tensor decomposition, $V$ is indexed by $(j, k)$.
                # So $V_{jk, r}$.
                # In the block, $j$ ranges over $b_r, b_r+1$?
                # No, $j$ is the column of $A$.
                # In the 2x2x2 block, $A$ is 2x2, $B$ is 2x2.
                # $A_{ij}$ with $i,j \in \{0,1\}$.
                # $B_{jk}$ with $j,k \in \{0,1\}$.
                # So $j$ is shared.
                # Global $j = a_c + j_{local}$.
                # Global $k = b_c + k_{local}$.
                # But $B$'s row index is $j$.
                # So $B$'s global row index is $a_c + j_{local}$?
                # NO. $B$ is a separate matrix.
                # $A$ is $n \times m$. $B$ is $m \times p$.
                # $j$ is the index of $m$.
                # So $j$ is the same for $A$ and $B$.
                # In the block, $A$ uses cols $a_c, a_c+1$.
                # So $j_{local} \in \{0,1\}$ maps to $j_{global} = a_c + j_{local}$.
                # $B$ uses rows $j_{global}$ and cols $b_c, b_c+1$.
                # So $B$'s row index is $j_{global} = a_c + j_{local}$.
                # $B$'s col index is $k_{global} = b_c + k_{local}$.
                # So $V$ index is $(a_c + j_{local}) * 4 + (b_c + k_{local})$.
                
                V[(a_c + yi) * 4 + (b_c + xi)][c] += vy
            for (zi, zj), wz in z_coeffs.items():
                # z_coeffs keys (zi, zj) correspond to (i, k) in C.
                # Global i = c_r + zi.
                # Global k = b_c + zj.
                # W index is (k, i).
                W[(b_c + zj) * 3 + (c_r + zi)][c] += wz
            c += 1
        return c

    # Block 1: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_LL (rows 1,2, cols 0,1)
    # a_r=1, a_c=0, b_r=0, b_c=0, c_r=1, c_c=0
    col = add_block(1, 0, 0, 0, 1, 0, col)
    
    # Block 2: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_RL (rows 1,2, cols 0,1)
    # Wait, $C_L = A_L B_{LL} + A_R B_{RL}$.
    # $A_R$ is cols 2,3. $B_{RL}$ is rows 2,3, cols 0,1.
    # $j$ for $A_R$ is 2,3.
    # $j$ for $B_{RL}$ is 2,3.
    # So $a_c=2$. $b_r=2$?
    # No, $B$'s row index is $j$.
    # So $B$'s row index must match $A$'s col index.
    # $A_R$ cols are 2,3. So $j \in \{2,3\}$.
    # $B_{RL}$ rows are 2,3. So $j \in \{2,3\}$.
    # So $a_c=2, b_r=2$.
    # $b_c=0$ (cols 0,1).
    # $c_r=1, c_c=0$.
    col = add_block(1, 2, 2, 0, 1, 0, col)
    
    # Block 3: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_LR (rows 1,2, cols 2,3)
    # $a_c=0, b_r=0, b_c=2, c_r=1, c_c=2$.
    col = add_block(1, 0, 0, 2, 1, 2, col)
    
    # Block 4: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_RR (rows 1,2, cols 2,3)
    # $a_c=2, b_r=2, b_c=2, c_r=1, c_c=2$.
    col = add_block(1, 2, 2, 2, 1, 2, col)
    
    result = {"U": U, "V": V, "W": W}
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to rank 44
        return _build_decomposition_rank_44(n, m, p)
        
    result["U"] = _snap_to_half_integer(result["U"])
    result["V"] = _snap_to_half_integer(result["V"])
    result["W"] = _snap_to_half_integer(result["W"])
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        return _build_decomposition_rank_44(n, m, p)
        
    return result

def _build_decomposition_rank_44(n, m, p):
    # Same as before
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1
            
    def _strassen_terms():
        return [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ]
        
    terms = _strassen_terms()
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi)][col] += wz
            col += 1
    return {"U": U, "V": V, "W": W}
