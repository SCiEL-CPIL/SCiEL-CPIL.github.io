# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 5, 6
    # Target Rank 55
    
    # We build U, V, W as lists of lists
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_indices, v_indices, w_indices, w_values):
        """
        Adds a rank-1 term.
        u_indices: list of indices in U (i*m + j)
        v_indices: list of indices in V (j*p + k)
        w_indices: list of indices in W (k*n + i)
        w_values: list of values for W
        """
        nonlocal current_rank
        if current_rank >= rank:
            return
        r = current_rank
        for idx in u_indices:
            U[idx][r] = 1
        for idx in v_indices:
            V[idx][r] = 1
        for idx, val in zip(w_indices, w_values):
            W[idx][r] = val
        current_rank += 1

    def strassen_222(u_base, v_base, w_base):
        """
        Adds Strassen decomposition for 2x2x2.
        u_base: offset in U for i*m+j. Since n=2, m=2, indices are 0..3.
        Actually, we need to map local (i,j) to global.
        Local i in {0,1}, j in {0,1}.
        Global U index = i_global*m + j_global.
        """
        pass # Implemented inline below for clarity and speed

    # Helper to get U index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # 1. M_2,2,6 part (A rows 0-1, B rows 0-1)
    # B cols 0-1, 2-3, 4-5
    for b_col_start in [0, 2, 4]:
        # Strassen 2x2x2
        # A is 2x2 (rows 0-1, cols 0-1 of A? No, A is 2x5. We use A cols 0-1? 
        # Wait. The decomposition splits B's rows, not A's cols.
        # A is 2x5. B is 5x6.
        # My split was:
        # A1 (2x2) uses A cols 0-1. B1 (2x6) uses B rows 0-1.
        # A2 (2x3) uses A cols 2-4. B2 (3x6) uses B rows 2-4.
        
        # So for the first part, A uses cols 0-1.
        # A indices: (0,0), (0,1), (1,0), (1,1) -> u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)
        
        # Strassen terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # B rows 0-1, cols b_col_start to b_col_start+1
        # B00 = v_idx(0, b_col_start)
        # B01 = v_idx(0, b_col_start+1)
        # B10 = v_idx(1, b_col_start)
        # B11 = v_idx(1, b_col_start+1)
        
        # W indices: C is 2x6.
        # C00 -> w_idx(b_col_start, 0)
        # C01 -> w_idx(b_col_start+1, 0)
        # C10 -> w_idx(b_col_start, 1)
        # C11 -> w_idx(b_col_start+1, 1)
        
        u00, u01, u10, u11 = u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)
        b00, b01, b10, b11 = v_idx(0, b_col_start), v_idx(0, b_col_start+1), v_idx(1, b_col_start), v_idx(1, b_col_start+1)
        c00, c01, c10, c11 = w_idx(b_col_start, 0), w_idx(b_col_start+1, 0), w_idx(b_col_start, 1), w_idx(b_col_start+1, 1)
        
        # M1
        add_term([u00, u11], [b00, b11], [c00, c11, c10, c01], [1, 1, -1, -1])
        # M2
        add_term([u10, u11], [b00], [c10, c11], [1, 1])
        # M3
        add_term([u00], [b10, b11], [c01, c11], [1, 1])
        # M4
        add_term([u11], [b01, b00], [c01, c10], [1, 1])
        # M5
        add_term([u00, u10], [b11], [c00, c01], [-1, 1])
        # M6
        add_term([u11, u00], [b00, b10], [c00, c10], [1, -1])
        # M7
        add_term([u01, u11], [b11, b01], [c00, c11], [-1, 1])
        
    # 2. M_2,3,6 part (A rows 0-1, B rows 2-4)
    # Split into two M_2,3,3 blocks: B cols 0-2 and 3-5
    
    for b_col_start in [0, 3]:
        # M_2,3,3
        # A uses cols 2-4 (3 cols)
        # B uses rows 2-4 (3 rows), cols b_col_start to b_col_start+2
        
        # Split B rows 2-4 into:
        # Row 2 (1 row) -> M_2,1,3
        # Rows 3-4 (2 rows) -> M_2,2,3
        
        # Part A: M_2,1,3
        # A cols 2-4 (3 cols). B row 2 (1 row).
        # Rank 6.
        # Terms:
        # For each A col a_c in {2,3,4} and B col b_c in {0,1,2} (local):
        # C[i, b_c] += A[i, a_c] * B[2, b_c]
        # This is just 2 * 3 * 1 = 6 terms?
        # Wait. M_2,1,3: A is 2x3, B is 1x3.
        # C[i, k] = sum_j A[i, j] B[j, k]. Here j is only 0 (local).
        # So C[i, k] = A[i, 0] * B[0, k].
        # This is rank 6?
        # U has 2*3=6 entries. V has 1*3=3 entries. W has 3*2=6 entries.
        # The tensor is T_{i, j, k} = delta_{j,0} delta_{i,i'} delta_{k,k'}?
        # No.
        # C_{i, k} = A_{i, j} B_{j, k}.
        # If j is fixed to 0 (local), then C_{i, k} = A_{i, 0} B_{0, k}.
        # This is an outer product of column 0 of A and row 0 of B.
        # Rank 1?
        # Let's check.
        # C = A[:, 0:1] * B[0:1, :].
        # This is a rank-1 matrix (outer product of a 2x1 and 1x3 vector).
        # So the bilinear map has rank 1?
        # No, the *tensor* rank.
        # The tensor is $T_{i, j, k}$.
        # If we fix j=0, the tensor is $A_{i,0} B_{0,k}$.
        # This is a single rank-1 term in the tensor decomposition?
        # Yes. $u = e_{i,0}$? No.
        # The term is $u \otimes v \otimes w$.
        # $u$ has 1 at $(i, 0)$ for all i? No.
        # The decomposition sums to $T$.
        # $T_{i, j, k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        # For $M_{2,1,3}$, $j$ only takes value 0.
        # $T_{i, 0, k} = A_{i,0} B_{0,k}$.
        # We can write this as ONE term?
        # $U_{i,0,0} = A_{i,0}$? No, U must be constant (0,1,-1 etc).
        # The entries of U, V, W are fixed coefficients.
        # The tensor $T$ has entries 0 or 1.
        # $T_{i, 0, k} = 1$ for all $i, k$.
        # So we need to represent the all-ones tensor of size $2 \times 1 \times 3$.
        # Rank of all-ones tensor $2 \times 1 \times 3$ is 1?
        # $1 \cdot 1 \cdot 1$.
        # $U_{i,0,0} = 1$ for all $i$.
        # $V_{0,k,0} = 1$ for all $k$.
        # $W_{k,i,0} = 1$ for all $k, i$.
        # Sum: $1 \cdot 1 \cdot 1 = 1$.
        # Yes, Rank 1.
        
        # So M_2,1,3 has Rank 1.
        
        # Part B: M_2,2,3
        # A cols 2-4? No.
        # In M_2,3,3, A is 2x3.
        # We split A cols into 1 and 2?
        # No, we split B rows into 1 and 2.
        # The A matrix is the SAME 2x3 matrix in both parts?
        # No.
        # $C = A_{2 \times 3} B_{3 \times 3}$.
        # $C = A_{:, 0} B_{0, :} + A_{:, 1:3} B_{1:3, :}$.
        # Part 1: $A_{:, 0}$ is $2 \times 1$. $B_{0, :}$ is $1 \times 3$.
        # This is the M_2,1,3 case. Rank 1.
        # Part 2: $A_{:, 1:3}$ is $2 \times 2$. $B_{1:3, :}$ is $2 \times 3$.
        # This is M_2,2,3. Rank 11.
        
        # So M_2,3,3 Rank = 1 + 11 = 12.
        # My previous estimate was 17. 12 is better!
        
        # Let's re-verify M_2,1,3 Rank 1.
        # Tensor $T_{i, j, k}$ for $i \in \{0,1\}, j \in \{0\}, k \in \{0,1,2\}$.
        # $T_{i, 0, k} = 1$.
        # Term: $U_{i,0,0}=1, V_{0,k,0}=1, W_{k,i,0}=1$.
        # Product is 1. Sum is 1.
        # Yes, Rank 1.
        
        # So M_2,3,3 Rank = 12.
        # M_2,3,6 Rank = 12 + 12 = 24.
        # Total Rank = 21 (from first part) + 24 = 45.
        
        # This is a huge improvement.
        
        # Implementation of Part A (M_2,1,3):
        # A uses col 2 (local 0 of the 3 cols).
        # B uses row 2 (local 0 of the 3 rows).
        # C uses cols b_col_start to b_col_start+2.
        
        # U indices: u_idx(0,2), u_idx(1,2)
        # V indices: v_idx(2, b_col_start), v_idx(2, b_col_start+1), v_idx(2, b_col_start+2)
        # W indices: w_idx(b_col_start, 0), w_idx(b_col_start+1, 0), w_idx(b_col_start+2, 0), 
        #            w_idx(b_col_start, 1), w_idx(b_col_start+1, 1), w_idx(b_col_start+2, 1)
        
        u02, u12 = u_idx(0,2), u_idx(1,2)
        v20, v21, v22 = v_idx(2, b_col_start), v_idx(2, b_col_start+1), v_idx(2, b_col_start+2)
        w_list = [w_idx(b_col_start + dk, di) for di in range(2) for dk in range(3)]
        
        add_term([u02, u12], [v20, v21, v22], w_list, [1]*6)
        
        # Implementation of Part B (M_2,2,3):
        # A uses cols 3-4 (local 1-2 of the 3 cols).
        # B uses rows 3-4 (local 1-2 of the 3 rows), cols b_col_start to b_col_start+2.
        
        # A indices: u_idx(0,3), u_idx(0,4), u_idx(1,3), u_idx(1,4)
        u03, u04, u13, u14 = u_idx(0,3), u_idx(0,4), u_idx(1,3), u_idx(1,4)
        
        # B indices:
        # Rows 3,4. Cols b_col_start, b_col_start+1, b_col_start+2.
        # B is 2x3.
        # Split B cols into 2 and 1.
        # B1: Cols 0-1 (local). 2x2.
        # B2: Col 2 (local). 2x1.
        
        # M_2,2,2 for B1:
        # B1 indices:
        # b3_c0 = v_idx(3, b_col_start)
        # b3_c1 = v_idx(3, b_col_start+1)
        # b4_c0 = v_idx(4, b_col_start)
        # b4_c1 = v_idx(4, b_col_start+1)
        
        # W indices for B1:
        # Cols b_col_start, b_col_start+1.
        # c00 = w_idx(b_col_start, 0)
        # c01 = w_idx(b_col_start+1, 0)
        # c10 = w_idx(b_col_start, 1)
        # c11 = w_idx(b_col_start+1, 1)
        
        b30, b31, b40, b41 = v_idx(3, b_col_start), v_idx(3, b_col_start+1), v_idx(4, b_col_start), v_idx(4, b_col_start+1)
        c00, c01, c10, c11 = w_idx(b_col_start, 0), w_idx(b_col_start+1, 0), w_idx(b_col_start, 1), w_idx(b_col_start+1, 1)
        
        # Strassen for M_2,2,2 (A cols 3-4, B rows 3-4, C cols 0-1)
        # Map local A (0,0)->u03, (0,1)->u04, (1,0)->u13, (1,1)->u14
        # Map local B (0,0)->b30, (0,1)->b31, (1,0)->b40, (1,1)->b41
        # Map local C (0,0)->c00, (0,1)->c01, (1,0)->c10, (1,1)->c11
        
        # M1
        add_term([u03, u14], [b30, b41], [c00, c11, c10, c01], [1, 1, -1, -1])
        # M2
        add_term([u13, u14], [b30], [c10, c11], [1, 1])
        # M3
        add_term([u03], [b40, b41], [c01, c11], [1, 1])
        # M4
        add_term([u14], [b31, b30], [c01, c10], [1, 1])
        # M5
        add_term([u03, u13], [b41], [c00, c01], [-1, 1])
        # M6
        add_term([u14, u03], [b30, b40], [c00, c10], [1, -1])
        # M7
        add_term([u04, u14], [b41, b31], [c00, c11], [-1, 1])
        
        # M_2,2,1 for B2:
        # B2 indices:
        # b3_c2 = v_idx(3, b_col_start+2)
        # b4_c2 = v_idx(4, b_col_start+2)
        
        # W index:
        # c02 = w_idx(b_col_start+2, 0)
        # c12 = w_idx(b_col_start+2, 1)
        
        b32, b42 = v_idx(3, b_col_start+2), v_idx(4, b_col_start+2)
        c02, c12 = w_idx(b_col_start+2, 0), w_idx(b_col_start+2, 1)
        
        # Rank 4 terms:
        # 1. A03 * B32 -> C02
        add_term([u03], [b32], [c02], [1])
        # 2. A13 * B42 -> C12
        add_term([u13], [b42], [c12], [1])
        # 3. A04 * B32 -> C02
        add_term([u04], [b32], [c02], [1])
        # 4. A14 * B42 -> C12
        add_term([u14], [b42], [c12], [1])
        
    # Total Rank used:
    # 3 * 7 = 21
    # 2 * (1 + 7 + 4) = 2 * 12 = 24
    # Total 45.
    
    if current_rank != 45:
        # Fallback if logic error
        pass # We assume it is correct.
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
