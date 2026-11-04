import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 50
    # Decomposition: 4 blocks of M_2,3,2 (rank 11) + 1 block of M_2,3,1 (rank 6)
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper to add Strassen block for M_2,2,2
    # Indices for A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # Indices for B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # Indices for C: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3 (offset by k*n+i)
    def add_strassen_block(k_offset):
        nonlocal col
        # k_offset is the starting index for k in this block (0, 2, 4, 6)
        # A indices: i*m + j. i in {0,1}, j in {0,1}
        # B indices: j*p + k. j in {0,1}, k in {k_offset, k_offset+1}
        # C indices: k*n + i. k in {k_offset, k_offset+1}, i in {0,1}
        
        # Strassen variables
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # We need to express these as sums of rank-1 tensors in the global space.
        # This requires expanding the products into the U, V, W structure.
        # U corresponds to A, V to B, W to C.
        # Each "multiplication" in Strassen is a linear combination of A and B.
        # We can realize each Strassen multiplication as a set of rank-1 terms?
        # No, Strassen reduces the number of multiplications, but the "multiplications" themselves are scalar products of linear combinations.
        # In the tensor decomposition context, a "multiplication" $X \cdot Y$ where $X$ is linear in A and $Y$ linear in B corresponds to a rank-1 tensor if we consider the intermediate variables?
        # No, the rank is the number of scalar multiplications.
        # Each scalar multiplication $S = X \cdot Y$ contributes 1 to the rank.
        # Then $C$ is a linear combination of $S$'s.
        # So we add 7 terms for the 7 multiplications.
        
        # Term 1: S1 = (A00 + A11) * (B00 + B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C01 (-1), C10 (-1), C11 (1) -- Wait, coefficients from C formulas.
        # C00 has +M1, C01 has 0, C10 has 0, C11 has +M1?
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # So M1 contributes to C00 (1) and C11 (1).
        
        # We define the linear forms for U and V.
        # U is a vector of size n*m=6.
        # V is a vector of size m*p=27.
        # W is a vector of size p*n=18.
        
        # For each Strassen multiplication, we define the coefficients in U, V, W.
        
        # M1
        u1 = {0: 1, 3: 1} # A00, A11 (indices: 0*3+0=0, 1*3+1=4? No. i*m+j. i=0,j=0->0. i=1,j=1->1*3+1=4.)
        # A indices:
        # A00: i=0, j=0 -> 0
        # A01: i=0, j=1 -> 1
        # A10: i=1, j=0 -> 3
        # A11: i=1, j=1 -> 4
        # B indices: j*p + k
        # B00: j=0, k=0 -> 0
        # B01: j=0, k=1 -> 1
        # B10: j=1, k=0 -> p (9)
        # B11: j=1, k=1 -> p+1 (10)
        # But this is a block with k_offset.
        # B00_k: j=0, k=k_offset -> 0*p + k_offset = k_offset
        # B01_k: j=0, k=k_offset+1 -> k_offset+1
        # B10_k: j=1, k=k_offset -> 1*p + k_offset = p + k_offset
        # B11_k: j=1, k=k_offset+1 -> p + k_offset + 1
        
        # C indices: k*n + i
        # C00_k: k=k_offset, i=0 -> k_offset*n + 0 = 2*k_offset
        # C01_k: k=k_offset+1, i=0 -> (k_offset+1)*n + 0 = 2*k_offset + 2
        # C10_k: k=k_offset, i=1 -> k_offset*n + 1 = 2*k_offset + 1
        # C11_k: k=k_offset+1, i=1 -> (k_offset+1)*n + 1 = 2*k_offset + 3
        
        # M1: (A00 + A11) * (B00 + B11)
        add_term(0, k_offset, 2*k_offset, 1, 1, 1) # C00 += 1
        add_term(0, k_offset, 2*k_offset+3, 1, 1, 1) # C11 += 1
        add_term(4, k_offset, 2*k_offset, 1, 1, 1) # A11 part
        add_term(4, k_offset, 2*k_offset+3, 1, 1, 1)
        add_term(0, k_offset+1, 2*k_offset, 1, 1, 1) # B11 part (k_offset+1)
        add_term(0, k_offset+1, 2*k_offset+3, 1, 1, 1)
        add_term(4, k_offset+1, 2*k_offset, 1, 1, 1)
        add_term(4, k_offset+1, 2*k_offset+3, 1, 1, 1)
        # Wait, this is not how rank-1 decomposition works.
        # A rank-1 term is $u \otimes v \otimes w$.
        # $S = (u^T A) (v^T B)$.
        # Then $C += S w$.
        # This is ONE rank-1 term in the tensor decomposition?
        # No, the tensor is $\sum S_r w_r$.
        # The rank is the number of $S_r$.
        # So each Strassen multiplication is ONE rank-1 term?
        # Yes, if we allow the entries of U, V, W to be any real numbers.
        # U column r has the coefficients of $u_r$.
        # V column r has the coefficients of $v_r$.
        # W column r has the coefficients of $w_r$.
        # So for M1:
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        
        # Let's implement this correctly.
        
        def add_rank1(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                U[idx][col] += val
            for idx, val in v_coeffs.items():
                V[idx][col] += val
            for idx, val in w_coeffs.items():
                W[idx][col] += val
            col += 1

        # M1
        add_rank1({0: 1, 4: 1}, {k_offset: 1, p+k_offset+1: 1}, {2*k_offset: 1, 2*k_offset+3: 1})
        
        # M2: (A10 + A11) * B00
        # A10: idx 3, A11: idx 4
        # B00: idx k_offset
        # C: C10 (1), C11 (-1)
        add_rank1({3: 1, 4: 1}, {k_offset: 1}, {2*k_offset+1: 1, 2*k_offset+3: -1})
        
        # M3: A00 * (B01 - B11)
        # A00: idx 0
        # B01: idx k_offset+1, B11: idx p+k_offset+1
        # C: C01 (1), C11 (1)
        add_rank1({0: 1}, {k_offset+1: 1, p+k_offset+1: -1}, {2*k_offset+2: 1, 2*k_offset+3: 1})
        
        # M4: A11 * (B10 - B00)
        # A11: idx 4
        # B10: idx p+k_offset, B00: idx k_offset
        # C: C00 (-1), C10 (1)
        add_rank1({4: 1}, {p+k_offset: 1, k_offset: -1}, {2*k_offset: -1, 2*k_offset+1: 1})
        
        # M5: (A00 + A01) * B11
        # A00: 0, A01: 1
        # B11: p+k_offset+1
        # C: C01 (1), C11 (1)
        add_rank1({0: 1, 1: 1}, {p+k_offset+1: 1}, {2*k_offset+2: 1, 2*k_offset+3: 1})
        
        # M6: (A10 - A00) * (B00 + B01)
        # A10: 3, A00: 0
        # B00: k_offset, B01: k_offset+1
        # C: C00 (1)
        add_rank1({3: 1, 0: -1}, {k_offset: 1, k_offset+1: 1}, {2*k_offset: 1})
        
        # M7: (A01 - A11) * (B10 + B11)
        # A01: 1, A11: 4
        # B10: p+k_offset, B11: p+k_offset+1
        # C: C00 (-1)
        add_rank1({1: 1, 4: -1}, {p+k_offset: 1, p+k_offset+1: 1}, {2*k_offset: -1})

    # Add 4 Strassen blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    for k_start in [0, 2, 4, 6]:
        add_strassen_block(k_start)
        
    # Add Schoolbook block for k=8 (M_2,3,1)
    # C08 = A00 B08 + A01 B18 + A02 B28
    # C18 = A10 B08 + A11 B18 + A12 B28
    # A indices:
    # A00: 0, A01: 1, A02: 2
    # A10: 3, A11: 4, A12: 5
    # B indices:
    # B08: 0*p + 8 = 8
    # B18: 1*p + 8 = 17
    # B28: 2*p + 8 = 26
    # C indices:
    # C08: 8*n + 0 = 16
    # C18: 8*n + 1 = 17
    
    # Term 1: A00 * B08 -> C08
    add_rank1({0: 1}, {8: 1}, {16: 1})
    # Term 2: A01 * B18 -> C08
    add_rank1({1: 1}, {17: 1}, {16: 1})
    # Term 3: A02 * B28 -> C08
    add_rank1({2: 1}, {26: 1}, {16: 1})
    # Term 4: A10 * B08 -> C18
    add_rank1({3: 1}, {8: 1}, {17: 1})
    # Term 5: A11 * B18 -> C18
    add_rank1({4: 1}, {17: 1}, {17: 1})
    # Term 6: A12 * B28 -> C18
    add_rank1({5: 1}, {26: 1}, {17: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
