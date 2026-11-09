"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    Total rank: 65.
    """
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    
    # We will build the U, V, W matrices column by column (each column is one rank-1 term).
    # Term r corresponds to U[:, r], V[:, r], W[:, r].
    
    # Helper to add a term
    # U_idx: i * m + j
    # V_idx: j * p + k
    # W_idx: k * n + i
    
    # We will store coefficients in dictionaries for each term index
    # Since we are building from scratch, we can just use lists of lists and append.
    # But it's easier to define the terms first.
    
    terms = []
    
    # Part 1: 4x4x4 multiplication using Strassen
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols)
    # Split A, B, C into 2x2 blocks:
    # A00 A01
    # A10 A11
    # Indices:
    # A00: i in 0-1, j in 0-1
    # A01: i in 0-1, j in 2-3
    # A10: i in 2-3, j in 0-1
    # A11: i in 2-3, j in 2-3
    
    # B00 B01
    # B10 B11
    # Indices:
    # B00: j in 0-1, k in 0-1
    # B01: j in 0-1, k in 2-3
    # B10: j in 2-3, k in 0-1
    # B11: j in 2-3, k in 2-3
    
    # C00 C01
    # C10 C11
    # Indices:
    # C00: i in 0-1, k in 0-1
    # C01: i in 0-1, k in 2-3
    # C10: i in 2-3, k in 0-1
    # C11: i in 2-3, k in 2-3
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map block coefficients to element coefficients.
    # For a block multiplication, e.g., M1, the A-block is A00 + A11.
    # This means for each element (i,j) in A00, coeff is 1. For each (i,j) in A11, coeff is 1.
    # Similarly for B.
    
    # We will define the Strassen terms for the top-left 2x2 block of the 4x4 multiplication.
    # But wait, the 4x4 multiplication is done by applying Strassen to the 4x4 matrices directly?
    # Yes, Strassen can be applied to any size, but usually it's recursive.
    # For 4x4, we split into 2x2 blocks. The "scalar" multiplications are 2x2 matrix multiplications.
    # Each 2x2 matrix multiplication can be done with rank 7.
    # So total rank for 4x4x4 is 7 * 7 = 49.
    
    # Let's define the structure for one 2x2 Strassen step.
    # Inputs: A_block (2x2), B_block (2x2).
    # Outputs: C_block (2x2).
    # The 7 terms correspond to the 7 multiplications.
    
    # We need to compose this for the 4x4 case.
    # The 4x4 case has 4 such 2x2 "levels" for the A and B partitions?
    # No, the 4x4 case is one level of Strassen where the "scalars" are 2x2 matrices.
    # So we have 7 terms. Each term is a 2x2 matrix multiplication.
    # Each 2x2 matrix multiplication has rank 7.
    # So we have 7 * 7 = 49 terms.
    
    # Let's iterate through the 7 Strassen terms for the 4x4 level.
    # For each term t (1..7), we have:
    # A_t = linear combo of A blocks
    # B_t = linear combo of B blocks
    # The product A_t B_t contributes to C blocks with coefficients.
    
    # Let's define the block coefficient matrices for the 7 terms.
    # A_coeffs[t][block] = coeff
    # B_coeffs[t][block] = coeff
    # C_coeffs[t][block] = coeff (contribution of M_t to C_block)
    
    # Blocks: 0=A00, 1=A11, 2=A10, 3=A01 (A matrix)
    # Blocks: 0=B00, 1=B11, 2=B00 (same as B00?), 3=B11?
    # Let's use indices 0,1,2,3 for the 2x2 blocks.
    # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right.
    
    # Strassen:
    # M1: A(0+3), B(0+3) -> C(0)=1, C(3)=1
    # M2: A(2+3), B(0) -> C(2)=1
    # M3: A(0), B(1-3) -> C(1)=1
    # M4: A(3), B(2-0) -> C(0)=1, C(2)=1
    # M5: A(0+1), B(3) -> C(0)=-1, C(1)=1
    # M6: A(2-0), B(0+1) -> C(3)=1
    # M7: A(1-3), B(2+3) -> C(0)=1, C(3)=1
    
    # Let's refine the C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each term t, we have a set of (A_block, coeff), (B_block, coeff), and for each C_block, a coeff.
    
    # We will generate 49 terms.
    # For each of the 7 high-level terms t:
    #   For each of the 7 low-level terms s (from Strassen on 2x2):
    #     Create a term.
    
    # Low-level Strassen (for 2x2):
    # Inputs a,b,c,d and e,f,g,h.
    # m1=(a+d)(e+h)
    # m2=(c+d)e
    # m3=a(f-h)
    # m4=d(g-c)
    # m5=(a+b)h
    # m6=(c-d)(e+f)
    # m7=(b-d)(g+h)
    # c11=m1+m4-m5+m7
    # c12=m3+m5
    # c21=m2+m4
    # c22=m1-m2+m3+m6
    
    # We need to map the low-level elements to the high-level block elements.
    # High-level block A_t is a 2x2 matrix.
    # Its elements are linear combinations of the 4x4 A elements.
    
    # Example: High-level term 1 uses A_block 0 and 3.
    # A_block 0 is A[0:2, 0:2]. A_block 3 is A[2:4, 2:4].
    # So A_t is a 2x2 matrix where:
    # A_t[0,0] = A[0,0] + A[2,2]
    # A_t[0,1] = A[0,1] + A[2,3]
    # A_t[1,0] = A[1,0] + A[3,2]
    # A_t[1,1] = A[1,1] + A[3,3]
    
    # Then we apply low-level Strassen to A_t and B_t.
    # The low-level Strassen produces 7 terms.
    # Each low-level term is a scalar multiplication (conceptually) but in this context, it's a "term" in the decomposition.
    # Wait, the low-level Strassen is an exact decomposition of the 2x2 multiplication tensor.
    # So if A_t and B_t are given, the product A_t B_t is computed by summing the 7 low-level terms.
    # So for each high-level term t, we have 7 low-level terms s.
    # Total 49 terms.
    
    # Let's define the low-level Strassen coefficients.
    # Low-level inputs: a,b,c,d (A_t) and e,f,g,h (B_t).
    # Low-level outputs: c11,c12,c21,c22 (C_t).
    
    # Coefficients for low-level terms:
    # Term 1: (a+d)(e+h)
    #   A: a=1, d=1
    #   B: e=1, h=1
    #   C: c11=1, c22=1
    # Term 2: (c+d)e
    #   A: c=1, d=1
    #   B: e=1
    #   C: c21=1
    # Term 3: a(f-h)
    #   A: a=1
    #   B: f=1, h=-1
    #   C: c12=1
    # Term 4: d(g-c)
    #   A: d=1
    #   B: g=1, c=-1
    #   C: c11=1, c21=1
    # Term 5: (a+b)h
    #   A: a=1, b=1
    #   B: h=1
    #   C: c11=-1, c12=1
    # Term 6: (c-d)(e+f)
    #   A: c=1, d=-1
    #   B: e=1, f=1
    #   C: c22=1
    # Term 7: (b-d)(g+h)
    #   A: b=1, d=-1
    #   B: g=1, h=1
    #   C: c11=1, c22=1
    
    # Now we combine.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # High-level Strassen definitions
    # t: 0..6
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    hl_terms = [
        { # M1
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)] # C00, C11
        },
        { # M2
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)] # C10
        },
        { # M3
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1)] # C01
        },
        { # M4
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)] # C00, C10
        },
        { # M5
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)] # C00, C01
        },
        { # M6
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)] # C11
        },
        { # M7
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1), (3, 1)] # C00, C11
        }
    ]
    
    # Low-level Strassen definitions
    # s: 0..6
    # A_elems: list of (elem_idx, coeff) where elem 0=a, 1=b, 2=c, 3=d
    # B_elems: list of (elem_idx, coeff) where elem 0=e, 1=f, 2=g, 3=h
    # C_elems: list of (elem_idx, coeff) where elem 0=c11, 1=c12, 2=c21, 3=c22
    
    ll_terms = [
        { # m1
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        { # m2
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)]
        },
        { # m3
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1)]
        },
        { # m4
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        { # m5
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        { # m6
            'A': [(2, 1), (3, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        { # m7
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        }
    ]
    
    # Mapping of block index to (i_range, j_range)
    # 0: 0-1, 0-1
    # 1: 0-1, 2-3
    # 2: 2-3, 0-1
    # 3: 2-3, 2-3
    
    # Mapping of low-level element index to (i_offset, j_offset) within the block
    # 0: 0,0
    # 1: 0,1
    # 2: 1,0
    # 3: 1,1
    
    # For A: block (r,c) -> i in r, j in c.
    # Low-level elem (i,j) -> i_offset, j_offset.
    # Global i = r + i_offset, j = c + j_offset.
    
    # For B: block (r,c) -> j in r, k in c.
    # Low-level elem (j,k) -> j_offset, k_offset.
    # Global j = r + j_offset, k = c + k_offset.
    
    # For C: block (r,c) -> i in r, k in c.
    # Low-level elem (i,k) -> i_offset, k_offset.
    # Global i = r + i_offset, k = c + k_offset.
    
    # We will iterate through hl_terms and ll_terms.
    
    for t in hl_terms:
        # Construct A_t and B_t matrices (2x2) as linear combos of global elements
        # We need to sum the contributions from all blocks in t['A'] and t['B']
        # For each low-level term, we need the coefficients for each global element.
        
        # Let's precompute the A_t and B_t coefficients for each global element.
        # A_t is 2x2.
        # A_t[i,j] = sum_{block (br, bc) in t['A']} coeff * (1 if i,j matches block offset)
        
        # Actually, it's easier to just add the terms directly.
        # For each ll_term s in ll_terms:
        #   Create U_vec, V_vec, W_vec.
        #   For each (a_elem, a_coeff) in s['A']:
        #     For each (block, block_coeff) in t['A']:
        #       Map a_elem to global (i,j).
        #       U[i*m+j] += a_coeff * block_coeff
        #   Similarly for B and C.
        
        for s in ll_terms:
            U_vec = [0.0] * (n * m)
            V_vec = [0.0] * (m * p)
            W_vec = [0.0] * (p * n)
            
            # Process A
            for a_elem, a_coeff in s['A']:
                # a_elem is 0..3 (i,j in 2x2)
                a_i, a_j = divmod(a_elem, 2)
                for block_idx, block_coeff in t['A']:
                    # block_idx 0..3
                    br, bc = divmod(block_idx, 2)
                    # Global i, j
                    gi = br * 2 + a_i
                    gj = bc * 2 + a_j
                    idx = gi * m + gj
                    U_vec[idx] += a_coeff * block_coeff
            
            # Process B
            for b_elem, b_coeff in s['B']:
                # b_elem is 0..3 (j,k in 2x2)
                b_j, b_k = divmod(b_elem, 2)
                for block_idx, block_coeff in t['B']:
                    # block_idx 0..3
                    br, bc = divmod(block_idx, 2)
                    # Global j, k
                    gj = br * 2 + b_j
                    gk = bc * 2 + b_k
                    # B is 4x5, but we only use first 4 cols.
                    # k is 0..3 here.
                    idx = gj * p + gk
                    V_vec[idx] += b_coeff * block_coeff
            
            # Process C
            for c_elem, c_coeff in s['C']:
                # c_elem is 0..3 (i,k in 2x2)
                c_i, c_k = divmod(c_elem, 2)
                for block_idx, block_coeff in t['C']:
                    # block_idx 0..3
                    br, bc = divmod(block_idx, 2)
                    # Global i, k
                    gi = br * 2 + c_i
                    gk = bc * 2 + c_k
                    # W is p x n = 5x4. Index k*n + i.
                    idx = gk * n + gi
                    W_vec[idx] += c_coeff * block_coeff
            
            add_term(U_vec, V_vec, W_vec)
            
    # Part 2: 4x4x1 multiplication (5th column)
    # A is 4x4, B is 4x1 (col 4), C is 4x1 (col 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Schoolbook: 16 terms.
    for i in range(n):
        for j in range(m):
            U_vec = [0.0] * (n * m)
            V_vec = [0.0] * (m * p)
            W_vec = [0.0] * (p * n)
            
            U_vec[i * m + j] = 1.0
            V_vec[j * p + 4] = 1.0
            # C[i, 4] -> W index 4*n + i
            W_vec[4 * n + i] = 1.0
            
            add_term(U_vec, V_vec, W_vec)
            
    # Convert to lists of lists (or numpy arrays if needed, but lists are fine)
    # The return format is {'U': U, 'V': V, 'W': W}
    # U is list of rows? No, U is (n*m, rank).
    # My U_vec is a row of the U matrix?
    # Seed: U = [[0]*rank for _ in range(n*m)].
    # So U is a list of lists, where U[i][r] is the entry.
    # My add_term appends a vector. So U is a list of columns.
    # I need to transpose.
    
    U_final = [[U[r][i] for r in range(len(U))] for i in range(n*m)]
    V_final = [[V[r][i] for r in range(len(V))] for i in range(m*p)]
    W_final = [[W[r][i] for r in range(len(W))] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
