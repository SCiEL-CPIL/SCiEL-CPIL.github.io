"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2_terms():
    """
    Return the 5 rank-1 tensors for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-dim vectors.
    Indexing: A = [[a,b],[c,d]] -> [a,b,c,d]
               B = [[e,f],[g,h]] -> [e,f,g,h]
               C = [[C00,C01],[C10,C11]] -> [C00,C01,C10,C11]
    """
    terms = []
    # P1 = (a+d)*(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # P2 = (c+d)*e
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # P3 = a*(g-h)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    # P4 = d*(e-f)
    terms.append(([0,0,0,1], [1,-1,0,0], [1,0,1,0]))
    # P5 = (a+b)*h
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    return terms

def _add_block_product(U, V, W, block_offset, a_block, b_block, c_block, terms):
    """
    Add a 2x2x2 block product to the decomposition.
    
    a_block: (block_row, block_col) position in A's 2x2 block structure
    b_block: (block_row, block_col) position in B's 2x2 block structure  
    c_block: (block_row, block_col) position in C's 2x2 block structure
    
    A is 4x4, split into 2x2 blocks of 2x2.
    B is 4x5 (for the main case) or 4x4.
    C is 4x5 or 4x4.
    
    For A (4x4): element (i,j) has index i*4+j
      Block (br,bc) contains elements (2*br+r, 2*bc+c) for r,c in {0,1}
      Local index in block: r*2+c
      Global index: (2*br+r)*4 + (2*bc+c)
    
    For B (4x5): element (j,k) has index j*5+k
      Block (br,bc) for 2x2 block: elements (2*br+r, 2*bc+c) for r in {0,1}, c in {0,1}
      But B has 5 columns, so we only handle 2x2 blocks where both dims fit.
      
    For C (4x5): element (k,i) has index k*4+i (trace convention W[k*n+i])
      Block (br,bc): elements (2*br+r, 2*bc+c)
      Global index: (2*br+r)*4 + (2*bc+c)
    """
    for (u, v, w) in terms:
        # u has 4 entries corresponding to local A block positions [a,b,c,d]
        # v has 4 entries corresponding to local B block positions [e,f,g,h]
        # w has 4 entries corresponding to local C block positions [C00,C01,C10,C11]
        
        # Map local A indices to global
        a_local_to_global = {
            0: (2*a_block[0]+0)*4 + (2*a_block[1]+0),  # a
            1: (2*a_block[0]+0)*4 + (2*a_block[1]+1),  # b
            2: (2*a_block[0]+1)*4 + (2*a_block[1]+0),  # c
            3: (2*a_block[0]+1)*4 + (2*a_block[1]+1),  # d
        }
        
        # Map local B indices to global (B is 4x5, so j*5+k)
        b_local_to_global = {
            0: (2*b_block[0]+0)*5 + (2*b_block[1]+0),  # e
            1: (2*b_block[0]+0)*5 + (2*b_block[1]+1),  # f
            2: (2*b_block[0]+1)*5 + (2*b_block[1]+0),  # g
            3: (2*b_block[0]+1)*5 + (2*b_block[1]+1),  # h
        }
        
        # Map local C indices to global (C is 4x5, W index is k*4+i)
        c_local_to_global = {
            0: (2*c_block[0]+0)*4 + (2*c_block[1]+0),  # C00
            1: (2*c_block[0]+0)*4 + (2*c_block[1]+1),  # C01
            2: (2*c_block[0]+1)*4 + (2*c_block[1]+0),  # C10
            3: (2*c_block[0]+1)*4 + (2*c_block[1]+1),  # C11
        }
        
        for local_idx, val in enumerate(u):
            if val != 0:
                global_idx = a_local_to_global[local_idx]
                U[global_idx][block_offset] += val
        for local_idx, val in enumerate(v):
            if val != 0:
                global_idx = b_local_to_global[local_idx]
                V[global_idx][block_offset] += val
        for local_idx, val in enumerate(w):
            if val != 0:
                global_idx = c_local_to_global[local_idx]
                W[global_idx][block_offset] += val

def _construct_winograd_4x4x5(n=4, m=4, p=5):
    """
    Construct a decomposition for 4x4 * 4x5 using Winograd 2x2x2 blocks.
    
    Split A into 2x2 blocks of 2x2:
    A = [[A00, A01], [A10, A11]]
    
    Split B into 2x2 blocks (first 4 cols) plus 1 extra column:
    B = [[B00, B01, b0], [B10, B11, b1]]
    where B00,B01,B10,B11 are 2x2 and b0,b1 are 2x1 columns.
    
    C = A*B = [[C00, C01, c0], [C10, C11, c1]]
    where C00,C01,C10,C11 are 2x2 and c0,c1 are 2x1.
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    c0  = A00*b0 + A01*b1
    c1  = A10*b0 + A11*b1
    
    For the 4x4x4 part (C00,C01,C10,C11): 8 block products, each rank 5 = 40
    For the 4x4x1 part (c0,c1): 4 block products, each... 
      Actually, A00*b0 is a 2x2 times 2x1, which is just 4 scalar mults (rank 4)
      Or we can use a rank-2 decomposition? No, 2x2*2x1 has rank 4 (it's just matrix-vector).
      Actually 2x2 * 2x1: C = A*B where A is 2x2, B is 2x1.
      C[0] = A[0,0]*B[0] + A[0,1]*B[1]
      C[1] = A[1,0]*B[0] + A[1,1]*B[1]
      This is rank 4 (schoolbook) since there's no sharing.
      
      So 4 products * 4 rank = 16 for the extra column.
      
    Total rank: 40 + 16 = 56
    """
    terms = _winograd_2x2x2_terms()
    total_rank = 8 * 5 + 4 * 4  # 40 + 16 = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 56
    
    offset = 0
    
    # 8 block products for the 4x4x4 part
    # C00 = A00*B00 + A01*B10
    _add_block_product(U, V, W, offset, (0,0), (0,0), (0,0), terms)
    offset += 5
    _add_block_product(U, V, W, offset, (0,1), (1,0), (0,0), terms)
    offset += 5
    
    # C01 = A00*B01 + A01*B11
    _add_block_product(U, V, W, offset, (0,0), (0,1), (0,1), terms)
    offset += 5
    _add_block_product(U, V, W, offset, (0,1), (1,1), (0,1), terms)
    offset += 5
    
    # C10 = A10*B00 + A11*B10
    _add_block_product(U, V, W, offset, (1,0), (0,0), (1,0), terms)
    offset += 5
    _add_block_product(U, V, W, offset, (1,1), (1,0), (1,0), terms)
    offset += 5
    
    # C11 = A10*B01 + A11*B11
    _add_block_product(U, V, W, offset, (1,0), (0,1), (1,1), terms)
    offset += 5
    _add_block_product(U, V, W, offset, (1,1), (1,1), (1,1), terms)
    offset += 5
    
    # 4 block products for the extra column (4x4x1 part)
    # c0 = A00*b0 + A01*b1
    # c1 = A10*b0 + A11*b1
    # Each is 2x2 * 2x1, rank 4 (schoolbook)
    
    # A00*b0: A00 is block (0,0), b0 is column 4, rows 0-1
    # C contribution: c0 is rows 0-1, column 4
    for a_br in range(2):
        for a_bc in range(2):
            # A block (a_br, a_bc)
            # b column: if a_bc == 0, use b0 (col 4, rows 0-1); if a_bc == 1, use b1 (col 4, rows 2-3)
            # Wait, I need to reconsider the block structure for B.
            # B is 4x5. Blocks: B[0:2,0:2]=B00, B[0:2,2:4]=B01, B[2:4,0:2]=B10, B[2:4,2:4]=B11
            # Extra column: B[0:2,4]=b0, B[2:4,4]=b1
            # C is 4x5. C[0:2,0:2]=C00, C[0:2,2:4]=C01, C[2:4,0:2]=C10, C[2:4,2:4]=C11
            # Extra column: C[0:2,4]=c0, C[2:4,4]=c1
            
            # c0 = A00*b0 + A01*b1
            # c1 = A10*b0 + A11*b1
            
            # For c0: A blocks in row 0 of block structure multiply b0 or b1
            # A00 (block row 0, col 0) * b0 (col 4, rows 0-1)
            # A01 (block row 0, col 1) * b1 (col 4, rows 2-3)
            
            pass
    
    # Let me do this more carefully
    # c0 (rows 0,1 of C, col 4) = A[0:2,:] * B[:,4]
    # = A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4]
    # = A00 * b0 + A01 * b1
    # A00 is 2x2 at A[0:2,0:2], b0 is 2x1 at B[0:2,4]
    # A01 is 2x2 at A[0:2,2:4], b1 is 2x1 at B[2:4,4]
    
    # c1 (rows 2,3 of C, col 4) = A[2:4,:] * B[:,4]
    # = A[2:4,0:2]*B[0:2,4] + A[2:4,2:4]*B[2:4,4]
    # = A10 * b0 + A11 * b1
    
    # Schoolbook for 2x2 * 2x1 (rank 4 each):
    # For A2x2 = [[a,b],[c,d]], B2x1 = [x,y]:
    # C[0] = a*x + b*y
    # C[1] = c*x + d*y
    # 4 rank-1 terms:
    # T1: a*x -> C[0]+=a*x
    # T2: b*y -> C[0]+=b*y
    # T3: c*x -> C[1]+=c*x
    # T4: d*y -> C[1]+=d*y
    
    def add_2x2x1_product(U, V, W, offset, a_block, b_col_start, c_row_start):
        """Add 2x2 * 2x1 schoolbook product.
        a_block: (br, bc) in A's 2x2 block grid
        b_col_start: row in B where the 2x1 column starts (0 or 2)
        c_row_start: row in C where the 2x1 result starts (0 or 2)
        """
        # A block (br, bc): elements at global A indices
        # A[2*br+r, 2*bc+c] for r,c in {0,1}
        # Local: [a,b,c,d] -> global: 
        #   a: (2*br+0)*4 + (2*bc+0)
        #   b: (2*br+0)*4 + (2*bc+1)
        #   c: (2*br+1)*4 + (2*bc+0)
        #   d: (2*br+1)*4 + (2*bc+1)
        
        # B 2x1 column starting at row b_col_start, col 4:
        #   x: B[b_col_start+0, 4] = (b_col_start+0)*5 + 4
        #   y: B[b_col_start+1, 4] = (b_col_start+1)*5 + 4
        
        # C 2x1 result starting at row c_row_start, col 4:
        #   C[0]: W[(c_row_start+0)*4 + 4]
        #   C[1]: W[(c_row_start+1)*4 + 4]
        
        a_idx = {
            'a': (2*a_block[0]+0)*4 + (2*a_block[1]+0),
            'b': (2*a_block[0]+0)*4 + (2*a_block[1]+1),
            'c': (2*a_block[0]+1)*4 + (2*a_block[1]+0),
            'd': (2*a_block[0]+1)*4 + (2*a_block[1]+1),
        }
        b_idx = {
            'x': (b_col_start+0)*5 + 4,
            'y': (b_col_start+1)*5 + 4,
        }
        c_idx = {
            'C0': (c_row_start+0)*4 + 4,
            'C1': (c_row_start+1)*4 + 4,
        }
        
        # T1: a*x -> C0
        U[a_idx['a']][offset] += 1
        V[b_idx['x']][offset] += 1
        W[c_idx['C0']][offset] += 1
        offset += 1
        
        # T2: b*y -> C0
        U[a_idx['b']][offset] += 1
        V[b_idx['y']][offset] += 1
        W[c_idx['C0']][offset] += 1
        offset += 1
        
        # T3: c*x -> C1
        U[a_idx['c']][offset] += 1
        V[b_idx['x']][offset] += 1
        W[c_idx['C1']][offset] += 1
        offset += 1
        
        # T4: d*y -> C1
        U[a_idx['d']][offset] += 1
        V[b_idx['y']][offset] += 1
        W[c_idx['C1']][offset] += 1
        offset += 1
        
        return offset
    
    # c0 = A00*b0 + A01*b1
    # A00: block (0,0), b0: rows 0-1 of B col 4, c0: rows 0-1 of C col 4
    offset = add_2x2x1_product(U, V, W, offset, (0,0), 0, 0)
    # A01: block (0,1), b1: rows 2-3 of B col 4, c0: rows 0-1 of C col 4
    offset = add_2x2x1_product(U, V, W, offset, (0,1), 2, 0)
    
    # c1 = A10*b0 + A11*b1
    # A10: block (1,0), b0: rows 0-1 of B col 4, c1: rows 2-3 of C col 4
    offset = add_2x2x1_product(U, V, W, offset, (1,0), 0, 2)
    # A11: block (1,1), b1: rows 2-3 of B col 4, c1: rows 2-3 of C col 4
    offset = add_2x2x1_product(U, V, W, offset, (1,1), 2, 2)
    
    assert offset == total_rank, f"Expected {total_rank}, got {offset}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 80)
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback: construct manually
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try Winograd-based decomposition (rank 56)
    if time.time() < deadline:
        try:
            winograd_result = _construct_winograd_4x4x5(n, m, p)
            if _validate(winograd_result["U"], winograd_result["V"], winograd_result["W"], n, m, p):
                winograd_rank = winograd_result["U"][0].__len__() if winograd_result["U"] else 0
                if winograd_rank < best_rank:
                    best_result = winograd_result
                    best_rank = winograd_rank
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
