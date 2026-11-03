"""Find a low-rank exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
_START = time.time()
_DEADLINE = 130.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2(n_offset, m_offset, p_offset, n, m, p):
    """
    Return Strassen rank-7 decomposition for a 2x2x2 sub-block.
    The sub-block covers rows n_offset..n_offset+1 of A,
    columns m_offset..m_offset+1 of A (and B rows),
    columns p_offset..p_offset+1 of B (and output cols).
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our tensor: A[i,j] * B[j,k] contributes to C[i,k]
    # In the 2x2x2 sub-block, we need:
    # U[row, col] * V[col, row] * W[row, col] = 1 for the appropriate entries
    
    # Let me define the 7 rank-1 terms explicitly.
    # Each term c has U entries for the 2x2 = 4 positions (i,j) in the A-block,
    # V entries for the 2x2 = 4 positions (j,k) in the B-block,
    # W entries for the 2x2 = 4 positions (k,i) in the C-block.
    
    # Position indices in the full matrices:
    # A positions: (n_offset + di, m_offset + dj) for di,dj in {0,1}
    # B positions: (m_offset + dj, p_offset + dk) for dj,dk in {0,1}
    # C positions: (p_offset + dk, n_offset + di) for dk,di in {0,1}
    
    # Strassen terms (M1 through M7):
    # Each term is defined by which A entries, B entries, and C entries it touches.
    
    # Let's define each term as (U_dict, V_dict, W_dict) where keys are (di,dj), (dj,dk), (dk,di)
    
    terms = []
    
    # M1 = (a+d)(e+h): 
    # A: (0,0)=1, (1,1)=1
    # B: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1  -- wait no
    # Actually M1 = (A00+A11)*(B00+B11)
    # This expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # So U: (0,0)=1, (1,1)=1
    #    V: (0,0)=1, (1,1)=1  -- no wait, B00 means B[0,0], B11 means B[1,1]
    #    V: (0,0)=1, (1,1)=1
    #    W: (0,0)=1, (1,1)=1  -- C00 gets A00*B00, C11 gets A11*B11, but also C00 gets A00*B11? No...
    
    # Let me be more careful. M1 = (A00 + A11) * (B00 + B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Contribution to C00: A00*B00 (from first term)
    # Contribution to C01: A00*B11 (from second term) -- wait, B11 is B[1,1], so this contributes to C[0,1]? No.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # A00*B00: i=0,j=0,k=0 → C[0,0]
    # A00*B11: i=0,j=0,k=1 → C[0,1]  
    # A11*B00: i=1,j=1,k=0 → C[1,0]
    # A11*B11: i=1,j=1,k=1 → C[1,1]
    
    # So M1 contributes to all 4 C positions.
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1 -- no, that's wrong.
    # V[j,k]: for A00*B00, j=0,k=0. For A00*B11, j=0,k=1. For A11*B00, j=1,k=0. For A11*B11, j=1,k=1.
    # So V: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # W[k,i]: for C[0,0]: k=0,i=0. C[0,1]: k=1,i=0. C[1,0]: k=0,i=1. C[1,1]: k=1,i=1.
    # So W: (0,0)=1, (1,0)=1, (0,1)=1, (1,1)=1
    
    # Hmm, this is getting complex. Let me just directly enumerate the 7 Strassen terms.
    
    # Actually, the standard way: each Strassen product M_c is a bilinear form in A and B.
    # For the rank-1 tensor term, we need:
    # U[i,j] * V[j,k] * W[k,i] = coefficient of A[i,j]*B[j,k] in M_c
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # Coefficients: (0,0,0)=1, (0,0,1)=1, (1,1,0)=1, (1,1,1)=1
    # So U[0,0]=1, U[1,1]=1; V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1; W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    # But that gives U[0,0]*V[0,0]*W[0,0]=1 ✓, U[0,0]*V[0,1]*W[1,0]=1*1*1=1 ✓, etc.
    # Wait, W[k,i]: for (i=0,j=0,k=1): W[1,0]. Yes.
    
    # M2 = (A10+A11)*B00 = A10B00 + A11B00
    # Coefficients: (1,0,0)=1, (1,1,0)=1
    # U[1,0]=1, U[1,1]=1; V[0,0]=1; W[0,1]=1
    
    # M3 = A00*(B01-B11) = A00B01 - A00B11
    # Coefficients: (0,0,1)=1, (0,0,2)=1 -- wait, k goes 0,1 only for 2x2
    # (0,0,1)=1, (0,0,1)=-1? No: B01 is j=0,k=1, B11 is j=1,k=1
    # (i=0,j=0,k=1): +1, (i=0,j=1,k=1): -1
    # U[0,0]=1; V[0,1]=1, V[1,1]=-1; W[1,0]=1
    
    # M4 = A11*(B10-B00) = A11B10 - A11B00
    # (i=1,j=1,k=0): +1, (i=1,j=0,k=0): -1
    # U[1,1]=1; V[1,0]=1, V[0,0]=-1; W[0,1]=1
    
    # M5 = (A00+A01)*B11 = A00B11 + A01B11
    # (i=0,j=0,k=1): +1, (i=0,j=1,k=1): +1
    # U[0,0]=1, U[0,1]=1; V[1,1]=1; W[1,0]=1
    
    # M6 = (A10-A00)*(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # (1,0,0)=1, (1,0,1)=1, (0,0,0)=-1, (0,0,1)=-1
    # U[1,0]=1, U[0,0]=-1; V[0,0]=1, V[0,1]=1; W[0,1]=1, W[1,0]=1
    
    # M7 = (A01-A11)*(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    # (0,1,0)=1, (0,1,1)=1, (1,1,0)=-1, (1,1,1)=-1
    # U[0,1]=1, U[1,1]=-1; V[1,0]=1, V[1,1]=1; W[0,0]=1, W[1,0]=1
    
    # Let me define each term as a list of (i,j,k,coef) tuples
    strassen_terms = [
        # M1
        [(0,0,0,1), (0,0,1,1), (1,1,0,1), (1,1,1,1)],
        # M2
        [(1,0,0,1), (1,1,0,1)],
        # M3
        [(0,0,1,1), (0,1,1,-1)],
        # M4
        [(1,1,0,1), (1,0,0,-1)],
        # M5
        [(0,0,1,1), (0,1,1,1)],
        # M6
        [(1,0,0,1), (1,0,1,1), (0,0,0,-1), (0,0,1,-1)],
        # M7
        [(0,1,0,1), (0,1,1,1), (1,1,0,-1), (1,1,1,-1)],
    ]
    
    # For each term, we need to find U, V, W vectors such that
    # U[i,j]*V[j,k]*W[k,i] = coef for each (i,j,k,coef) in the term,
    # and U[i,j]*V[j,k]*W[k,i] = 0 for all other (i,j,k) in the 2x2x2 block.
    
    # This is a constraint satisfaction problem. The standard Strassen decomposition
    # has a specific structure. Let me just use the known rank-7 decomposition.
    
    # Actually, the simplest approach: for each of the 7 terms, assign a unique column.
    # For term c, set U, V, W such that the product gives the right coefficients.
    
    # The key insight: for each term, we can factor the coefficient tensor.
    # Let me just hardcode the standard Strassen U, V, W matrices for 2x2x2.
    
    # Standard Strassen (one common form):
    # U (2x2 x 7):
    #   col0: [1, 0, 0, 1]  (M1: A00, A11)
    #   col1: [0, 0, 1, 1]  (M2: A10, A11)  
    #   col2: [1, 0, 0, 0]  (M3: A00)
    #   col3: [0, 0, 0, 1]  (M4: A11)
    #   col4: [1, 1, 0, 0]  (M5: A00, A01)
    #   col5: [1, 0, 1, 0]  (M6: A00, A10) -- with signs
    #   col6: [0, 1, 0, 1]  (M7: A01, A11) -- with signs
    
    # This is getting complicated. Let me just use a direct construction.
    
    # For each term, I'll create a rank-1 approximation by assigning values.
    # The constraint is: for each (i,j,k) in the term with coefficient c,
    # U[i,j]*V[j,k]*W[k,i] = c, and for all other (i,j,k), the product = 0.
    
    # The zero constraint means at least one of U[i,j], V[j,k], W[k,i] must be 0.
    
    # Let me just return None and handle it differently - use schoolbook for now
    # and focus on getting a working improved solution.
    return None

def construct():
    global _START
    _START = time.time()
    
    n, m, p = DIMENSIONS
    
    # Use a block decomposition with Strassen for 2x2 sub-blocks
    # A is 3x5, B is 5x7, C is 3x7
    # Split: A = [A_top (2x5); A_bot (1x5)], B = [B_left (5x6) B_right (5x1)]
    # Or better: split into 2x2 blocks where possible
    
    # Let's use: rows of A split as [2, 1], cols of B split as [2, 2, 2, 1]
    # This gives 8 output blocks:
    # C[0:2, 0:2], C[0:2, 2:4], C[0:2, 4:6], C[0:2, 6:7]
    # C[2:3, 0:2], C[2:3, 2:4], C[2:3, 4:6], C[2:3, 6:7]
    
    # For 2x2 blocks: use Strassen (rank 7)
    # For 2x1 blocks: schoolbook (rank 2)
    # For 1x2 blocks: schoolbook (rank 2)
    # For 1x1 block: schoolbook (rank 1)
    
    # Total: 3*7 + 3*2 + 3*2 + 1 = 21 + 6 + 6 + 1 = 34
    
    # Let me build this explicitly.
    # I'll create the U, V, W matrices column by column.
    
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # Define the block structure
    # Row blocks of A: [0:2], [2:3]
    # Col blocks of B: [0:2], [2:4], [4:6], [6:7]
    
    row_blocks = [(0, 2), (2, 3)]
    col_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    # For each pair of row block and col block, determine the method
    def add_schoolbook_term(U, V, W, i, j, k):
        """Add a single schoolbook rank-1 term for A[i,j]*B[j,k] -> C[i,k]"""
        col = len(U[0])
        for row in range(U_rows):
            U[row].append(0)
        for row in range(V_rows):
            V[row].append(0)
        for row in range(W_rows):
            W[row].append(0)
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
    
    def add_strassen_block(U, V, W, r_start, c_start):
        """Add Strassen rank-7 for 2x2x2 block starting at row r_start, col c_start of output"""
        # The 2x2x2 block covers:
        # A rows: r_start, r_start+1
        # A cols (B rows): c_start, c_start+1  -- wait, this isn't right
        # 
        # Actually for C[i,k] = sum_j A[i,j]*B[j,k], a 2x2 output block C[r:r+2, c:c+2]
        # requires A[r:r+2, :] and B[:, c:c+2], but the sum is over all j.
        # 
        # For Strassen to work, we need a 2x2 A block and a 2x2 B block with the
        # same intermediate dimension. So we need to also split the middle dimension.
        #
        # Let me reconsider. For a proper Strassen block, we need:
        # A_sub (2x2) * B_sub (2x2) = C_sub (2x2)
        # This means we're taking a 2x2 sub-block of A and a 2x2 sub-block of B
        # where the column index of A matches the row index of B.
        #
        # So we need to split the middle dimension (5) as well.
        # A is 3x5, B is 5x7.
        # If we split middle dim as [2, 2, 1]:
        #   A = [A[:,0:2], A[:,2:4], A[:,4:5]]
        #   B = [B[0:2,:], B[2:4,:], B[4:5,:]]
        #
        # Then C[i,k] = sum over middle blocks: A[i, mid_block] * B[mid_block, k]
        #
        # For Strassen, we need 2x2 A blocks and 2x2 B blocks with matching middle indices.
        # Middle split [2,2,1]:
        #   Middle block 0: A[:,0:2] (3x2) * B[0:2,:] (2x7) -> 3x7
        #   Middle block 1: A[:,2:4] (3x2) * B[2:4,:] (2x7) -> 3x7
        #   Middle block 2: A[:,4:5] (3x1) * B[4:5,:] (1x7) -> 3x7
        #
        # For middle block 0: A is 3x2, B is 2x7.
        #   Split A rows [2,1], B cols [2,2,2,1]:
        #   2x2 blocks: A[0:2,0:2]*B[0:2,0:2], A[0:2,0:2]*B[0:2,2:4], etc.
        #   But A[0:2,0:2] is 2x2, B[0:2,0:2] is 2x2 -> Strassen works!
        #   A[0:2,0:2]*B[0:2,2:4]: A is 2x2, B is 2x2 -> Strassen
        #   etc.
        #
        # This is getting very complex. Let me just use a simpler approach:
        # Use schoolbook but with a smarter grouping, or just return the known rank-99 solution.
        
        pass
    
    # Given the complexity, let me just return the schoolbook solution for now
    # but with a better structure. The known best is rank 99 using Strassen blocks.
    
    # Let me implement a proper Strassen-based decomposition.
    # Split middle dimension as [2, 2, 1]:
    #   Part 0: A[:,0:2] (3x2) * B[0:2,:] (2x7)
    #   Part 1: A[:,2:4] (3x2) * B[2:4,:] (2x7)  
    #   Part 2: A[:,4:5] (3x1) * B[4:5,:] (1x7)
    
    # For Part 0 and Part 1 (3x2 * 2x7):
    #   Split A rows [2,1], B cols [2,2,2,1]:
    #   2x2 blocks (3 of them per part): Strassen rank 7 each -> 21 per part
    #   2x1 blocks (1 per part): schoolbook rank 2
    #   1x2 blocks (3 per part): schoolbook rank 2 each -> 6
    #   1x1 block (1 per part): schoolbook rank 1
    #   Total per part: 21 + 2 + 6 + 1 = 30
    #   Two parts: 60
    
    # For Part 2 (3x1 * 1x7):
    #   Split A rows [2,1], B cols [2,2,2,1]:
    #   All 1x* blocks: schoolbook
    #   2x1 blocks: rank 2 each, 4 blocks -> 8
    #   1x1 blocks: rank 1 each, 4 blocks -> 4
    #   Wait: A is 3x1, B is 1x7. C[i,k] = A[i,0]*B[0,k]
    #   This is an outer product. Rank = 1! Just one term: U[i,0]=1, V[0,k]=1, W[k,i]=1
    #   Actually: C[i,k] = A[i,0]*B[0,k]. One rank-1 term suffices.
    #   U[i*m+0] for i=0,1,2: all 1
    #   V[0*p+k] for k=0..6: all 1
    #   W[k*n+i] for all k,i: all 1
    #   Check: sum_c U[i,0]*V[0,k]*W[k,i] = 1*1*1 = 1 ✓
    #   So Part 2: rank 1
    
    # Total: 60 + 1 = 61
    
    # Let me implement this.
    
    U = [[0] * 0 for _ in range(U_rows)]
    V = [[0] * 0 for _ in range(V_rows)]
    W = [[0] * 0 for _ in range(W_rows)]
    
    def add_term(U_vals, V_vals, W_vals):
        """Add a rank-1 term. U_vals: dict (i,j)->val, V_vals: dict (j,k)->val, W_vals: dict (k,i)->val"""
        col = len(U[0])
        for row in range(U_rows):
            U[row].append(0)
        for row in range(V_rows):
            V[row].append(0)
        for row in range(W_rows):
            W[row].append(0)
        for (i, j), val in U_vals.items():
            U[i*m + j][col] = val
        for (j, k), val in V_vals.items():
            V[j*p + k][col] = val
        for (k, i), val in W_vals.items():
            W[k*n + i][col] = val
    
    # Strassen 2x2x2 decomposition
    # For A_sub (2x2) at rows r0,r1 and cols c0,c1 (middle dim)
    # B_sub (2x2) at rows c0,c1 (middle dim) and cols k0,k1
    # C_sub (2x2) at rows r0,r1 and cols k0,k1
    
    def add_strassen(U, V, W, r0, c0, k0, m_offset, p_offset):
        """
        Add Strassen rank-7 for 2x2x2 block.
        r0: first row of A (and C)
        c0: first col of A / first row of B (middle dim)
        k0: first col of B (and C)
        m_offset: offset in middle dimension (0 for first part, 2 for second)
        p_offset: offset in column dimension
        """
        # A indices: (r0+c0_local, m_offset+c0_local) for c0_local in 0,1
        # B indices: (m_offset+c0_local, p_offset+k0_local) for c0_local,k0_local in 0,1
        # C indices: (p_offset+k0_local, r0+c0_local) for k0_local,c0_local in 0,1
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {(r0, m_offset): 1, (r0+1, m_offset+1): 1},
            {(m_offset, p_offset): 1, (m_offset, p_offset+1): 1, (m_offset+1, p_offset): 1, (m_offset+1, p_offset+1): 1},
            {(p_offset, r0): 1, (p_offset+1, r0): 1, (p_offset, r0+1): 1, (p_offset+1, r0+1): 1}
        )
        # M2 = (A10+A11)B00
        add_term(
            {(r0+1, m_offset): 1, (r0+1, m_offset+1): 1},
            {(m_offset, p_offset): 1},
            {(p_offset, r0+1): 1}
        )
        # M3 = A00(B01-B11)
        add_term(
            {(r0, m_offset): 1},
            {(m_offset, p_offset+1): 1, (m_offset+1, p_offset+1): -1},
            {(p_offset+1, r0): 1}
        )
        # M4 = A11(B10-B00)
        add_term(
            {(r0+1, m_offset+1): 1},
            {(m_offset+1, p_offset): 1, (m_offset, p_offset): -1},
            {(p_offset, r0+1): 1}
        )
        # M5 = (A00+A01)B11
        add_term(
            {(r0, m_offset): 1, (r0, m_offset+1): 1},
            {(m_offset+1, p_offset+1): 1},
            {(p_offset+1, r0): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_term(
            {(r0+1, m_offset): 1, (r0, m_offset): -1},
            {(m_offset, p_offset): 1, (m_offset, p_offset+1): 1},
            {(p_offset, r0+1): 1, (p_offset+1, r0): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            {(r0, m_offset+1): 1, (r0+1, m_offset+1): -1},
            {(m_offset+1, p_offset): 1, (m_offset+1, p_offset+1): 1},
            {(p_offset, r0): 1, (p_offset+1, r0): 1}
        )
    
    # Part 0: middle dim 0:1 (cols 0,1 of A, rows 0,1 of B)
    # A rows 0,1 with B cols 0,1,2,3: Strassen
    add_strassen(U, V, W, r0=0, c0=0, k0=0, m_offset=0, p_offset=0)
    add_strassen(U, V, W, r0=0, c0=0, k0=2, m_offset=0, p_offset=2)
    # A rows 0,1 with B col 4: schoolbook (2x1)
    for i in range(2):
        add_term({(i, 0): 1}, {(0, 4): 1}, {(4, i): 1})
    # A row 2 with B cols 0,1: schoolbook (1x2)
    for k in range(2):
        add_term({(2, 0): 1}, {(0, k): 1}, {(k, 2): 1})
    # A row 2 with B cols 2,3: schoolbook (1x2)
    for k in range(2, 4):
        add_term({(2, 0): 1}, {(0, k): 1}, {(k, 2): 1})
    # A row 2 with B col 4: schoolbook (1x1)
    add_term({(2, 0): 1}, {(0, 4): 1}, {(4, 2): 1})
    
    # Part 1: middle dim 2:3 (cols 2,3 of A, rows 2,3 of B)
    add_strassen(U, V, W, r0=0, c0=0, k0=0, m_offset=2, p_offset=0)
    add_strassen(U, V, W, r0=0, c0=0, k0=2, m_offset=2, p_offset=2)
    # A rows 0,1 with B col 4: schoolbook (2x1)
    for i in range(2):
        add_term({(i, 2): 1, (i, 3): 1}, {(2, 4): 1, (3, 4): 1}, {(4, i): 1})
    # Wait, this isn't right. For schoolbook, each (i,j,k) is separate.
    # Let me redo.
    
    # Actually, I'm making this too complicated. Let me just verify the count and return schoolbook if validation fails.
    
    # Reset and do it properly
    U = [[0] * 0 for _ in range(U_rows)]
    V = [[0] * 0 for _ in range(V_rows)]
    W = [[0] * 0 for _ in range(W_rows)]
    
    # Part 0: middle indices 0,1
    # Strassen for (A rows 0,1) x (B cols 0,1) -> C[0:2, 0:2]
    add_strassen(U, V, W, r0=0, c0=0, k0=0, m_offset=0, p_offset=0)
    # Strassen for (A rows 0,1) x (B cols 2,3) -> C[0:2, 2:4]
    add_strassen(U, V, W, r0=0, c0=0, k0=2, m_offset=0, p_offset=2)
    # Schoolbook for (A rows 0,1) x (B col 4) -> C[0:2, 4:7] -- wait, B col 4 is just one col
    # A[i,0]*B[0,4] and A[i,1]*B[1,4] for i=0,1
    for i in range(2):
        for j in range(2):
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    # Schoolbook for (A row 2) x (B cols 0,1) -> C[2, 0:2]
    for k in range(2):
        for j in range(2):
            add_term({(2, j): 1}, {(j, k): 1}, {(k, 2): 1})
    # Schoolbook for (A row 2) x (B cols 2,3) -> C[2, 2:4]
    for k in range(2, 4):
        for j in range(2):
            add_term({(2, j): 1}, {(j, k): 1}, {(k, 2): 1})
    # Schoolbook for (A row 2) x (B col 4) -> C[2, 4]
    for j in range(2):
        add_term({(2, j): 1}, {(j, 4): 1}, {(4, 2): 1})
    
    # Part 1: middle indices 2,3
    # Strassen for (A rows 0,1) x (B cols 0,1) -> C[0:2, 0:2]
    add_strassen(U, V, W, r0=0, c0=0, k0=0, m_offset=2, p_offset=0)
    # Strassen for (A rows 0,1) x (B cols 2,3) -> C[0:2, 2:4]
    add_strassen(U, V, W, r0=0, c0=0, k0=2, m_offset=2, p_offset=2)
    # Schoolbook for (A rows 0,1) x (B col 4) -> C[0:2, 4]
    for i in range(2):
        for j in range(2, 4):
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    # Schoolbook for (A row 2) x (B cols 0,1) -> C[2, 0:2]
    for k in range(2):
        for j in range(2, 4):
            add_term({(2, j): 1}, {(j, k): 1}, {(k, 2): 1})
    # Schoolbook for (A row 2) x (B cols 2,3) -> C[2, 2:4]
    for k in range(2, 4):
        for j in range(2, 4):
            add_term({(2, j): 1}, {(j, k): 1}, {(k, 2): 1})
    # Schoolbook for (A row 2) x (B col 4) -> C[2, 4]
    for j in range(2, 4):
        add_term({(2, j): 1}, {(j, 4): 1}, {(4, 2): 1})
    
    # Part 2: middle index 4
    # A[:,4] (3x1) * B[4,:] (1x7) -> outer product, rank 1
    add_term(
        {(0, 4): 1, (1, 4): 1, (2, 4): 1},
        {(4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1, (4, 5): 1, (4, 6): 1},
        {(0, 0): 1, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
    )
    
    rank = len(U[0])
    
    # Validate
    if _validate(U, V, W, n, m, p, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # If validation fails, return schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {"U": U_s, "V": V_s, "W": W_s}
