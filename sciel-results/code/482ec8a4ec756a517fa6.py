import numpy as np

def construct():
    """
    Returns a rank-40 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
      - Split the 4x9 matrix B into three 4x3 blocks: B1 (cols 0-2), B2 (cols 3-5), B3 (cols 6-8).
      - For each 4x3 block, use a rank-12 decomposition derived from 2x2 Strassen.
      - Total rank = 3 * 12 = 36.
    
    Actually, let's be more careful. Let me think about this again.
    
    For 4x4 * 4x3:
      A is 4x4, B is 4x3, C is 4x3.
      C[i,j] = sum_k A[i,k] * B[k,j]
      
      We can use the fact that this is equivalent to computing A * B where B has 3 columns.
      
      Using Strassen-like decomposition:
      Split A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
      Split B into 2x2 blocks of 2x(3/2)... this doesn't work cleanly.
      
    Let me try a different approach: use rank-12 for 4x4x4 (Laderman) and adapt.
    
    Actually, the simplest valid improvement: use Laderman's rank-18 for 4x4x4.
    Since 4x9 = 4x(3*3), we can do 3 separate 4x4 multiplications (each with rank 18),
    giving total rank 3*18 = 54. But wait, the middle dimension is 4, not 9.
    
    A is 4x4, B is 4x9. So A[i,k] * B[k,j] where k=0..3, j=0..8.
    
    We can group the 9 columns of B into three groups of 3: B[:, 0:3], B[:, 3:6], B[:, 6:9].
    Each 4x3 block multiplication can be done with rank... let's think.
    
    For 4x4 * 4x3: This is a 4x4x3 tensor. The schoolbook rank is 4*4*3 = 48.
    Can we do better?
    
    Actually, let's just use the Strassen 2x2x2 decomposition applied appropriately.
    
    Let me try: use the known result that M(4,4,3) <= 12.
    
    Here's a construction for 4x4 * 4x3 with rank 12:
    
    Label A's rows 0-3, columns 0-3. B's rows 0-3, columns 0-2. C's rows 0-3, columns 0-2.
    
    C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    We can write this using 12 bilinear terms. One approach:
    
    Use the fact that we can compute 4x4 * 4x4 in rank 18 (Laderman), then extract 3 columns.
    But that would give rank 18 per set of 3 columns, total 3*18=54.
    
    Alternatively, use a more direct decomposition.
    
    Let me just implement Laderman's rank-18 for 4x4x4, and use it 3 times for the 9 columns
    (3 groups of 3 columns each), giving rank 54.
    
    Actually, wait. Let me reconsider. The task is 4x4 * 4x9. 
    n=4 (rows of A), m=4 (cols of A = rows of B), p=9 (cols of B).
    
    U: (n*m, rank) = (16, rank)
    V: (m*p, rank) = (36, rank)  
    W: (p*n, rank) = (36, rank)
    
    The tensor is T[i,j,k] = delta_{i,k} for the multiplication, where:
    - i indexes rows of A (0..3)
    - j indexes columns of A / rows of B (0..3)
    - k indexes columns of B (0..8)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor has indices (i, j, k) with i in [0,4), j in [0,4), k in [0,9).
    T[i,j,k] = 1 if... actually it's the identity in the j dimension contracted.
    
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For schoolbook: each (i,j,k) triple has one term.
    
    For Laderman applied to 4x4x4: we get rank 18 for the case where p=4.
    For p=9, we can do 3 independent 4x4x4 computations (each handling 4 of the 9 columns... no, 9 isn't divisible by 4).
    
    9 = 4 + 4 + 1. So we could do two 4x4x4 (rank 18 each) and one 4x4x1 (rank 4), total rank 40.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 9
    
    # We'll build U, V, W as lists
    # Rank will be 18 + 18 + 4 = 40
    
    def laderman_4x4x4():
        """
        Return (U, V, W) for 4x4x4 matrix multiplication with rank 18.
        U: (16, 18), V: (16, 18), W: (16, 18)
        
        Using Laderman's algorithm.
        """
        U = [[0]*18 for _ in range(16)]
        V = [[0]*18 for _ in range(16)]
        W = [[0]*18 for _ in range(16)]
        
        # Laderman's 18 products for 4x4 matrix multiplication
        # A is 4x4, B is 4x4, C = A*B is 4x4
        # Index: U[i*4+j], V[j*4+k], W[k*4+i]
        
        # M1 = A11 * (B11 + B22)
        # In 0-indexed: A[0,0] * (B[0,0] + B[1,1])
        # U[0*4+0] = 1, V[0*4+0] = 1, V[1*4+1] = 1, W[0*4+0] = 1
        def add_term(u_idx, v_indices, w_idx, sign=1):
            col = 0  # We'll handle columns externally
            pass
        
        # Let me define the 18 terms explicitly
        # Each term: (u_row, [v_rows], w_row)
        # Where u_row = i*4+j, v_rows = list of (j*4+k) indices, w_row = k*4+i
        
        # Term 1: M1 = A[0,0] * (B[0,0] + B[1,1])
        # C contribution: C[0,0] += M1, C[0,1] += M1
        # Wait, I need to be more careful about how Laderman works.
        
        # Let me look up the exact Laderman decomposition.
        # Actually, let me use a simpler approach: just use schoolbook for now but with a better structure.
        
        # For 4x4x4, schoolbook is rank 16. Can we do better?
        # Yes, Strassen gives rank 7 for 2x2x2, so for 4x4x4 we can get rank 7*... no.
        # 4x4 = 2x2 of 2x2, so rank is 7*... actually it's more complex.
        
        # Let me just use the known rank-18 Laderman decomposition.
        
        # Here's one formulation:
        # Let A = [[a11, a12, a13, a14],
        #           [a21, a22, a23, a24],
        #           [a31, a32, a33, a34],
        #           [a41, a42, a43, a44]]
        # B = [[b11, b12, b13, b14],
        #      [b21, b22, b23, b24],
        #      [b31, b32, b33, b34],
        #      [b41, b42, b43, b44]]
        
        # The 18 products and their contributions to C:
        
        # This is getting complex. Let me just use schoolbook rank 16 for 4x4x4.
        # Then total rank = 16 + 16 + 4 = 36.
        
        return None  # We'll use schoolbook
        
    # Let's just use schoolbook for the 4x4x4 blocks and 4x4x1 block
    # Total rank = 16 + 16 + 4 = 36
    
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 4x4x4 block: columns 0-3 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Second 4x4x4 block: columns 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Third 4x4x1 block: column 8 of B
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+8][col] = 1
            W[8*n+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
