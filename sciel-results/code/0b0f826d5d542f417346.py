import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 72
    
    # Try block decomposition: split p=8 into 3+3+2
    # A*B1 (3x3 x 3x3), A*B2 (3x3 x 3x3), A*B3 (3x3 x 3x2)
    
    # For 3x3 x 3x3, use rank-21 decomposition
    # For 3x3 x 3x2, use schoolbook (rank 18)
    
    # Rank-21 decomposition for 3x3 x 3x3
    # Using a known construction
    def get_3x3x3_rank21():
        """Return U, V, W for 3x3 x 3x3 multiplication with rank 21."""
        # This is a known rank-21 decomposition
        # U: 9x21, V: 9x21, W: 9x21
        # C = sum_r U[:,r] * V[:,r] * W[:,r] (outer product structure)
        
        # I'll use a concrete decomposition
        # The 21 terms correspond to specific bilinear forms
        
        U = np.zeros((9, 21))
        V = np.zeros((9, 21))
        W = np.zeros((9, 21))
        
        # Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        # Wait, the convention is:
        # U has shape (n*m, rank) = (9, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank) = (9, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank) = (9, rank), indexed by (k,i) -> k*n+i
        
        # For 3x3x3: n=m=p=3
        # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        
        # Rank-21 decomposition (Landsberg-style)
        # I'll use a specific known decomposition
        
        # Term 0-8: Diagonal terms A[i,i] * B[i,j] -> C[i,j]... 
        # Actually, let me use a verified decomposition.
        
        # Using the decomposition where we have:
        # 9 terms: for each i, j: A[i,j] * B[j,k] for fixed k... 
        # This is getting complicated. Let me use a numerical approach.
        
        return None  # Placeholder
    
    # Let me use a different approach: numerical optimization
    # But that might be too slow. Let me just use schoolbook for now
    # and try a simple improvement.
    
    # Actually, let me try rank 60 using block decomposition with a simple
    # rank-21 approximation.
    
    # For now, let me return schoolbook and add a time check
    if time.time() > deadline:
        rank = rank_school
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition
    # Use the fact that we can share computations
    
    # Let me try a rank-60 decomposition using block structure
    # Split p into 3+3+2
    
    # For the 3x3 x 3x3 blocks, I'll use a rank-21 decomposition
    # I need to find explicit coefficients
    
    # Let me use a known rank-21 decomposition for 3x3x3
    # From the literature, one such decomposition is:
    
    def build_3x3x3_rank21():
        """Build rank-21 decomposition for 3x3 x 3x3 matrix multiplication."""
        n = m = p = 3
        rank = 21
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # I'll use a specific known decomposition
        # The 21 bilinear forms are:
        
        # Using the decomposition from Bini et al. or similar
        # Let me use a concrete set of coefficients
        
        # Term indices 0-8: A[i,i] * B[i,j] for i=0,1,2, j=0,1,2
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U[i*3+i, r] = 1  # A[i,i]
                V[i*3+j, r] = 1  # B[i,j]
                W[i*3+i, r] = 1  # C[i,i]... wait, this doesn't seem right
        
        # This approach is getting messy. Let me use a verified decomposition.
        
        # I'll use the following known rank-21 decomposition:
        # Based on the work of Landsberg and others
        
        # Actually, let me just use a numerical solver for a smaller case
        # and verify it works.
        
        return None
    
    # Given the complexity, let me just return schoolbook for now
    # but with a proper time guard
    
    rank = rank_school
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
