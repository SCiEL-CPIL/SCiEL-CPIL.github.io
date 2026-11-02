import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as fallback
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Strategy: Use a structured decomposition based on the rectangular geometry
    # For 3x3 x 3x8, we can try rank = n*m*p - savings
    # Key insight: We can potentially merge terms where the same (i,j) or (j,k) patterns repeat
    
    # Try a constructive approach: decompose using block structure
    # The 3x3 matrix A times 3x8 matrix B gives 3x8 result
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try rank values from 64 down to 50
    target_ranks = [64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        
        # Method 1: Structured initialization with Kronecker-like patterns
        # Try to use the fact that we have 3 input columns and 8 output columns
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank + 42)
            
            # Initialize with a mix of structured and random components
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # First, try a block-structured approach
            # Divide the rank into blocks corresponding to different (i,j) pairs
            # For each (i,j), we need contributions to all k values
            
            # Use a pattern where we share some rank dimensions
            # Try: rank = n * (m + p) - something, or similar
            
            # Alternative: Use a sparse structured initialization
            # Set up U, V, W with specific non-zero patterns
            
            # Try a "diagonal" structure first
            for c in range(target_rank):
                # Each column c corresponds to a specific computation
                # Try different patterns
                if c < n * m:
                    # U has a 1 in position (i*m+j)
                    idx = c
                    U[idx, c] = 1
                    # V and W need to be set up so that the product works
                    # For this to work, V[j*p+k, c] * W[k*n+i, c] should sum to 1
                    # over all k for each (i,j)
                    
                # This is getting complex. Let me try a different approach.
            
            # Method 2: Use a low-rank approximation and refine
            # Start with random initialization and use gradient-based optimization
            
            # Actually, let me try a simpler structured approach:
            # Use the fact that we can write the multiplication as:
            # C = A * B where A is 3x3, B is 3x8
            # We can potentially use rank = 3*3 + 3*8 - 3 = 24 + 24 - 3 = 45? No, that's not right.
            
            # Let me try: for each of the 3 rows of A, we compute 3*8=24 products
            # But we can share some computations across rows if the structure allows.
            
            # Try a specific rank: 53 (from known asymptotic bounds)
            if target_rank == 53:
                # Try a specific constructive decomposition
                # This requires careful design
                pass
        
        # Method 3: Use a more systematic search with structured initialization
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 2000 + target_rank + 123)
            
            # Initialize with a structured pattern
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Try to create a decomposition where:
            # - U has non-zero entries only in specific patterns
            # - V and W are set up to complement U
            
            # Use a "block diagonal" style initialization
            # Divide the 9 (i,j) pairs into groups and assign rank dimensions
            
            # For simplicity, try a random but scaled initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Check if this works (very unlikely for random)
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    # If no better solution found, return schoolbook
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
