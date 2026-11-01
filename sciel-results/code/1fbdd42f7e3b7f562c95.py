"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Try to find a low-rank decomposition
    # Start with a reasonable upper bound and try to reduce
    
    # First, get the schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try a structured decomposition using block methods
    # Approach: decompose M_{2,3,8} using tensor product structure
    
    # Method 1: Use a 2x3x3 decomposition (rank 9 schoolbook) for first 3 cols,
    # then 2x3x5 for remaining 5 cols
    # But let's try to find better
    
    # Try random rank-r decompositions and check validity
    def check_decomposition(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Check for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try a structured approach: use linear combinations
    # For each rank-1 term, U is a 6-vector, V is a 24-vector, W is a 16-vector
    
    # Let's try to construct a decomposition with rank < 48
    # Using the idea that we can share computations across k values
    
    # Construct a candidate using block diagonal structure
    # Split 8 columns into groups and use shared A-multiplications
    
    # Try rank 42: use 6 rank-1 terms for each of 7 groups of ~1.14 cols... not clean
    
    # Let me try a different structural approach:
    # Use the fact that M_{2,3,8} can be embedded in M_{2,3,9} or similar
    
    # For now, try a simple improvement: use rank 44 by combining
    # 2x3x2 (rank 6) * 4 = 24 for first 8 cols? No, that's schoolbook.
    
    # Let me try a random search for low-rank decompositions
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in [30, 35, 40, 42, 44, 45]:
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Simple check: can we find a valid decomposition?
            # This is a non-convex problem, so random init is unlikely to work
            # But let's try a structured approach instead
            
            # Structured: for each j, define linear combos of k
            # U[i*m+j, c] depends on i, j, c
            # V[j*p+k, c] depends on j, k, c  
            # W[k*n+i, c] depends on k, i, c
            
            # Try: fix U and V, solve for W
            # Or use a greedy approach
            
            # For now, skip random and use a known structure
            pass
    
    # Use a known result: try rank 44 decomposition
    # Based on: 4 blocks of 2x3x2, each with rank 11 (Strassen-like)
    # 4 * 11 = 44
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Standard Strassen for 2x2x2
        U = [
            [1, 0, 0, 0, 0, 0, 0],  # a
            [0, 0, 1, 0, 0, 0, 0],  # b  
            [0, 1, 0, 0, 0, 0, 0],  # c
            [0, 0, 0, 1, 0, 0, 0],  # d
        ]
        V = [
            [1, 0, 0, 0],  # e
            [0, 1, 0, 0],  # f
            [0, 0, 1, 0],  # g
            [0, 0, 0, 1],  # h
        ]
        # Wait, this isn't right. Let me use the standard Strassen.
        # For 2x2 matrices:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-g)
        # m4 = d(h-g)
        # m5 = (a+b)h
        # m6 = (c-d)(g+h)
        # m7 = (a-b)(e+f)
        # p11 = m1 + m4 - m5 + m7
        # p12 = m3 + m5
        # p21 = m2 + m4
        # p22 = m1 - m2 + m3 + m6
        
        # U: 4x7 (a,b,c,d)
        U_s = [
            [1, 0, 1, 0, 1, 0, 1],  # a
            [0, 0, 0, 0, 1, 0, -1], # b
            [0, 1, 0, 0, 0, 1, 0],  # c
            [1, 1, 0, 1, 0, -1, 0], # d
        ]
        # V: 4x7 (e,f,g,h)
        V_s = [
            [1, 1, 0, 0, 0, 0, 1],  # e
            [0, 0, 1, 0, 0, 0, 1],  # f
            [0, 0, -1, 0, 0, 1, 0], # g
            [1, 0, 0, 1, 1, 1, 0],  # h
        ]
        # W: 4x7 (p11,p12,p21,p22)
        W_s = [
            [1, 0, 0, 0],  # p11 gets m1+m4-m5+m7
            [0, 1, 0, 0],  # p12 gets m3+m5
            [0, 0, 1, 0],  # p21 gets m2+m4
            [0, 0, 0, 1],  # p22 gets m1-m2+m3+m6
        ]
        # But we need W[k*n+i, c] convention
        # For 2x2x2: n=2, m=2, p=2
        # W has shape (p*n, rank) = (4, 7)
        # W[k*n+i, c] where k is col of B, i is row of A
        # p11: i=0,k=0 -> W[0*2+0] = W[0]
        # p12: i=0,k=1 -> W[0*2+1] = W[1]
        # p21: i=1,k=0 -> W[1*2+0] = W[2]
        # p22: i=1,k=1 -> W[1*2+1] = W[3]
        
        # So W matrix (4x7):
        W_s = [
            [1, 0, 0, 0, -1, 0, 1],   # p11: m1+m4-m5+m7
            [0, 0, 1, 0, 1, 0, 0],    # p12: m3+m5
            [0, 1, 0, 1, 0, 0, 0],    # p21: m2+m4
            [1, -1, 1, 0, 0, 1, 0],   # p22: m1-m2+m3+m6
        ]
        
        return U_s, V_s, W_s
    
    # Now build a 2x3x8 decomposition using 4 blocks of 2x3x2
    # For each block of 2 columns, use a 2x3x2 decomposition
    # 2x3x2: we can use Strassen for the 2x2 part and handle the 3rd row of B
    
    # Actually, let me try a simpler structural approach:
    # Use 6 rank-1 terms for each of 7 column-groups... 
    # 
    # Let me just try to return the schoolbook for now but with a slight improvement
    # by checking if we can do better with a simple heuristic
    
    # Try: can we do rank 46 by merging two schoolbook terms?
    # Unlikely for exact decomposition
    
    # For this structural change, let me implement a proper search
    # using the fact that we can use linear algebra to find decompositions
    
    # Try a rank-44 construction based on block Strassen
    # 4 blocks of 2 columns each
    # For each 2x3x2 block, use rank 11 (Strassen 7 + 4 extra for the 3rd B-row)
    
    # Actually, let me try a cleaner approach:
    # Decompose M_{2,3,8} as sum of 4 M_{2,3,2} tensors
    # Each M_{2,3,2} can be done in rank 6 (schoolbook) or less
    
    # For M_{2,3,2}: rank is at most 6, likely exactly 6
    # So 4*6 = 24? No, that's not right because the tensors overlap in structure
    
    # I think the schoolbook rank 48 is hard to beat without a clever algorithm
    # Let me try a different structural family: use a Vandermonde-based approach
    
    # For now, return the best we have
    if time.time() > deadline:
        return best_result
    
    # Try a simple rank reduction: see if rank 47 works
    # This requires finding a specific decomposition, which is complex
    
    # Return schoolbook as the safe choice
    return best_result
# EVOLVE-BLOCK-END
