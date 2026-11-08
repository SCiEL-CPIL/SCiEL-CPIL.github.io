"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if this contributes to C[i,k]
    # Actually the tensor is T[i,j,k] = delta (the multiplication tensor)
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # The tensor for matmul has T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # the (i,k)-entry of C is sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # and the decomposition is: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W index is (k,i) not (k,i) in the tensor
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is the output column and i is the output row
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matmul: T[i,j,k] = 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Build the tensor T from U,V,W
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    max_rank = min(len(U[0]), len(V[0]), len(W[0]))
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(max_rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    errors += 1
    return errors == 0

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Start with schoolbook rank 100
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to reduce rank using a greedy elimination approach
    # Idea: try to find linear dependencies and eliminate terms
    # For now, try a structured approach using Strassen-like blocks
    
    # Approach: Use 2x2 Strassen blocks where possible
    # 4x5x5: pad to 4x8x8? No, let's try direct construction.
    
    # Let's try rank 78, 75, etc. by searching for valid decompositions
    # with half-integer entries in {-1, -0.5, 0, 0.5, 1}
    
    # Actually, let's try a more systematic approach:
    # Use the known result that R(4,4,4) <= 21, and build from there.
    # 4x5x5 = 4x4x4 + 4x1x5 + 4x4x1 + ... (not quite)
    
    # Simpler: try to find a valid decomposition at various ranks using random search
    # with structured initialization
    
    candidate_ranks = [80, 75, 70, 68, 65, 60, 55, 50]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try random search with half-integer entries
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            # Initialize with random half-integers
            vals = [-1, -0.5, 0, 0.5, 1]
            for i in range(n*m):
                for r in range(target_rank):
                    U[i][r] = vals[np.random.randint(0, 5)]
            for j in range(m*p):
                for r in range(target_rank):
                    V[j][r] = vals[np.random.randint(0, 5)]
            for k in range(p*n):
                for r in range(target_rank):
                    W[k][r] = vals[np.random.randint(0, 5)]
            
            # Check validity
            if _validate(U, V, W, n, m, p):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
