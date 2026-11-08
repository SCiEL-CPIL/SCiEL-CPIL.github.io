"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses structured search with deadline guard. Returns schoolbook if search fails.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 120  # 120s budget with safety margin
    
    # Schoolbook fallback (rank = n*m*p = 80)
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
    
    # Try to find lower rank decompositions
    # For matrix multiplication, the tensor has structure that may allow rank < n*m*p
    
    # Strategy: Use the fact that matrix multiplication tensor T[i,j,k] = 1 if indices align
    # T is a 3D tensor of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But for matrix multiplication, better lower bounds exist
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try a few candidate ranks below schoolbook
    # Known results: For 4x4*4x5, the optimal rank is likely around 40-50
    # Let's try to find a decomposition with rank < 80
    
    # Use a structured approach: fix some patterns and solve linear systems
    # The tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # This is a bilinear problem in U,V,W. We can fix one matrix and solve for others.
    
    # Approach: Randomized search with verification
    # For each candidate rank r, try random initializations and verify
    
    candidate_ranks = [50, 45, 40, 36, 32, 28, 24, 20]
    
    for r in candidate_ranks:
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 30:
                break
                
            # Random sparse initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Initialize with random 1s in sparse pattern
            for c in range(r):
                # Each column of U, V, W should have some structure
                # Randomly select positions
                u_idx = random.randint(0, n * m - 1)
                v_idx = random.randint(0, m * p - 1)
                w_idx = random.randint(0, p * n - 1)
                
                # Ensure consistency: if U[i*m+j, c] != 0 and V[j'*p+k, c] != 0
                # then j should equal j' for the product to contribute to T[i,j,k]
                
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
            
            # Verify this decomposition
            # Compute the tensor from U, V, W and compare to target
            tensor = np.zeros((n * m, m * p, p * n))
            for c in range(r):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
                # Outer product
                for i in range(n * m):
                    if uc[i] != 0:
                        for j in range(m * p):
                            if vc[j] != 0:
                                for k in range(p * n):
                                    if wc[k] != 0:
                                        tensor[i, j, k] += uc[i] * vc[j] * wc[k]
            
            # Target tensor
            target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i * m + j, j * p + k, k * n + i] = 1
            
            if np.allclose(tensor, target, atol=1e-10):
                # Convert to lists
                U_list = [[int(U[i, c]) for c in range(r)] for i in range(n * m)]
                V_list = [[int(V[i, c]) for c in range(r)] for i in range(m * p)]
                W_list = [[int(W[i, c]) for c in range(r)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
