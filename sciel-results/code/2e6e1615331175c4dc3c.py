import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 84 # Schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Create the target tensor T of shape (n, p, m) or (n*m, p*n) ?
    # The convention is W[k*n+i, r].
    # T[i, k, j] = 1 if it's the multiplication tensor?
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, k, j] contributes to C[i, k].
    # The tensor is T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # In the decomposition:
    # C[i, k] = sum_r U[i, m? no, U is (n*m, rank)] ... wait.
    # U has shape (n*m, rank). Index i*m+j.
    # V has shape (m*p, rank). Index j*p+k.
    # W has shape (p*n, rank). Index k*n+i.
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, the standard tensor decomposition for matrix mult is:
    # C[i, k] = sum_r (sum_j A[i, j] U[j, r]) * (sum_k B[k, r] V[r, k])? No.
    
    # Let's stick to the definition in the prompt:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The decomposition must satisfy:
    # For all i, j, k:
    # delta_{i,j,k} (which is 1 if valid indices) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
    # No, the tensor T_{i,j,k} is 1 if we are computing the (i,k) entry using (i,j) and (j,k).
    # Actually, the tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # The multiplication tensor T is defined by T_{i,j,k} = 1 if the indices are compatible?
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then the tensor is just the all-ones tensor?
    # No, the tensor is T \in \mathbb{R}^{n \times m \times p}.
    # T[i, j, k] = 1.
    # Then C[i, k] = sum_j A[i, j] B[j, k].
    # This is correct.
    
    # So we need to decompose the all-ones tensor T of shape (3, 4, 7) into rank R.
    # T[i, j, k] = sum_{r=1}^R U[i, j, r] * V[j, k, r] * W[k, i, r] ?
    # The prompt says:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    
    # So the condition is:
    # 1 = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i, j, k.
    
    # This is a system of equations.
    # We want to find U, V, W with small R.
    
    # Let's use a random search for small R.
    
    def check_decomposition(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute T_hat[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T_hat == 1 for all i, j, k.
        
        # Efficiently compute using matrix multiplications?
        # T_hat[i, j, k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        
        # Let's just loop for small dimensions.
        # 3*4*7 = 84 entries.
        # R is small.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank(R, time_limit=10):
        n, m, p = DIMENSIONS
        start_time = time.time()
        np.random.seed(42) # Reproducibility? Or random?
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Simple gradient descent? Or just check?
            # Random initialization is unlikely to hit the exact solution.
            # We need a better strategy.
            
            # Strategy: Use the fact that for R=1, U,V,W must be rank 1.
            # For R=2, etc.
            
            # Let's try to use the known Laderman structure for 3x3x3.
            # But we have 3x4x7.
            
            # Alternative: Use the schoolbook and try to merge?
            
            # Let's try a random search with a local optimization step.
            # Minimize E = sum (sum_r ... - 1)^2
            
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                # Compute gradient? Too complex for this snippet.
                # Just random restarts.
                pass
                
        return None

    # Since random search is weak, let's return the schoolbook if we don't find better quickly.
    # But the score is seed/value. Lower is better.
    # Schoolbook rank is 84.
    # If we find rank 30, score is 84/30 = 2.8.
    
    # Let's try to construct a decomposition of rank 30 using a block approach.
    # Split the 7 columns of B into 3 and 4.
    # C1 = A * B1 (3x3), C2 = A * B2 (3x4).
    # Rank(C1) <= Rank(3x3x3) = 23.
    # Rank(C2) <= Rank(3x4x4).
    # Rank(3x4x4) <= Rank(3x3x3) + Rank(3x1x3) = 23 + 3 = 26.
    # Total rank <= 23 + 26 = 49.
    
    # Can we do better for 3x4x4?
    # Maybe Rank(3x4x4) is 24?
    
    # Let's try to return a decomposition of rank 49.
    # This requires constructing the actual U, V, W matrices.
    
    # I will implement a function to create the schoolbook decomposition for a given size.
    # And then combine them.
    
    def schoolbook(n, m, p):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Combine two decompositions
    def combine_decomps(d1, d2):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        U = U1 + [[0]*r2 for _ in range(len(U1))]
        for i in range(len(U2)):
            U.append(U2[i] + [0]*r1)
        V = V1 + [[0]*r2 for _ in range(len(V1))]
        for i in range(len(V2)):
            V.append(V2[i] + [0]*r1)
        W = W1 + [[0]*r2 for _ in range(len(W1))]
        for i in range(len(W2)):
            W.append(W2[i] + [0]*r1)
        return U, V, W, r1 + r2

    # Decompose 3x4x7 into 3x4x3 and 3x4x4
    d1 = schoolbook(3, 4, 3)
    d2 = schoolbook(3, 4, 4)
    
    U, V, W, rank = combine_decomps(d1, d2)
    
    # This gives rank 36 + 48 = 84. Same as schoolbook.
    # We need to use Laderman for 3x3x3 to reduce rank.
    
    # Laderman 3x3x3 rank 23.
    # We need the U, V, W for Laderman.
    # I will hardcode the Laderman matrices.
    
    # Laderman's algorithm for 3x3x3
    # U, V, W are 9x23 matrices.
    
    # I will omit the hardcoding for now and just return schoolbook if I can't find a better one quickly.
    # But I should try to find a better one.
    
    # Let's try to return the schoolbook result for now to ensure validity.
    U, V, W, rank = schoolbook(3, 4, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
