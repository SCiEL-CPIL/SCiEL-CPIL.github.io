"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Schoolbook decomposition (always valid, rank = n*m*p = 42)
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

    def verify(U, V, W, n, m, p):
        """Verify that U,V,W give a valid tensor decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(s - 1) > 1e-9:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > 1e-9:
                            return False
        return True

    def verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[a,b,c] = 1 for valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries: T[a,b,c] = 0 for invalid combos
        # Only check a subset for speed, then full check if subset passes
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True

    # Constructive approach: exploit structure to find lower rank
    # Key insight: For 2x3 @ 3x7, we can group computations.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Let's try to find a decomposition with rank < 42.
    # The tensor has 42 non-zero entries. 
    # Lower bound: max over slices of the number of non-zeros in that slice.
    # A-slice (fixed a=(i,j)): 7 non-zeros (one for each k)
    # B-slice (fixed b=(j,k)): 2 non-zeros (one for each i)  
    # C-slice (fixed c=(k,i)): 3 non-zeros (one for each j)
    # So rank >= 7. But that's very loose.
    
    # Better approach: try to find a structured decomposition.
    # For each k, we need to compute the 2x1 vector:
    # C[:,k] = A @ B[:,k]
    # where A is 2x3 and B[:,k] is 3x1.
    # This is a 2x3 @ 3x1 matrix multiplication, which has tensor rank 6 (schoolbook).
    # But we can do better: rank of 2x3@3x1 is actually 4 (or less?).
    
    # Actually, let's just do a smart random search with better initialization.
    # Start from schoolbook and try to merge terms.
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Try a structured approach: for each pair (i,k), the 3 products can potentially
    # be reduced. But since i is only 0,1 and k is 0..6, we have 14 outputs.
    
    # Let's try to find rank 36, 35, ... by random search with better heuristics.
    
    def random_search_structured(rank, max_iters=3000, seed=42):
        """Random search with structured initialization."""
        random.seed(seed)
        np.random.seed(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            
            # Initialize with zeros, then add random rank-1 terms
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Add random rank-1 components
            for r in range(rank):
                # Random sparse vectors
                u_vec = np.random.choice([-1, -0.5, 0, 0, 0, 0, 0.5, 1], size=n * m)
                v_vec = np.random.choice([-1, -0.5, 0, 0, 0, 0, 0.5, 1], size=m * p)
                w_vec = np.random.choice([-1, -0.5, 0, 0, 0, 0, 0.5, 1], size=p * n)
                U[:, r] = u_vec
                V[:, r] = v_vec
                W[:, r] = w_vec
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_fast(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try decreasing rank from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 15.0:
            break
        result = random_search_structured(r, max_iters=1000, seed=r * 12345)
        if result is not None:
            best_result = result
            best_rank = r
            break

    # If we found a better rank, try to improve further
    if best_rank < n * m * p:
        for r in range(best_rank - 1, 24, -1):
            if time.time() > deadline - 10.0:
                break
            result = random_search_structured(r, max_iters=500, seed=r * 67890)
            if result is not None:
                best_result = result
                best_rank = r
                break

    return best_result
# EVOLVE-BLOCK-END
