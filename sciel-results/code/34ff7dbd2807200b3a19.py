import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Schoolbook fallback: rank = n*m*p = 56
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Known results suggest rank for 2x4x7 might be around 28-35
    
    # Strategy: Use a structured search based on block decomposition
    # We can decompose the 2x4 matrix multiplication into smaller blocks
    # A (2x4) * B (4x7) = C (2x7)
    
    # Let's try a rank-28 decomposition first (theoretical minimum)
    # If that fails, try higher ranks
    
    # First, let's try a specific construction based on known results
    # For 2x4x7, we can try to use the structure of the matrices
    
    # Alternative approach: Use random search with better initialization
    # and local optimization
    
    # Try rank 28
    target_rank = 28
    if time.time() < deadline - 30:
        result = find_decomposition_structured(n, m, p, target_rank, deadline)
        if result:
            return result
    
    # Try rank 30
    if time.time() < deadline - 20:
        result = find_decomposition_structured(n, m, p, 30, deadline)
        if result:
            return result
    
    # Try rank 35
    if time.time() < deadline - 10:
        result = find_decomposition_structured(n, m, p, 35, deadline)
        if result:
            return result
    
    return {"U": U_school, "V": V_school, "W": W_school}

def find_decomposition_structured(n, m, p, rank, deadline):
    """
    Find a rank-rank decomposition for multiplying n x m by m x p matrices.
    
    The tensor T[i,j,k] = 1 for all valid i,j,k
    We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    
    # Initialize with a structured pattern
    # Use a pattern that might work: distribute the 1s across the rank dimension
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Try a specific construction:
    # For each (i,j,k), we need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # One approach: use orthogonal-like patterns
    
    # Let's try a different initialization: 
    # Set U[i*m+j, r] = 1 if r corresponds to some function of (i,j)
    # This is tricky, so let's use random initialization with local search
    
    # Better initialization: use a pattern based on the dimensions
    # For n=2, m=4, p=7, rank=28
    
    # Let's try: U[i*m+j, r] = 1 for r in some set
    # V[j*p+k, r] = 1 for r in some set  
    # W[k*n+i, r] = 1 for r in some set
    # And ensure that for each (i,j,k), exactly one r has all three = 1
    
    # This is equivalent to finding a 3D array R[i,j,k] in [0,rank) such that:
    # For each r, the set of (i,j,k) with R[i,j,k]=r forms a "slice"
    # And U[i*m+j, r] = 1 iff exists k with R[i,j,k]=r
    # V[j*p+k, r] = 1 iff exists i with R[i,j,k]=r
    # W[k*n+i, r] = 1 iff exists j with R[i,j,k]=r
    
    # This is a combinatorial design problem
    
    # Let's try a simpler approach: random search with local optimization
    
    # Initialize with random values from {-1, 0, 1}
    U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
    
    def compute_error(U, V, W):
        """Compute the number of (i,j,k) where the sum != 1"""
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        error += 1
        return error
    
    best_error = compute_error(U, V, W)
    if best_error == 0:
        return {"U": U, "V": V, "W": W}
    
    # Local search with simulated annealing
    current_error = best_error
    temperature = 10.0
    cooling_rate = 0.99
    
    while time.time() < deadline - 5 and temperature > 0.01:
        # Pick a random element to change
        mat = random.choice(["U", "V", "W"])
        
        if mat == "U":
            idx = random.randint(0, n*m-1)
            r = random.randint(0, rank-1)
            old_val = U[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            U[idx][r] = new_val
            new_error = compute_error(U, V, W)
            
            if new_error < current_error or random.random() < np.exp(-(new_error - current_error) / temperature):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    if best_error == 0:
                        return {"U": U, "V": V, "W": W}
            else:
                U[idx][r] = old_val
        elif mat == "V":
            idx = random.randint(0, m*p-1)
            r = random.randint(0, rank-1)
            old_val = V[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            V[idx][r] = new_val
            new_error = compute_error(U, V, W)
            
            if new_error < current_error or random.random() < np.exp(-(new_error - current_error) / temperature):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    if best_error == 0:
                        return {"U": U, "V": V, "W": W}
            else:
                V[idx][r] = old_val
        else:
            idx = random.randint(0, p*n-1)
            r = random.randint(0, rank-1)
            old_val = W[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            W[idx][r] = new_val
            new_error = compute_error(U, V, W)
            
            if new_error < current_error or random.random() < np.exp(-(new_error - current_error) / temperature):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    if best_error == 0:
                        return {"U": U, "V": V, "W": W}
            else:
                W[idx][r] = old_val
        
        temperature *= cooling_rate
    
    return None

# Alternative: Try a known construction for small matrix multiplication
# For 2x2x2, rank 7 (Strassen) is known
# For 2x4x7, we might be able to compose Strassen-like blocks

def construct_strassen_based():
    """Try to construct a decomposition based on Strassen's algorithm"""
    n, m, p = 2, 4, 7
    
    # We can decompose the 2x4 matrix into blocks
    # A = [A11 A12; A21 A22] where each block is 1x2
    # B = [B11 B12; B21 B22; B31 B32; B41 B42] where each block is 1x7/4... this doesn't work cleanly
    
    # Alternative: use the fact that 4 = 2+2
    # A (2x4) = A1 (2x2) | A2 (2x2)
    # B (4x7) = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    
    # We can compute A1*B1 (2x2 * 2x7) and A2*B2 (2x2 * 2x7) separately
    # Each 2x2 * 2x7 multiplication can be done with rank at most 7*2 = 14 (schoolbook)
    # But we can do better: 2x2 * 2x1 has rank 7 (Strassen), so 2x2 * 2x7 has rank 7*7 = 49? No...
    
    # Actually, 2x2 * 2x7: we can write this as 7 separate 2x2 * 2x1 multiplications
    # Each 2x2 * 2x1 has rank 7 (Strassen for 2x2x1 is rank 7? Let me check)
    # Actually, 2x2 * 1x2 has rank 7 (Strassen)
    # 2x2 * 2x1: this is similar, rank should be 7
    
    # So 2x2 * 2x7 would have rank 7*7 = 49 if we do each column separately
    # But we can do better by batching
    
    # This approach is getting complicated. Let's stick with the random search.
    return None
