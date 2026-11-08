"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    deadline = time.time() + 140  # safety margin
    
    # Build the multiplication tensor T[i, j, k] = 1 for all valid (i,j,k)
    # T has shape (n, m, p) where T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The rank-r decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Schoolbook: rank = n*m*p = 75
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces the tensor."""
        # For each (i, j, k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        # Build the 3D array of the decomposition
        result = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    result[i, j, k] = s
        # Check all entries equal 1
        return np.allclose(result, 1.0, atol=1e-9)
    
    def check_half_integer(arr):
        """Check if all entries are multiples of 0.5."""
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=1e-9)
    
    def is_valid(U, V, W, rank):
        if not (check_half_integer(U) and check_half_integer(V) and check_half_integer(W)):
            return False
        return validate(U, V, W, rank)
    
    # Try to reduce rank by removing columns and re-solving
    # Strategy: for each column, try setting it to zero and see if we can adjust others
    # More practical: use local search with small perturbations
    
    # Try structured reductions: group terms and use Strassen-like tricks on 2x2x2 sub-blocks
    # For 3x5x5, we can partition into blocks and apply Strassen where possible
    
    # First, try a direct search over small integer/half-integer entries for lower ranks
    # This is expensive, so we'll do a limited search
    
    # Try rank from 74 down to some minimum, with randomized initialization
    min_rank_to_try = 60  # Don't go too low initially
    
    for target_rank in range(rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts at this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small half-integers
            U_try = np.zeros((n * m, target_rank), dtype=float)
            V_try = np.zeros((m * p, target_rank), dtype=float)
            W_try = np.zeros((p * n, target_rank), dtype=float)
            
            # Use a structured initialization: start from schoolbook and randomly merge columns
            # Take first target_rank columns from schoolbook, then randomly adjust
            if target_rank >= 1:
                # Copy some columns from schoolbook
                num_copy = min(target_rank, rank)
                U_try[:, :num_copy] = U[:, :num_copy]
                V_try[:, :num_copy] = V[:, :num_copy]
                W_try[:, :num_copy] = W[:, :num_copy]
                
                # Randomly perturb some entries to allow for better packing
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    # Pick random entries to perturb
                    r_idx = random.randint(0, n * m - 1)
                    c_idx = random.randint(0, target_rank - 1)
                    delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                    U_try[r_idx, c_idx] += delta
                    U_try[r_idx, c_idx] = round(U_try[r_idx, c_idx] * 2) / 2
                    
                    r_idx = random.randint(0, m * p - 1)
                    c_idx = random.randint(0, target_rank - 1)
                    delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                    V_try[r_idx, c_idx] += delta
                    V_try[r_idx, c_idx] = round(V_try[r_idx, c_idx] * 2) / 2
                    
                    r_idx = random.randint(0, p * n - 1)
                    c_idx = random.randint(0, target_rank - 1)
                    delta = random.choice([-0.5, 0.5, -1.0, 1.0])
                    W_try[r_idx, c_idx] += delta
                    W_try[r_idx, c_idx] = round(W_try[r_idx, c_idx] * 2) / 2
            
            # Validate
            if is_valid(U_try, V_try, W_try, target_rank):
                best_U, best_V, best_W, best_rank = U_try.copy(), V_try.copy(), W_try.copy(), target_rank
                # Found a better rank, continue searching for even lower
                break  # Break out of attempts, continue to next (lower) rank
        
        # If we found a solution at this rank, we can stop or continue
        if best_rank == target_rank:
            # Try to go lower
            continue
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
