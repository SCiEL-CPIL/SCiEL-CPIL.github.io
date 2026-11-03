"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_and_update(U, V, W, rank, n, m, p):
        """Verify decomposition and update best if valid and better."""
        nonlocal best_rank, best_result
        if rank >= best_rank:
            return False
        if verify_decomposition(U, V, W, rank, n, m, p):
            best_rank = rank
            best_result = {
                "U": U.tolist() if isinstance(U, np.ndarray) else U,
                "V": V.tolist() if isinstance(V, np.ndarray) else V,
                "W": W.tolist() if isinstance(W, np.ndarray) else W
            }
            return True
        return False
    
    # Strategy 1: Structured decomposition using block structure
    # 3x3 * 3x6 can be seen as two 3x3*3x3 multiplications
    # Use known lower-rank decompositions for 3x3 matmul
    
    # Winograd-style decomposition for 3x3 matmul (rank 23 or less)
    # Then combine two such decompositions
    
    # First, let's try a simple block approach:
    # C = A * B where A is 3x3, B is 3x6
    # Split B into B1 (3x3) and B2 (3x3)
    # C1 = A * B1, C2 = A * B2
    # Total rank = rank_3x3 + rank_3x3
    
    # For 3x3 matmul, we can use the standard rank-27 or try lower
    # Let's implement a systematic search for 3x3 matmul first
    
    def find_3x3_decomposition(max_rank=27, time_limit=30):
        """Find a low-rank decomposition for 3x3 * 3x3 matmul."""
        n3, m3, p3 = 3, 3, 3
        best_r = n3 * m3 * p3
        best_U = None
        best_V = None
        best_W = None
        
        deadline_3x3 = time.time() + time_limit
        
        # Try various ranks
        for rank in range(min(max_rank, best_r - 1), 20, -1):
            if time.time() > deadline_3x3:
                break
            
            # Random search with structured initialization
            for attempt in range(20):
                if time.time() > deadline_3x3:
                    break
                    
                rng = np.random.RandomState(attempt * 12345 + rank * 678)
                
                # Use values from {-1, 0, 1} with some density
                density = 0.5
                U = rng.choice([-1, 0, 1], size=(9, rank))
                V = rng.choice([-1, 0, 1], size=(9, rank))
                W = rng.choice([-1, 0, 1], size=(9, rank))
                
                # Apply sparsity
                mask_u = rng.random((9, rank)) < density
                mask_v = rng.random((9, rank)) < density
                mask_w = rng.random((9, rank)) < density
                U = U * mask_u
                V = V * mask_v
                W = W * mask_w
                
                # Verify
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.dot(U[i*3+j], np.multiply(V[j*3+k], W[k*3+i]))
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_r = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        return best_r, best_U, best_V, best_W
    
    # Try to find a good 3x3 decomposition
    if time.time() < deadline - 60:
        r3, U3, V3, W3 = find_3x3_decomposition(max_rank=27, time_limit=40)
        
        if r3 < 27:
            # Combine two 3x3 decompositions for 3x3*3x6
            # C[:, 0:3] = A * B1, C[:, 3:6] = A * B2
            rank_combined = 2 * r3
            
            if rank_combined < best_rank:
                # Construct the combined decomposition
                U_comb = np.zeros((9, rank_combined))
                V_comb = np.zeros((18, rank_combined))
                W_comb = np.zeros((18, rank_combined))
                
                # First block: k in 0..2
                U_comb[:, 0:r3] = U3
                V_comb[0:9, 0:r3] = V3  # j*p+k for k in 0..2, p=6
                # Wait, V is indexed by j*p+k where p=6, so for B1 (cols 0-2), V[j*6+k]
                # But V3 is for 3x3, indexed by j*3+k
                # Need to remap: for the first 3 columns of B, k=0,1,2
                # V_comb[j*6+k, c] = V3[j*3+k, c] for k=0,1,2
                for j in range(3):
                    for k in range(3):
                        V_comb[j*6+k, 0:r3] = V3[j*3+k, 0:r3]
                
                # W_comb[k*n+i, c] = W3[k*3+i, c] for k=0,1,2
                for k in range(3):
                    for i in range(3):
                        W_comb[k*3+i, 0:r3] = W3[k*3+i, 0:r3]
                
                # Second block: k in 3..5
                U_comb[:, r3:2*r3] = U3
                for j in range(3):
                    for k in range(3, 6):
                        V_comb[j*6+k, r3:2*r3] = V3[j*3+(k-3), 0:r3]
                
                for k in range(3, 6):
                    for i in range(3):
                        W_comb[k*3+i, r3:2*r3] = W3[k*3+(k-3), 0:r3]
                
                if verify_and_update(U_comb, V_comb, W_comb, rank_combined, n, m, p):
                    print(f"Found combined decomposition with rank {rank_combined}")
    
    # Strategy 2: Direct random search on the full problem
    for rank in range(50, 25, -1):
        if time.time() > deadline - 15:
            break
        
        # Try several random initializations
        max_attempts = 5 if rank > 40 else 15
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(attempt * 7919 + rank * 31 + 1000)
            
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0, 0, 0.5, 1]
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
            W = rng.choice(vals, size=(p*n, rank))
            
            # Apply sparsity
            density = 0.3
            mask_u = rng.random((n*m, rank)) < density
            mask_v = rng.random((m*p, rank)) < density
            mask_w = rng.random((p*n, rank)) < density
            U = U * mask_u
            V = V * mask_v
            W = W * mask_w
            
            if verify_and_update(U, V, W, rank, n, m, p):
                print(f"Found direct decomposition with rank {rank}")
                break
    
    # Strategy 3: Local search - perturb and refine
    if time.time() < deadline - 20:
        # Start from current best and try local modifications
        U_curr = np.array(best_result["U"], dtype=float)
        V_curr = np.array(best_result["V"], dtype=float)
        W_curr = np.array(best_result["W"], dtype=float)
        curr_rank = best_rank
        
        # Try to reduce rank by finding linear dependencies
        # This is complex, so we'll just do random perturbations
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            U_new = U_curr.copy()
            V_new = V_curr.copy()
            W_new = W_curr.copy()
            
            # Perturb a few entries
            for _ in range(3):
                idx = rng.randint(0, U_new.shape[0]-1)
                jdx = rng.randint(0, U_new.shape[1]-1)
                U_new[idx, jdx] += rng.choice([-1, 1]) * 0.5
                
                idx = rng.randint(0, V_new.shape[0]-1)
                jdx = rng.randint(0, V_new.shape[1]-1)
                V_new[idx, jdx] += rng.choice([-1, 1]) * 0.5
                
                idx = rng.randint(0, W_new.shape[0]-1)
                jdx = rng.randint(0, W_new.shape[1]-1)
                W_new[idx, jdx] += rng.choice([-1, 1]) * 0.5
            
            # Snap to half-integers
            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2
            
            if verify_and_update(U_new, V_new, W_new, curr_rank, n, m, p):
                U_curr = U_new
                V_curr = V_new
                W_curr = W_new
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
