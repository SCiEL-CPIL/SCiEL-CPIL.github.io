"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    # Deadline guard: 150s wall-clock limit, use 115s for safety
    start_time = time.time()
    deadline = start_time + 115
    
    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try structured rank reductions
    # For small matrices, known results suggest rank reductions are possible
    # Try ranks from 99 down to 80 with structured approaches
    
    def try_structured_rank_reduction(target_rank):
        """Try to find a rank-r decomposition using structured patterns."""
        if time.time() > deadline:
            return None
        
        # Strategy 1: Use block structure
        # For 4x5 * 5x5, we can group terms
        # Try to merge schoolbook terms using algebraic identities
        
        # Generate random structured candidates
        for trial in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize with schoolbook and try to reduce
            U = [row[:] for row in U_school[:nm]]
            V = [row[:] for row in V_school[:mp]]
            W = [row[:] for row in W_school[:pn]]
            
            # Try to eliminate some terms by setting them to 0 and adjusting others
            # This is a heuristic approach
            
            # Randomly select terms to try to eliminate
            indices = list(range(schoolbook_rank))
            random.shuffle(indices)
            
            eliminated = 0
            for idx in indices:
                if eliminated >= schoolbook_rank - target_rank:
                    break
                if time.time() > deadline:
                    break
                
                # Try to eliminate this term by adjusting others
                # For simplicity, just set to 0 and verify
                for i in range(nm):
                    U[i][idx] = 0
                for j in range(mp):
                    V[j][idx] = 0
                for k in range(pn):
                    W[k][idx] = 0
                
                if verify(U, V, W):
                    eliminated += 1
            
            if eliminated >= schoolbook_rank - target_rank:
                # Trim to target_rank
                U = [row[:target_rank] for row in U]
                V = [row[:target_rank] for row in V]
                W = [row[:target_rank] for row in W]
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try random search with small rank reductions
    def try_random_rank_reduction(target_rank, num_trials=20):
        """Try random search for rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Use small random perturbations from schoolbook
            U = []
            for i in range(nm):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = U_school[i][c]
                        # Small probability of perturbation
                        if random.random() < 0.1:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                U.append(row)
            
            V = []
            for j in range(mp):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = V_school[j][c]
                        if random.random() < 0.1:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                V.append(row)
            
            W = []
            for k in range(pn):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = W_school[k][c]
                        if random.random() < 0.1:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                W.append(row)
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 99 down to 85
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 0), -1):
        if time.time() > deadline:
            break
        
        # Try structured approach first
        result = try_structured_rank_reduction(target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
        
        # Try random approach
        result = try_random_rank_reduction(target_rank, num_trials=10)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
