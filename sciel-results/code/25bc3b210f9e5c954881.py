import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Precompute target tensor indices
    target_indices = []
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                target_set.add((a, b, c))
    
    # Non-target indices (all possible triples minus targets)
    all_triples = []
    for a in range(nU):
        for b in range(nV):
            for c in range(nW):
                if (a, b, c) not in target_set:
                    all_triples.append((a, b, c))
    
    def compute_error(U, V, W, rank):
        """Compute the sum of squared errors for the decomposition."""
        error = 0.0
        # Target entries should be 1
        for a, b, c in target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            error += (val - 1.0) ** 2
        
        # Non-target entries should be 0
        for a, b, c in all_triples:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            error += val ** 2
        return error

    def is_valid(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is exact."""
        for a, b, c in target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val - 1.0) > tol:
                return False
        for a, b, c in all_triples:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val) > tol:
                return False
        return True

    # Build schoolbook decomposition as a fallback
    schoolbook_rank = n * m * p
    schoolbook_U = np.zeros((nU, schoolbook_rank))
    schoolbook_V = np.zeros((nV, schoolbook_rank))
    schoolbook_W = np.zeros((nW, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                schoolbook_U[a, col] = 1.0
                schoolbook_V[b, col] = 1.0
                schoolbook_W[c, col] = 1.0
                col += 1
    
    best_U = schoolbook_U.copy()
    best_V = schoolbook_V.copy()
    best_W = schoolbook_W.copy()
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition using randomized local search
    # Start from a lower bound and go up
    # Theoretical lower bound for 3x4x8 is at least max(n*m, m*p, p*n) = 32
    # But we can try even lower.
    
    min_rank_to_try = 20
    max_rank_to_try = min(schoolbook_rank - 1, 80)
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with small values
            U = np.random.uniform(-0.5, 0.5, (nU, rank))
            V = np.random.uniform(-0.5, 0.5, (nV, rank))
            W = np.random.uniform(-0.5, 0.5, (nW, rank))
            
            # Local search: iteratively perturb one entry and accept if error decreases
            current_error = compute_error(U, V, W, rank)
            
            # Simulated annealing style: start with high temperature, cool down
            temperature = 1.0
            cooling_rate = 0.99
            step_size = 0.1
            
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                # Choose a random matrix and entry to perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                elif mat_idx == 1:
                    mat = V
                else:
                    mat = W
                    
                row = random.randint(0, mat.shape[0] - 1)
                col_idx = random.randint(0, mat.shape[1] - 1)
                
                old_val = mat[row, col_idx]
                new_val = old_val + random.uniform(-step_size, step_size)
                mat[row, col_idx] = new_val
                
                new_error = compute_error(U, V, W, rank)
                
                # Accept if better, or with probability exp(-(new-old)/T)
                if new_error < current_error or random.random() < np.exp(-(new_error - current_error) / temperature):
                    current_error = new_error
                else:
                    mat[row, col_idx] = old_val
                    
                temperature *= cooling_rate
                step_size *= 0.995
                
                if current_error < 1e-12:
                    break
            
            if is_valid(U, V, W, rank):
                best_U = U
                best_V = V
                best_W = W
                best_rank = rank
                break  # Found a valid decomposition for this rank, try lower? No, we go up.
                # Actually, if we find a valid one at rank R, we should try to find a lower one.
                # But we are iterating rank from low to high. So if we find one at rank R, 
                # we can stop because we are looking for the minimal rank.
                # Wait, the loop is for rank in range(min, max). So if we find one at rank R,
                # we have found the minimal rank in this range. We can break the outer loop.
                break
        
        # If we found a valid decomposition at this rank, we can stop because we are going up from min.
        if best_rank == rank:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
