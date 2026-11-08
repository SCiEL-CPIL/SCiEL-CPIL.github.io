"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W):
    """Vectorized verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U)  # (n*m, rank)
    V_arr = np.array(V)  # (m*p, rank)
    W_arr = np.array(W)  # (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _structured_search(r, time_limit=30.0):
    """
    Structured search for rank-r decomposition.
    Strategy: Start from schoolbook (rank 48) and try to merge pairs of columns.
    Also try random structured patterns with finite-field-inspired sparsity.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Approach 1: Start from schoolbook and try to merge columns
    # Schoolbook has 48 columns, each corresponding to (i,j,k)
    # To reduce rank, we need to find columns that can be combined
    
    # Approach 2: Use a greedy construction
    # For each (i,j,k) triple, assign it to a "slot" c
    # The constraint is: for each (i,j,k), sum over c where (i,j,k) is assigned:
    #   U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # And for each (i,j,k), only the slots assigned to it should contribute
    
    # Approach 3: Random search with structured initialization
    # Use the fact that entries can be 0, ±1, ±0.5
    
    best_result = None
    
    # Try multiple random structured initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + r * 104729)
        
        # Initialize with sparse random pattern
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Each column gets a random pattern
        for c in range(r):
            # Randomly decide which (i,j,k) this column contributes to
            # Use a structured pattern: pick a subset of (i,j,k) triples
            # For each (i,j,k), assign random values from {0, 1, -1, 0.5, -0.5}
            
            # Strategy: each column "covers" a random subset of triples
            # The product U[i,j,c]*V[j,k,c]*W[k,i,c] must sum to 1 over all c
            # for each (i,j,k)
            
            # Simple approach: for each (i,j,k), randomly assign it to 1-3 columns
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # With some probability, this column contributes to (i,j,k)
                        if random.random() < 0.3:  # ~30% chance
                            u_val = random.choice([1, -1, 0.5, -0.5])
                            v_val = random.choice([1, -1, 0.5, -0.5])
                            w_val = random.choice([1, -1, 0.5, -0.5])
                            U[i * m + j][c] = u_val
                            V[j * p + k][c] = v_val
                            W[k * n + i][c] = w_val
        
        # Local search to fix errors
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Find errors
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                        if abs(s - 1) > 1e-9:
                            errors.append((i, j, k, s))
            
            if not errors:
                return U, V, W
            
            i, j, k, s = random.choice(errors)
            diff = 1 - s
            
            # Try to fix by modifying one entry
            # The derivative w.r.t. U[i,j,c] is V[j,k,c]*W[k,i,c]
            # So we want to find c and new_val such that:
            # old_U * V * W + (new_val - old_U) * V * W = 1
            # => new_val * V[j,k,c] * W[k,i,c] = diff + old_U * V[j,k,c] * W[k,i,c]
            
            c = random.randrange(r)
            vw = V[j*p+k][c] * W[k*n+i][c]
            if abs(vw) > 1e-12:
                # new_U = (1 - (s - U[i*m+j][c]*vw)) / vw
                # = (1 - s + U[i*m+j][c]*vw) / vw
                # = diff / vw + U[i*m+j][c]
                new_val = U[i*m+j][c] + diff / vw
                # Round to nearest half-integer
                new_val_rounded = round(new_val * 2) / 2
                old = U[i*m+j][c]
                U[i*m+j][c] = new_val_rounded
                if _verify(U, V, W):
                    return U, V, W
                U[i*m+j][c] = old
            
            # Try V
            uw = U[i*m+j][c] * W[k*n+i][c]
            if abs(uw) > 1e-12:
                new_val = V[j*p+k][c] + diff / uw
                new_val_rounded = round(new_val * 2) / 2
                old = V[j*p+k][c]
                V[j*p+k][c] = new_val_rounded
                if _verify(U, V, W):
                    return U, V, W
                V[j*p+k][c] = old
            
            # Try W
            uv = U[i*m+j][c] * V[j*p+k][c]
            if abs(uv) > 1e-12:
                new_val = W[k*n+i][c] + diff / uv
                new_val_rounded = round(new_val * 2) / 2
                old = W[k*n+i][c]
                W[k*n+i][c] = new_val_rounded
                if _verify(U, V, W):
                    return U, V, W
                W[k*n+i][c] = old
    
    return None

def _merge_columns(U, V, W):
    """
    Try to merge two columns of a valid decomposition into one.
    If columns a and b can be combined, we reduce rank by 1.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for a in range(rank):
        for b in range(a + 1, rank):
            # Check if columns a and b can be merged
            # We need: U_a * V_a * W_a + U_b * V_b * W_b = U_new * V_new * W_new
            # This is generally not possible unless the patterns are compatible
            
            # Check if the support patterns are compatible
            # For each (i,j,k), if both columns a and b contribute,
            # we need a way to combine them
            
            compatible = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ua = U[i*m+j][a]
                        va = V[j*p+k][a]
                        wa = W[k*n+i][a]
                        ub = U[i*m+j][b]
                        vb = V[j*p+k][b]
                        wb = W[k*n+i][b]
                        
                        contrib_a = ua * va * wa
                        contrib_b = ub * vb * wb
                        
                        if contrib_a != 0 and contrib_b != 0:
                            # Both contribute - need to check if they can be merged
                            # This is complex; skip for now
                            compatible = False
                            break
                    if not compatible:
                        break
                if not compatible:
                    break
            if not compatible:
                continue
            
            # If compatible, create new column
            U_new = [row[:] for row in U]
            V_new = [row[:] for row in V]
            W_new = [row[:] for row in W]
            
            # Merge b into a
            for i in range(n):
                U_new[i*m][a] = U_new[i*m][a] + U_new[i*m][b]  # This is wrong, need to be careful
            # Actually, this approach doesn't work simply. Skip.
            break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s budget
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Build schoolbook as fallback
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find better decompositions
    # Known lower bound: max(n*p, n*m, m*p) = 16
    # But practical bounds are higher
    
    # Try structured search for various ranks
    for r in range(rank_schoolbook - 1, max(15, rank_schoolbook - 25), -1):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Allocate time based on how many ranks are left
        ranks_remaining = rank_schoolbook - 1 - r + max(15, rank_schoolbook - 25)
        time_per_rank = max(2.0, min(20.0, time_remaining / max(1, ranks_remaining)))
        
        result = _structured_search(r, time_limit=time_per_rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = r
            if r <= 30:
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
