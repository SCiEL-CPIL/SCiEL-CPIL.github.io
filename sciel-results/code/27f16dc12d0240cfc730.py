"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 20
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    best_decomp = schoolbook()
    best_rank = schoolbook_rank
    
    # Values to search over
    values = [-1, -0.5, 0, 0.5, 1]
    
    # Try to find better ranks
    # We go from 12 up to 19. If we find 12, we stop.
    # If we don't find 12, we try 13, etc.
    # Actually, we should try 10, 11, 12... but 10 and 11 are very hard.
    # Let's try 12, 13, 14, 15.
    
    for target_rank in [12, 13, 14, 15, 16, 17, 18, 19]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        found = False
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Local Search
            # We want to minimize sum of squares of errors
            # Error for (i,j,k) is sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] - 1
            
            current_error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        current_error += (s - 1)**2
                        
            # If already perfect
            if current_error == 0:
                found = True
                best_decomp = (U, V, W, target_rank)
                best_rank = target_rank
                break
                
            # Hill Climbing
            # We will try to improve the error by changing one entry at a time
            # We do this for a fixed number of steps or until error is 0
            steps = 1000
            for step in range(steps):
                if time.time() > deadline:
                    break
                if current_error == 0:
                    break
                    
                # Pick a random tensor and a random entry
                # U: n*m x rank, V: m*p x rank, W: p*n x rank
                # Total entries: n*m*rank + m*p*rank + p*n*rank
                total_entries = (n*m + m*p + p*n) * target_rank
                idx = random.randint(0, total_entries - 1)
                
                # Determine which tensor and which entry
                um_size = n * m * target_rank
                vm_size = m * p * target_rank
                wm_size = p * n * target_rank
                
                if idx < um_size:
                    # U
                    row = idx // target_rank
                    col = idx % target_rank
                    tensor = 'U'
                    r, c = row, col
                elif idx < um_size + vm_size:
                    # V
                    idx_v = idx - um_size
                    row = idx_v // target_rank
                    col = idx_v % target_rank
                    tensor = 'V'
                    r, c = row, col
                else:
                    # W
                    idx_w = idx - um_size - vm_size
                    row = idx_w // target_rank
                    col = idx_w % target_rank
                    tensor = 'W'
                    r, c = row, col
                
                old_val = 0
                if tensor == 'U':
                    old_val = U[r][c]
                elif tensor == 'V':
                    old_val = V[r][c]
                else:
                    old_val = W[r][c]
                
                # Try all other values
                best_new_val = old_val
                min_new_error = current_error
                
                for new_val in values:
                    if new_val == old_val:
                        continue
                    
                    # Update
                    if tensor == 'U':
                        U[r][c] = new_val
                    elif tensor == 'V':
                        V[r][c] = new_val
                    else:
                        W[r][c] = new_val
                    
                    # Calculate new error
                    new_error = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for cc in range(target_rank):
                                    s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                                new_error += (s - 1)**2
                                
                    if new_error < min_new_error:
                        min_new_error = new_error
                        best_new_val = new_val
                    else:
                        # Revert
                        if tensor == 'U':
                            U[r][c] = old_val
                        elif tensor == 'V':
                            V[r][c] = old_val
                        else:
                            W[r][c] = old_val
                
                # Apply best new value
                if best_new_val != old_val:
                    if tensor == 'U':
                        U[r][c] = best_new_val
                    elif tensor == 'V':
                        V[r][c] = best_new_val
                    else:
                        W[r][c] = best_new_val
                    current_error = min_new_error
                else:
                    # Revert if no improvement (already reverted in loop if not best)
                    pass

            if current_error == 0:
                found = True
                best_decomp = (U, V, W, target_rank)
                best_rank = target_rank
                break
        
        if found:
            break
            
    U, V, W, rank = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
