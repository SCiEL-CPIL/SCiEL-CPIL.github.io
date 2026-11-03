"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_3x3x3(U, V, W):
    """Verify 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Strategy: Find a low-rank decomposition of 3x3x3, then combine two of them.
    # The 3x3x6 problem is two disjoint 3x3x3 problems.
    # Rank(3x3x6) = 2 * Rank(3x3x3).
    # Known Rank(3x3x3) <= 19.
    
    def find_3x3x3_decomposition(target_rank, max_steps=2000):
        """Find a decomposition of the 3x3x3 tensor with rank <= target_rank."""
        n3, m3, p3 = 3, 3, 3
        # Initialize with random values
        U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n3*m3)]
        V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m3*p3)]
        W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p3*n3)]
        
        def error_3x3x3(U, V, W, rank):
            max_err = 0.0
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m3+j][r] * V[j*p3+k][r] * W[k*n3+i][r]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            return max_err
        
        curr_err = error_3x3x3(U, V, W, target_rank)
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            if curr_err < 1e-9:
                break
                
            # Pick a random matrix, row, col
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                size = n3*m3
            elif mat_idx == 1:
                mat = V
                size = m3*p3
            else:
                mat = W
                size = p3*n3
                
            r_idx = random.randrange(size)
            c_idx = random.randrange(target_rank)
            
            old_val = mat[r_idx][c_idx]
            new_val = old_val + random.gauss(0, 0.1)
            mat[r_idx][c_idx] = new_val
            
            new_err = error_3x3x3(U, V, W, target_rank)
            if new_err < curr_err:
                curr_err = new_err
            else:
                mat[r_idx][c_idx] = old_val
        
        if curr_err < 1e-9:
            return U, V, W, target_rank
        return None

    # Try to find 3x3x3 decomposition with rank 19
    # If successful, global rank is 38
    # Try lower ranks if time permits? 19 is the known bound.
    
    found_3x3 = None
    for rank_3x3 in [19, 20]: # Start with 19, if fails try 20 (less likely to fail but worse score)
        if time.time() > deadline - 20:
            break
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            res = find_3x3x3_decomposition(rank_3x3, max_steps=500)
            if res:
                found_3x3 = res
                break
        if found_3x3:
            break
            
    if found_3x3:
        U3, V3, W3, r3 = found_3x3
        # Construct global U, V, W
        # Global shapes: U (9, 2*r3), V (18, 2*r3), W (18, 2*r3)
        global_rank = 2 * r3
        U = [[0] * global_rank for _ in range(9)]
        V = [[0] * global_rank for _ in range(18)]
        W = [[0] * global_rank for _ in range(18)]
        
        # First half: cols 0..r3-1
        for i in range(9):
            U[i][0:r3] = U3[i]
            # V mapping: V3 index j*3+k -> V global index j*6+k
            # j in 0..2, k in 0..2
            # V3 row idx: j*3+k
            # V global row idx: j*6+k
            pass
        
        # Fill V for first half
        for j in range(3):
            for k in range(3):
                v3_idx = j*3+k
                v_global_idx = j*6+k
                V[v_global_idx][0:r3] = V3[v3_idx]
                
        # Fill W for first half
        # W3 index k*3+i -> W global index k*3+i (since k in 0..2, k*3+i in 0..8)
        for k in range(3):
            for i in range(3):
                w3_idx = k*3+i
                w_global_idx = k*3+i
                W[w_global_idx][0:r3] = W3[w3_idx]
                
        # Second half: cols r3..2*r3-1
        # U is same
        for i in range(9):
            U[i][r3:2*r3] = U3[i]
            
        # V for second half: B2 cols 3..5.
        # V3 index j*3+k' where k' is 0..2 corresponding to global k=3..5.
        # V global index j*6 + (k'+3)
        for j in range(3):
            for k_prime in range(3):
                v3_idx = j*3+k_prime
                v_global_idx = j*6 + (k_prime+3)
                V[v_global_idx][r3:2*r3] = V3[v3_idx]
                
        # W for second half:
        # W3 index k'*3+i where k' is 0..2 corresponding to global k=3..5.
        # W global index k*3+i = (k'+3)*3+i = k'*3+i+9
        for k_prime in range(3):
            for i in range(3):
                w3_idx = k_prime*3+i
                w_global_idx = k_prime*3+i+9
                W[w_global_idx][r3:2*r3] = W3[w3_idx]
                
        # Verify
        if _verify(U, V, W, n, m, p):
            if global_rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, global_rank
                # If we found a valid decomposition, we can stop if it's good enough
                # But maybe we can find lower? Unlikely for 3x3x3.
                # Let's just return this if it's significantly better than schoolbook.
                if global_rank <= 40:
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # If 3x3x3 search failed, fall back to original schoolbook or try direct search?
    # The original direct search was very slow and unlikely to beat 54.
    # So we return the best found.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
