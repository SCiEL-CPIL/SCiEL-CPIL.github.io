import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    
    # Target tensor T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1
    
    # Known rank-28 solution
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)
    
    U_known_np = np.array(U_known, dtype=float)
    V_known_np = np.array(V_known, dtype=float)
    W_known_np = np.array(W_known, dtype=float)
    
    def verify(U, V, W, rank):
        """Fast verification using the specific tensor structure"""
        # Check all 1024 positions
        for x in range(8):
            for y in range(16):
                for z in range(8):
                    s = 0.0
                    for r in range(rank):
                        s += U[x, r] * V[y, r] * W[z, r]
                    if abs(s - T[x, y, z]) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Even faster: only check where T is non-zero and a sample of zeros"""
        # Check all non-zero positions
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    x = i * m + k
                    y = k * p + j
                    z = j * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[x, r] * V[y, r] * W[z, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check a sample of zero positions
        rng = random.Random(42)
        for _ in range(200):
            x = rng.randint(0, 7)
            y = rng.randint(0, 15)
            z = rng.randint(0, 7)
            if T[x, y, z] == 0:
                s = 0.0
                for r in range(rank):
                    s += U[x, r] * V[y, r] * W[z, r]
                if abs(s) > 1e-9:
                    return False
        return True
    
    best_rank = rank_known
    best_U = U_known_np.copy()
    best_V = V_known_np.copy()
    best_W = W_known_np.copy()
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Strategy: Try to reduce rank by finding redundant terms
    # For each pair of terms, check if they can be merged
    
    # First, try removing single terms from the known solution
    for target_rank in range(rank_known - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try each possible term removal
        for col_to_remove in range(rank_known):
            if time.time() > deadline:
                break
            
            U_sub = np.delete(U_known_np, col_to_remove, axis=1)
            V_sub = np.delete(V_known_np, col_to_remove, axis=1)
            W_sub = np.delete(W_known_np, col_to_remove, axis=1)
            
            if verify_fast(U_sub, V_sub, W_sub, target_rank):
                best_rank = target_rank
                best_U = U_sub.copy()
                best_V = V_sub.copy()
                best_W = W_sub.copy()
                break
        
        if best_rank == target_rank:
            break
    
    # If we didn't find rank 27 by simple removal, try local search
    if best_rank == rank_known:
        for target_rank in [27, 26]:
            if time.time() > deadline:
                break
            
            # Start from known solution, try to adjust
            U_try = U_known_np.copy()
            V_try = V_known_np.copy()
            W_try = W_known_np.copy()
            
            # Zero out one column and try to compensate
            for col_to_zero in range(rank_known):
                if time.time() > deadline:
                    break
                
                U_mod = U_try.copy()
                V_mod = V_try.copy()
                W_mod = W_try.copy()
                
                U_mod[:, col_to_zero] = 0
                V_mod[:, col_to_zero] = 0
                W_mod[:, col_to_zero] = 0
                
                # Now we have rank_known-1 active terms, but one is zero
                # Try to redistribute its contribution to other terms
                # The contribution of the zeroed term was:
                # U_col * V_col * W_col (outer product)
                # We need to absorb this into the remaining terms
                
                # Try small adjustments to other columns
                found_compensation = False
                for attempt in range(50):
                    if time.time() > deadline:
                        break
                    
                    U_adj = U_mod.copy()
                    V_adj = V_mod.copy()
                    W_adj = W_mod.copy()
                    
                    # Try to add the zeroed term's contribution to a random other term
                    target_col = random.randint(0, rank_known - 1)
                    if target_col == col_to_zero:
                        continue
                    
                    scale = random.choice([0.5, 1.0, -0.5, -1.0])
                    U_adj[:, target_col] += scale * U_known_np[:, col_to_zero]
                    V_adj[:, target_col] += scale * V_known_np[:, col_to_zero]
                    W_adj[:, target_col] += scale * W_known_np[:, col_to_zero]
                    
                    # Remove the zeroed column
                    U_final = np.delete(U_adj, col_to_zero, axis=1)
                    V_final = np.delete(V_adj, col_to_zero, axis=1)
                    W_final = np.delete(W_adj, col_to_zero, axis=1)
                    
                    if verify_fast(U_final, V_final, W_final, target_rank):
                        best_rank = target_rank
                        best_U = U_final.copy()
                        best_V = V_final.copy()
                        best_W = W_final.copy()
                        found_compensation = True
                        break
                
                if found_compensation:
                    break
        
        if best_rank < rank_known:
            break
    
    # Final verification with full check
    if best_rank < rank_known:
        if verify(best_U, best_V, best_W, best_rank):
            pass  # Valid
        else:
            # Fallback to known solution
            best_rank = rank_known
            best_U = U_known_np.copy()
            best_V = V_known_np.copy()
            best_W = W_known_np.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
