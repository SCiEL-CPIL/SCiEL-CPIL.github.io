"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass
    
    # Try to reduce rank by finding linear dependencies
    # Build the tensor and check if it has lower rank
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each rank-1 term r, the contribution to (i,l,k) is U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r]
    # We want to see if we can drop a term and adjust others
    
    # Try removing each term one at a time and see if we can compensate
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Check if any term is redundant by seeing if the tensor without it can be
    # approximated by the remaining terms (exact check)
    for r_drop in range(rank):
        if time.time() > _deadline - 10:
            break
        # Remove term r_drop and see if the remaining terms still satisfy the equations
        # This is a quick check: just verify without that term
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        U_test[:, r_drop] = 0
        V_test[:, r_drop] = 0
        W_test[:, r_drop] = 0
        
        valid = True
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    s = np.dot(U_test[i*m+l, :], V_test[l*p+k, :] * W_test[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Term r_drop is redundant!
            new_rank = rank - 1
            new_U = [row[:] for row in U_arr]
            new_V = [row[:] for row in V_arr]
            new_W = [row[:] for row in W_arr]
            # Remove column r_drop
            new_U = [row[:r_drop] + row[r_drop+1:] for row in new_U]
            new_V = [row[:r_drop] + row[r_drop+1:] for row in new_V]
            new_W = [row[:r_drop] + row[r_drop+1:] for row in new_W]
            
            if new_rank < best_rank:
                best_rank = new_rank
                best_U = new_U
                best_V = new_V
                best_W = new_W
            
            # Update U, V, W to the reduced version and continue checking
            U = best_U
            V = best_V
            W = best_W
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            rank = new_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
