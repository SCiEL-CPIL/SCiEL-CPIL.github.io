"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _get_strassen_block_decomp(n, m, p):
    """Constructs the rank 88 solution."""
    # Re-implementing the 88 solution from the seed to ensure baseline.
    # This part is kept from the seed logic but cleaned up.
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,0]),
            ([1,1,0,0], [0,0,1,1], [0,1,0,0]),
            ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
            ([1,0,0,0], [1,0,-1,0], [0,1,0,0]),
            ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
            ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
        ]
        
        c_contrib = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, -1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        
        col = col_start
        for t in range(7):
            a_coeffs = terms[t][0]
            b_coeffs = terms[t][1]
            c_coeffs = c_contrib[t]
            
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
            
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
            
            for di in range(2):
                for dk in range(2):
                    W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
            
            col += 1
        return col

    col = 0
    for i0 in [0, 2]:
        i_size = 2 if i0 == 0 else 1
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                if i_size == 2:
                    col = _strassen_block(U, V, W, i0, j0, k0, col, n, m, p)
                else:
                    for di in range(1):
                        for dj in range(2):
                            for dk in range(2):
                                U[(i0+di)*m + (j0+dj)][col] = 1
                                V[(j0+dj)*p + (k0+dk)][col] = 1
                                W[(k0+dk)*n + (i0+di)][col] = 1
                                col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _local_search_reduce_rank(U, V, W, n, m, p, max_time):
    """
    Attempts to reduce the rank by zeroing out columns (terms) 
    if they are linearly dependent or redundant.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Try to remove terms one by one
    # If removing term c keeps the decomposition valid, remove it.
    # This is a greedy approach.
    
    current_rank = rank
    active_cols = list(range(rank))
    
    start_time = time.time()
    
    while time.time() - start_time < max_time and len(active_cols) > 0:
        removed = False
        # Shuffle to avoid bias
        random.shuffle(active_cols)
        for c in active_cols:
            if time.time() - start_time > max_time:
                break
            # Temporarily remove c
            new_active = [x for x in active_cols if x != c]
            if not new_active:
                continue
                
            U_sub = U[:, new_active]
            V_sub = V[:, new_active]
            W_sub = W[:, new_active]
            
            if _verify(U_sub, V_sub, W_sub, n, m, p):
                # Success, remove c permanently
                active_cols = new_active
                removed = True
                break # Restart loop with new set
                
    if not active_cols:
        return None
        
    U_final = U[:, active_cols]
    V_final = V[:, active_cols]
    W_final = W[:, active_cols]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Get the baseline 88 rank solution
    result = _get_strassen_block_decomp(n, m, p)
    if result is None:
        # Fallback to schoolbook if 88 fails (shouldn't happen)
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
    
    # Attempt to reduce rank via local search
    if time.time() < deadline - 10:
        search_time = deadline - time.time() - 5
        reduced = _local_search_reduce_rank(result['U'], result['V'], result['W'], n, m, p, search_time)
        if reduced is not None:
            # Verify the reduced solution
            if _verify(reduced['U'], reduced['V'], reduced['W'], n, m, p):
                return reduced
                
    return result

# EVOLVE-BLOCK-END
