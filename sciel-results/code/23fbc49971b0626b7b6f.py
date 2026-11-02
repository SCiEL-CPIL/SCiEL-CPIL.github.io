import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = 120  # 120 seconds
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize U, V, W as numpy arrays for speed
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2 block
    # Format: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, coeff) for A[i,j]
    # V_entries: list of (j, k_local, coeff) for B[j, k_local]
    # W_entries: list of (i, k_local, coeff) for C[i, k_local]
    
    strassen_factors = [
        # M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (a10+a11)b00 -> C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = a00(b01-b11) -> C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = a11(b10-b00) -> C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (a00+a01)b11 -> C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (a10-a00)(b00+b01) -> C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (a01-a11)(b10+b11) -> C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen_factors[s]
            
            # Fill U: row = i*m + j = i*2 + j
            for (i, j, coeff) in u_entries:
                U[i * m + j, col] = coeff
            
            # Fill V: row = j*p + k = j*16 + (2t + k_local)
            for (j, k_local, coeff) in v_entries:
                V[j * p + (2 * t + k_local), col] = coeff
            
            # Fill W: row = k*n + i = (2t + k_local)*2 + i
            for (i, k_local, coeff) in w_entries:
                W[(2 * t + k_local) * n + i, col] = coeff
            
            # Check time occasionally
            if (col % 10 == 0) and (time.time() - start_time > deadline - 10):
                # If we are running out of time, return what we have (which is complete if we finished loop)
                # But we are in a loop, so just break and return
                break
        if (time.time() - start_time > deadline - 10):
            break
    
    # Convert to lists of lists as per standard interface expectation if needed, 
    # but numpy arrays are usually accepted and faster to pass.
    # The prompt says "factor matrices of shapes ...". 
    # I will return lists to be safe with "integer/half-integer entries" phrasing, 
    # though numpy int8 is fine.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
