DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    rank = U_arr.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = int(np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :])))
                if val != 1:
                    return False

    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = int(np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :])))
                    if val != 0:
                        return False
    return True

def _construct_strassen_2x16():
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # B indices for this block
        b00 = 2*t
        b01 = 2*t + 1
        b10 = 16 + 2*t
        b11 = 16 + 2*t + 1
        
        # C indices for this block
        c00 = 4*t
        c01 = 4*t + 1
        c10 = 4*t + 2
        c11 = 4*t + 3
        
        # Strassen intermediates
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 (c00) = M1 + M4 - M5 + M7
        # C01 (c01) = M2 + M4
        # C10 (c10) = M3 + M5
        # C11 (c11) = M1 - M2 + M3 + M6
        
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1; U[0][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[3][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    return U, V, W

def _construct_strassen_2x16_v2():
    """Alternative Strassen arrangement with different block grouping strategy."""
    n, m, p = 2, 2, 16
    # Try different block size: 4 columns at a time, 4 blocks of 4
    num_blocks = p // 4  # 4 blocks
    # Each 2x2 * 2x4 multiplication: 
    # Rank for 2x2 * 2x4 is at most 7*2 = 14 (two 2x2 Strassen products)
    # But potentially less with shared structure
    # Let's just use 8 blocks of 2 for now, same as before but with different assignment
    
    rank = num_blocks * 14  # 4 blocks * 14 = 56 same rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # Each block handles 4 columns: B[2*t:2*t+4] and B[16+2*t:16+2*t+4]
        # Split into two 2x2 sub-blocks
        for sub in range(2):
            # B indices for this sub-block
            b00 = 2*t + sub*2
            b01 = 2*t + sub*2 + 1
            b10 = 16 + 2*t + sub*2
            b11 = 16 + 2*t + sub*2 + 1
            
            # C indices for this sub-block
            c00 = 4*t + sub*2
            c01 = 4*t + sub*2 + 1
            c10 = 4*t + 2 + sub*2
            c11 = 4*t + 2 + sub*2 + 1
            
            # Same 7 Strassen steps as before
            # M1
            U[0][col] += 1; U[3][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1
            
            # M2
            U[2][col] += 1; U[3][col] += 1
            V[b00][col] += 1
            W[c01][col] += 1; W[c11][col] -= 1
            col += 1
            
            # M3
            U[0][col] += 1
            V[b01][col] += 1; V[b11][col] -= 1
            W[c10][col] += 1; W[c11][col] += 1
            col += 1
            
            # M4
            U[3][col] += 1
            V[b10][col] += 1; V[b00][col] -= 1
            W[c00][col] += 1; W[c01][col] += 1
            col += 1
            
            # M5
            U[0][col] += 1; U[1][col] += 1
            V[b11][col] += 1
            W[c00][col] -= 1; W[c10][col] += 1
            col += 1
            
            # M6
            U[2][col] += 1; U[0][col] -= 1
            V[b00][col] += 1; V[b01][col] += 1
            W[c11][col] += 1
            col += 1
            
            # M7
            U[1][col] += 1; U[3][col] -= 1
            V[b10][col] += 1; V[b11][col] += 1
            W[c00][col] += 1
            col += 1
        
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, max_iterations=50):
    """Attempt to reduce rank by identifying linear dependencies among tensor factors."""
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    current_rank = U_arr.shape[1]
    
    # Try to find columns that are linearly dependent
    # This is a heuristic approach - try removing columns and re-verify
    for trial in range(max_iterations):
        # Check time budget
        if trial % 10 == 0:
            pass  # Keep simple for now
        
        # Try removing each column one at a time
        found_removal = False
        for col_to_remove in range(current_rank):
            # Create reduced matrices
            keep_cols = [c for c in range(current_rank) if c != col_to_remove]
            if len(keep_cols) >= 1:
                U_red = U_arr[:, keep_cols]
                V_red = V_arr[:, keep_cols]
                W_red = W_arr[:, keep_cols]
                
                # Quick check: do the remaining columns still span what's needed?
                # Convert to lists for verification
                U_list = U_red.tolist()
                V_list = V_red.tolist()
                W_list = W_red.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        if not found_removal:
            break
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin before 150s kill
    
    # Primary: Strassen rank-56
    U, V, W = _construct_strassen_2x16()
    
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank further
        if time.time() - start_time < DEADLINE - 5:
            U2, V2, W2 = _try_rank_reduction(U, V, W, n, m, p)
            rank2 = len(U2[0]) if U2 else U.shape[1]
            rank1 = U.shape[1]
            if rank2 < rank1 and _verify(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-64 decomposition
    if time.time() - start_time > DEADLINE:
        raise RuntimeError("Deadline exceeded before fallback")
    
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
