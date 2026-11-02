import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    Uses rank-7 core for first 2 columns + rank-3 extension for 3rd column.
    Falls back to rank 11 if verification fails.
    """
    n, m, p = 2, 2, 3
    start_time = time.time()
    deadline = 120.0  # seconds

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1 if j == jp else 0
                        val = 0
                        for r in range(rank):
                            val += U[i*m + j, r] * V[jp*p + k, r] * W[k*n + i, r]
                        if val != target:
                            return False
        return True

    # Strategy: Rank-7 core for columns 0,1 + 3 additional terms for column 2
    # This gives total rank 10.
    
    # First, let's try to construct a valid rank-10 decomposition by:
    # 1. Using a known rank-7 decomposition for 2x2 * 2x2 (first two columns)
    # 2. Adding 3 rank-1 terms for the third column
    
    # Known rank-7 Strassen-like decomposition for 2x2 * 2x2:
    # The 7 terms for C[0,0], C[1,0], C[0,1], C[1,1]
    
    # Let's use a standard rank-7 decomposition for 2x2*2x2
    # U_core: (4, 7), V_core: (4, 7), W_core: (4, 7)
    # W indices for 2x2: 0:C00, 1:C10, 2:C01, 3:C11
    
    # Standard rank-7 decomposition
    U_core = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # A[0,0]
        [0, 1, 0, 1, 0, 1, 1],  # A[1,0]
        [0, 0, 1, 1, 1, 0, 1],  # A[0,1]
        [1, 1, 0, 0, 0, 0, 1]   # A[1,1]
    ], dtype=int)
    
    V_core = np.array([
        [1, 1, 1, 1, 0, 0, 1],  # B[0,0]
        [0, 0, 0, 0, 1, 1, 0],  # B[1,0]
        [0, 0, 0, 0, 0, 0, 0],  # B[0,1] - not used in core for 2x2
        [1, 1, 0, 0, 0, 0, 0]   # B[1,1]
    ], dtype=int)
    
    # Actually, for 2x2 * 2x2, B has shape 2x2, so V should be (4, 7)
    # B indices: 0:B[0,0], 1:B[1,0], 2:B[0,1], 3:B[1,1]
    V_core = np.array([
        [1, 1, 1, 1, 0, 0, 1],  # B[0,0]
        [0, 0, 0, 0, 1, 1, 0],  # B[1,0]
        [0, 0, 0, 0, 0, 0, 1],  # B[0,1]
        [1, 1, 0, 0, 0, 0, 0]   # B[1,1]
    ], dtype=int)
    
    # W for 2x2: indices 0:C00, 1:C10, 2:C01, 3:C11
    W_core = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # C00
        [0, 1, 0, 0, 0, 0, 0],  # C10
        [0, 0, 1, 1, 1, 0, 0],  # C01
        [0, 0, 0, 0, 0, 0, 1]   # C11
    ], dtype=int)
    
    # Verify the core decomposition for 2x2 * 2x2
    def verify_core(U_c, V_c, W_c):
        """Verify 2x2 * 2x2 decomposition."""
        rank = U_c.shape[1]
        for i in range(2):
            for j in range(2):
                for jp in range(2):
                    for k in range(2):
                        target = 1 if j == jp else 0
                        val = 0
                        for r in range(rank):
                            val += U_c[i*2 + j, r] * V_c[jp*2 + k, r] * W_c[k*2 + i, r]
                        if val != target:
                            return False
        return True
    
    if not verify_core(U_core, V_core, W_core):
        # Core is invalid, fall back to rank 11
        return _rank11_fallback()
    
    # Now build the full rank-10 decomposition
    # U_full: (4, 10), V_full: (6, 10), W_full: (6, 10)
    # First 7 columns: from core
    # Last 3 columns: for the third output column
    
    U_full = np.zeros((4, 10), dtype=int)
    V_full = np.zeros((6, 10), dtype=int)
    W_full = np.zeros((6, 10), dtype=int)
    
    # Copy core into first 7 columns
    U_full[:, :7] = U_core
    V_full[:4, :7] = V_core  # B[0,0], B[1,0], B[0,1], B[1,1]
    V_full[4:6, :7] = 0      # B[0,2], B[1,2] not used in core
    W_full[:4, :7] = W_core  # C[0,0], C[1,0], C[0,1], C[1,1]
    W_full[4:6, :7] = 0      # C[0,2], C[1,2] not used in core
    
    # Now we need to add 3 terms (columns 7, 8, 9) that handle the third output column
    # The third column requires: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #                         C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # For each of the 3 new terms r in {7, 8, 9}:
    # We need: sum_{r=7,8,9} U[i,j,r] * V[jp,2,r] * W[2*n+i,r] = 0 for (i,jp) not matching
    # And: sum_{r=7,8,9} U[i,j,r] * V[j,2,r] * W[2*n+i,r] = 1 for matching
    
    # This is a linear system. Let's set up the constraints.
    # Variables: U[0,0,r], U[1,0,r], U[0,1,r], U[1,1,r] for r in {7,8,9} = 12 vars
    #           V[0,2,r], V[1,2,r] for r in {7,8,9} = 6 vars
    #           W[2*0+0,r], W[2*0+1,r], W[2*1+0,r], W[2*1+1,r] for r in {7,8,9} = 12 vars
    # Total: 30 variables
    
    # But we can simplify by noting that the terms must be rank-1.
    # Let's try a direct approach: enumerate possible simple patterns.
    
    # Try: each term handles one product
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    # But we also need A[1,1]*B[1,2] -> C[1,2]
    
    # So we need at least 4 terms for the third column if we do it naively.
    # With 3 terms, we need some sharing.
    
    # Let's try:
    # Term 7: U has A[0,0] and A[1,0], V has B[0,2], W has C[0,2] and C[1,2]
    #   Contribution: A[0,0]*B[0,2]*C[0,2] + A[1,0]*B[0,2]*C[1,2]
    # Term 8: U has A[0,1] and A[1,1], V has B[1,2], W has C[0,2] and C[1,2]
    #   Contribution: A[0,1]*B[1,2]*C[0,2] + A[1,1]*B[1,2]*C[1,2]
    # That's only 2 terms! Let's check if this works.
    
    # Set up term 7 (index 7):
    U_full[0, 7] = 1  # A[0,0]
    U_full[2, 7] = 1  # A[1,0]
    V_full[4, 7] = 1  # B[0,2]
    W_full[4, 7] = 1  # C[0,2]
    W_full[5, 7] = 1  # C[1,2]
    
    # Set up term 8 (index 8):
    U_full[1, 8] = 1  # A[0,1]
    U_full[3, 8] = 1  # A[1,1]
    V_full[5, 8] = 1  # B[1,2]
    W_full[4, 8] = 1  # C[0,2]
    W_full[5, 8] = 1  # C[1,2]
    
    # Term 9 (index 9): leave as zero for now, or use for correction
    
    # Check if this rank-9 decomposition works (with 1 zero term, effectively rank 8)
    if verify(U_full, V_full, W_full):
        # Remove the zero column to get rank 8
        U8 = U_full[:, :8]
        V8 = V_full[:, :8]
        W8 = W_full[:, :8]
        if verify(U8, V8, W8):
            return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
        else:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If the simple 2-term extension doesn't work, try with 3 terms
    # Reset and try different patterns
    
    # Try pattern: 3 terms with more flexibility
    U_full[:, 7:10] = 0
    V_full[:, 7:10] = 0
    W_full[:, 7:10] = 0
    
    # Try various assignments
    patterns = [
        # (U_indices, V_indices, W_indices) for each term
        ([0, 2], [4], [4, 5]),  # Term 7
        ([1, 3], [5], [4, 5]),  # Term 8
        ([0, 1], [4, 5], [4, 5])  # Term 9 - correction
    ]
    
    for attempt in range(50):
        if time.time() - start_time > deadline - 5:
            break
            
        # Try a random pattern
        U_full[:, 7:10] = 0
        V_full[:, 7:10] = 0
        W_full[:, 7:10] = 0
        
        for r in range(3):
            # Choose which A entries to use
            u_mask = np.random.choice([0, 1], size=4)
            u_vals = np.where(u_mask == 1, np.random.choice([-1, 1]), 0)
            U_full[:, 7+r] = u_vals
            
            v_mask = np.random.choice([0, 1], size=2)
            v_vals = np.where(v_mask == 1, np.random.choice([-1, 1]), 0)
            V_full[4:6, 7+r] = v_vals
            
            w_mask = np.random.choice([0, 1], size=4)
            w_vals = np.where(w_mask == 1, np.random.choice([-1, 1]), 0)
            W_full[4:6, 7+r] = w_vals
        
        if verify(U_full, V_full, W_full):
            # Check if we can reduce rank
            for rank in range(8, 11):
                U_r = U_full[:, :rank]
                V_r = V_full[:, :rank]
                W_r = W_full[:, :rank]
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fall back to rank 11
    return _rank11_fallback()

def _rank11_fallback():
    """Return a known valid rank-11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((6, rank), dtype=int)
    W = np.zeros((6, rank), dtype=int)
    
    def set_u(i, j, r, val): U[i*m+j, r] = val
    def set_v(j, k, r, val): V[j*p+k, r] = val
    def set_w(k, i, r, val): W[k*n+i, r] = val

    # Term 1
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)
    
    # Term 2
    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)
    
    # Term 3
    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)
    
    # Term 4
    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)
    
    # Term 5
    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)
    
    # Term 6
    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)
    
    # Term 7
    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)
    
    # Term 8
    set_u(0,0,7,1)
    set_v(0,2,7,1)
    set_w(2,0,7,1)
    
    # Term 9
    set_u(0,1,8,1)
    set_v(1,2,8,1)
    set_w(2,0,8,1)
    
    # Term 10
    set_u(1,0,9,1)
    set_v(0,2,9,1)
    set_w(2,1,9,1)
    
    # Term 11
    set_u(1,1,10,1)
    set_v(1,2,10,1)
    set_w(2,1,10,1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
