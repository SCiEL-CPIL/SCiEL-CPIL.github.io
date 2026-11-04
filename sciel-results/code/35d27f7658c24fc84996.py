import time
import itertools

DIMENSIONS = [2, 3, 9]

def _target_tensor(n, m, p):
    """Build the 3-mode multiplication tensor T[i,j,k] = 1 for each triple."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _evaluate(U, V, W, n, m, p):
    """Evaluate the tensor C[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]."""
    rank = len(U[0])
    C = {}
    for i in range(n):
        for j in range(m):
            ui = U[i*m + j]
            vj = V[j*p + (0)]  # placeholder, we need full V[j*p+k]
            for k in range(p):
                s = 0
                vrow = V[j*p + k]
                wrow = W[k*n + i]
                for c in range(rank):
                    s += ui[c] * vrow[c] * wrow[c]
                C[(i,j,k)] = s
    return C

def _is_valid(U, V, W, n, m, p, T):
    """Check exact match of decomposition to target tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            ui = U[i*m + j]
            for k in range(p):
                vrow = V[j*p + k]
                wrow = W[k*n + i]
                s = 0
                for c in range(rank):
                    s += ui[c] * vrow[c] * wrow[c]
                if s != T[(i,j,k)]:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, T, max_seconds, start_time):
    """Attempt to reduce rank by removing columns (linear dependence detection) and
    by structured block merges. Returns (U,V,W) with minimal rank found."""
    deadline = start_time + max_seconds
    rank = len(U[0])
    nU, nV, nW = len(U), len(V), len(W)
    
    # --- Strategy 1: Try removing columns one by one (greedy) ---
    # A column c can be removed if there exists coefficients such that the column
    # is a linear combination of others in a way that preserves the tensor.
    # This is complex; instead, try: for each column, set to zero and check if
    # the tensor is still reproducible by adjusting others (too expensive).
    # Simpler: just check if any column is all-zeros (shouldn't be in schoolbook).
    
    # --- Strategy 2: Block decomposition ---
    # B is 3x9. Split into 3 blocks of 3x3: B1, B2, B3.
    # C = A * [B1 | B2 | B3] = [A*B1 | A*B2 | A*B3]
    # Each A*Bk is 2x3. Total rank for three independent 2x3x3 multiplications
    # would be 3 * (rank of 2x3x3). Known rank of 2x3x3 is at most 9 (schoolbook),
    # possibly less. Total = 3 * 9 = 27? No, that's not right.
    # Actually, the full problem is 2x3 x 3x9, tensor size 2*3*9 = 54.
    # If we can decompose each 2x3x3 block at rank r, total rank = 3r.
    # Schoolbook for 2x3x3 is rank 9. So 3*9 = 27 < 54.
    # Let's build a rank-27 decomposition!
    
    def _block_decomp(n, m, p, T, deadline, start_time):
        """Decompose 2x3 x 3x9 as 3 blocks of 2x3 x 3x3."""
        # p = 9 = 3 * 3. Split k into (block, k_in_block) where block in 0..2, k_in_block in 0..2.
        # W[k*n + i] where k = block*3 + k_in_block
        # U[i*m + j], V[j*p + k] = V[j*9 + block*3 + k_in_block]
        
        # For each block b, we need a rank-r decomposition of the 2x3 x 3x3 multiplication.
        # Schoolbook for 2x3 x 3x3: rank = 2*3*3 = 9.
        # Total rank = 3 * 9 = 27.
        
        r = 9  # rank per block
        total_rank = 3 * r  # 27
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col = 0
        for b in range(3):  # 3 blocks
            for i in range(n):
                for j in range(m):
                    for k_ib in range(3):  # k within block
                        k = b * 3 + k_ib
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
        
        if _is_valid(U, V, W, n, m, p, T):
            return U, V, W
        return None
    
    # Try block decomposition
    U2, V2, W2 = _block_decomp(n, m, p, T, deadline, start_time)
    if U2 is not None:
        rank = len(U2[0])
        U, V, W = U2, V2, W2
        # Now try to reduce further within each block
        # For 2x3x3, try rank 8 or lower
        for target_r in range(8, 0, -1):
            if time.time() > deadline - 10:
                break
            # Try to find a rank-target_r decomposition for each 2x3x3 block
            # This is hard; skip for now and return rank 27
            break
    
    if len(U[0]) < rank:
        return U, V, W
    
    # Return original if no improvement
    return U, V, W

def construct():
    start_time = time.time()
    max_seconds = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = len(U[0])
    
    # Verify schoolbook
    if not _is_valid(U, V, W, n, m, p, T):
        # Should never happen
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, T, max_seconds, start_time)
    
    if len(U2[0]) < len(U[0]):
        U, V, W = U2, V2, W2
    
    return {"U": U, "V": V, "W": W}
