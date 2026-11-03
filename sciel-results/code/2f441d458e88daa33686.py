import time
import random

DIMENSIONS = [4, 4, 7]

def _verify_444(U, V, W, r):
    n, m, p = 4, 4, 4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _find_444_decomp(max_rank=30, trials=5000):
    n, m, p = 4, 4, 4
    rng = random.Random(42)
    for r in range(23, max_rank + 1):
        for t in range(trials):
            U = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(16)]
            V = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(16)]
            W = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(16)]
            if _verify_444(U, V, W, r):
                return U, V, W, r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try to find a low rank decomposition for 4x4x4
    decomp_444 = None
    if time.time() < deadline - 60:
        decomp_444 = _find_444_decomp(max_rank=30, trials=10000)
        
    if decomp_444:
        U1, V1, W1, r1 = decomp_444
        # Construct 4x4x7 decomposition
        # B is 4x7. Split into B1 (4x4) and B2 (4x3).
        # C = A B1 + A B2.
        # Use r1 terms for A B1.
        # Use schoolbook for A B2. Rank = 4*4*3 = 48.
        total_rank = r1 + 48
        U = [0] * total_rank
        V = [0] * total_rank
        W = [0] * total_rank
        
        # Copy 4x4x4 part
        for c in range(r1):
            U[c] = U1[c]
            V[c] = V1[c]
            W[c] = W1[c]
            
        # Schoolbook for 4x4x3 part
        # A is 4x4, B2 is 4x3.
        # Indices for B2: j in 0..3, k in 0..2 (relative to B2).
        # In the full 4x4x7 tensor, B is 4x7.
        # B1 corresponds to k=0..3. B2 corresponds to k=4..6.
        # The U, V, W matrices need to be aligned with the full indices.
        # U is (n*m, rank) = (16, rank).
        # V is (m*p, rank) = (28, rank).
        # W is (p*n, rank) = (28, rank).
        
        # The 4x4x4 decomposition uses V indices for k=0..3.
        # We need to shift V indices for the second part?
        # No, we are constructing a new decomposition for 4x4x7.
        # The first r1 terms compute A * B1.
        # B1 is the first 4 columns of B.
        # So V for these terms should be non-zero only for j*p + k where k in 0..3.
        # The found U1, V1, W1 are for 4x4x4, so V1 has size 16 (4*4).
        # We need to embed V1 into the first 16 rows of V (which has 28 rows).
        # Rows 0..15 of V correspond to k=0..3.
        # Rows 16..27 of V correspond to k=4..6.
        
        # For the first r1 terms:
        # U[c] is already correct (size 16).
        # V[c] should be V1[c] padded with zeros for k=4..6.
        # W[c] is already correct (size 16, for i,k in 0..3).
        # But W for 4x4x7 has size 28 (p*n = 7*4).
        # W indices are k*n + i.
        # For the first part, k is 0..3. So W indices 0..15 are used.
        # We need to pad W[c] with zeros for k=4..6 (indices 16..27).
        
        for c in range(r1):
            V_full = [0] * 28
            for idx in range(16):
                V_full[idx] = V1[c][idx]
            V[c] = V_full
            
            W_full = [0] * 28
            for idx in range(16):
                W_full[idx] = W1[c][idx]
            W[c] = W_full
            
            U[c] = U1[c]
            
        # Now add schoolbook terms for A * B2.
        # B2 is columns 4, 5, 6 of B.
        # j in 0..3, k_rel in 0..2.
        # k_abs = k_rel + 4.
        col = r1
        for i in range(4):
            for j in range(4):
                for k_rel in range(3):
                    k_abs = k_rel + 4
                    U[col] = [0] * total_rank
                    U[col][i * 4 + j] = 1
                    V[col] = [0] * total_rank
                    V[col][j * 7 + k_abs] = 1
                    W[col] = [0] * total_rank
                    W[col][k_abs * 4 + i] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook for 4x4x7
    rank = 4 * 4 * 7
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
