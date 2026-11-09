"""Find rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    n_m = n * m  # 10
    m_p = m * p  # 25
    p_n = p * n  # 10

    # Precompute target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((n_m, m_p, p_n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W):
        """Vectorized verification of tensor decomposition."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        # Reconstruct: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for speed
        recon = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        return np.allclose(recon, target, atol=1e-9)

    # --- Strassen-based construction (rank 46) ---
    def strassen_2x2x2(U, V, W, rank_offset, i0, j0, k0):
        """Add 7 Strassen terms for 2x2 x 2x2 -> 2x2 block at (i0,j0,k0)."""
        c = rank_offset
        U[i0*m + j0][c] = 1
        U[(i0+1)*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + (i0+1)][c] = 1

        c = rank_offset + 1
        U[(i0+1)*m + j0][c] = 1
        U[(i0+1)*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        W[k0*n + (i0+1)][c] = 1

        c = rank_offset + 2
        U[i0*m + j0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        V[(j0+1)*p + (k0+1)][c] = -1
        W[(k0+1)*n + i0][c] = 1

        c = rank_offset + 3
        U[(i0+1)*m + (j0+1)][c] = 1
        V[(j0+1)*p + j0][c] = 1
        V[j0*p + j0][c] = -1
        W[j0*n + (i0+1)][c] = 1

        c = rank_offset + 4
        U[i0*m + j0][c] = 1
        U[i0*m + (j0+1)][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[(k0+1)*n + i0][c] = -1

        c = rank_offset + 5
        U[(i0+1)*m + j0][c] = 1
        U[i0*m + j0][c] = -1
        V[j0*p + k0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        W[k0*n + (i0+1)][c] = 1
        W[(k0+1)*n + (i0+1)][c] = 1

        c = rank_offset + 6
        U[i0*m + (j0+1)][c] = 1
        U[(i0+1)*m + (j0+1)][c] = -1
        V[(j0+1)*p + j0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[j0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1

        return rank_offset + 7

    def schoolbook_2x2x1(U, V, W, rank_offset, i0, j0, k0):
        """Add 4 schoolbook terms for 2x2 x 2x1 -> 2x1 block at (i0,j0,k0)."""
        for di in range(2):
            for dj in range(2):
                c = rank_offset + di*2 + dj
                U[(i0+di)*m + (j0+dj)][c] = 1
                V[(j0+dj)*p + k0][c] = 1
                W[k0*n + (i0+di)][c] = 1
        return rank_offset + 4

    def schoolbook_2x1x2(U, V, W, rank_offset, i0, j0, k0):
        """Add 4 schoolbook terms for 2x1 x 1x2 -> 2x2 block at (i0,j0,k0)."""
        for di in range(2):
            for dk in range(2):
                c = rank_offset + di*2 + dk
                U[(i0+di)*m + j0][c] = 1
                V[j0*p + (k0+dk)][c] = 1
                W[(k0+dk)*n + (i0+di)][c] = 1
        return rank_offset + 4

    def schoolbook_2x1x1(U, V, W, rank_offset, i0, j0, k0):
        """Add 2 schoolbook terms for 2x1 x 1x1 -> 2x1 block at (i0,j0,k0)."""
        for di in range(2):
            c = rank_offset + di
            U[(i0+di)*m + j0][c] = 1
            V[j0*p + k0][c] = 1
            W[k0*n + (i0+di)][c] = 1
        return rank_offset + 2

    def build_strassen():
        """Build the Strassen-based decomposition of rank 46."""
        total_rank = 46
        U = [[0] * total_rank for _ in range(n_m)]
        V = [[0] * total_rank for _ in range(m_p)]
        W = [[0] * total_rank for _ in range(p_n)]
        rank_offset = 0

        # C1 = A1*B11 + A2*B21 + A3*B31
        rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 0, 0)
        rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 2, 0)
        rank_offset = schoolbook_2x1x2(U, V, W, rank_offset, 0, 4, 0)

        # C2 = A1*B12 + A2*B22 + A3*B32
        rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 0, 2)
        rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 2, 2)
        rank_offset = schoolbook_2x1x2(U, V, W, rank_offset, 0, 4, 2)

        # C3 = A1*B13 + A2*B23 + A3*B33
        rank_offset = schoolbook_2x2x1(U, V, W, rank_offset, 0, 0, 4)
        rank_offset = schoolbook_2x2x1(U, V, W, rank_offset, 0, 2, 4)
        rank_offset = schoolbook_2x1x1(U, V, W, rank_offset, 0, 4, 4)

        return U, V, W

    def build_schoolbook():
        """Build the schoolbook decomposition of rank 50."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Start with Strassen construction
    U, V, W = build_strassen()
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}

    # --- Search for lower rank decompositions ---
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actually for matrix multiplication, rank >= n*m*p / max(n,m,p) is not tight.
    # Known: rank(M_{n,m,p}) >= n*m + m*p + p*n - n - m - p + 1 (for some cases)
    # For 2x5x5: lower bound is at least 25 (from max dimension product)
    # Let's search from best_rank-1 down to 25
    
    def random_decomposition(rank, seed_val, sparsity=0.3):
        """Generate a random sparse decomposition."""
        rng = random.Random(seed_val)
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        
        for r in range(rank):
            # Random sparse pattern with integer/half-integer values
            for a in range(n_m):
                if rng.random() < sparsity:
                    U[a][r] = rng.choice([-1, -0.5, 0.5, 1])
            for b in range(m_p):
                if rng.random() < sparsity:
                    V[b][r] = rng.choice([-1, -0.5, 0.5, 1])
            for c in range(p_n):
                if rng.random() < sparsity:
                    W[c][r] = rng.choice([-1, -0.5, 0.5, 1])
        
        return U, V, W

    def structured_search(rank, seed_val):
        """Try structured patterns: each rank term covers a small block."""
        rng = random.Random(seed_val)
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        
        # Each term: pick a small subset of indices
        for r in range(rank):
            # Pick 1-3 indices for U, 1-3 for V, 1-3 for W
            nu = rng.randint(1, 3)
            nv = rng.randint(1, 3)
            nw = rng.randint(1, 3)
            
            us = rng.sample(range(n_m), nu)
            vs = rng.sample(range(m_p), nv)
            ws = rng.sample(range(p_n), nw)
            
            for a in us:
                U[a][r] = rng.choice([-1, 1])
            for b in vs:
                V[b][r] = rng.choice([-1, 1])
            for c in ws:
                W[c][r] = rng.choice([-1, 1])
        
        return U, V, W

    # Search for lower rank
    for candidate_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline:
            break
        
        # Try different strategies
        num_attempts = 200
        found = False
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Alternate between random and structured
            if attempt % 3 == 0:
                U_c, V_c, W_c = random_decomposition(candidate_rank, seed_val=candidate_rank * 10000 + attempt, sparsity=0.2)
            elif attempt % 3 == 1:
                U_c, V_c, W_c = structured_search(candidate_rank, seed_val=candidate_rank * 10000 + attempt)
            else:
                U_c, V_c, W_c = random_decomposition(candidate_rank, seed_val=candidate_rank * 10000 + attempt + 9999, sparsity=0.4)
            
            if verify(U_c, V_c, W_c):
                best_rank = candidate_rank
                best_result = {"U": U_c, "V": V_c, "W": W_c}
                found = True
                break
        
        if found:
            continue  # Keep searching for even lower rank
    
    return best_result
