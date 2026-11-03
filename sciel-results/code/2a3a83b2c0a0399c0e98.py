"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication.
    """
    factors = []
    factors.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    factors.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    factors.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    factors.append({
        'U': {(1,1): 1},
        'V': {(1,1): 1, (0,1): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    factors.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): -1}
    })
    factors.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    factors.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return factors

def _build_strassen_tiled_4x4_4x6():
    """
    Build a Strassen-tiled decomposition for 4x4 * 4x6.
    Split into 4x4*4x4 (Strassen, rank 49) + 4x4*4x2 (Strassen, rank 28).
    Total rank: 77.
    """
    n, m, p = 4, 4, 6
    rank_strassen = 49
    rank_strassen_2 = 28
    rank_total = rank_strassen + rank_strassen_2

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Part 1: 4x4 * 4x4 (cols 0-3 of B)
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    for i in range(n * m):
        for c in range(rank_strassen):
            U[i][c] = U_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            V[i][c] = V_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            W[i][c] = W_s[i][c]

    # Part 2: 4x4 * 4x2 (cols 4-5 of B)
    U_s2, V_s2, W_s2 = _build_strassen_4x4_4x2()
    col_offset = rank_strassen
    for i in range(n * m):
        for c in range(rank_strassen_2):
            U[i][c + col_offset] = U_s2[i][c]
    for i in range(8):
        for c in range(rank_strassen_2):
            j = i // 2
            k = i % 2
            full_v_idx = j * 6 + (k + 4)
            V[full_v_idx][c + col_offset] = V_s2[i][c]
    for i in range(8):
        for c in range(rank_strassen_2):
            k = i // 4
            out_i = i % 4
            full_w_idx = (k + 4) * 4 + out_i
            W[full_w_idx][c + col_offset] = W_s2[i][c]

    return {"U": U, "V": V, "W": W}

def _build_strassen_4x4_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 * 4x4.
    """
    level1_factors = _strassen_2x2_factors()

    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]

    col = 0
    for l1 in range(7):
        for l2 in range(7):
            l1_factor = level1_factors[l1]
            l2_factor = level1_factors[l2]

            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['U']:
                        continue
                    c1 = l1_factor['U'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['U']:
                                continue
                            c2 = l2_factor['U'][(si, sj)]
                            ai = bi * 2 + si
                            aj = bj * 2 + sj
                            U[ai * 4 + aj][col] += c1 * c2

            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['V']:
                        continue
                    c1 = l1_factor['V'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['V']:
                                continue
                            c2 = l2_factor['V'][(si, sj)]
                            bri = bi * 2 + si
                            bcj = bj * 2 + sj
                            V[bri * 4 + bcj][col] += c1 * c2

            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['W']:
                        continue
                    c1 = l1_factor['W'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['W']:
                                continue
                            c2 = l2_factor['W'][(si, sj)]
                            cri = bi * 2 + si
                            ccj = bj * 2 + sj
                            W[ccj * 4 + cri][col] += c1 * c2

            col += 1

    return U, V, W

def _build_strassen_4x4_4x2():
    """
    Build rank-28 Strassen decomposition for 4x4 * 4x2.
    """
    factors = _strassen_2x2_factors()
    rank = 4 * 7
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]

    col = 0
    for block_a_i in range(2):
        for block_a_j in range(2):
            b_row_offset = block_a_i * 2
            out_row_offset = block_a_i * 2

            for f in range(7):
                factor = factors[f]
                for (ri, ci), val in factor['U'].items():
                    a_row = block_a_i * 2 + ri
                    a_col = block_a_j * 2 + ci
                    U[a_row * 4 + a_col][col] += val

                for (ri, ci), val in factor['V'].items():
                    b_row = b_row_offset + ri
                    b_col = ci
                    V[b_row * 2 + b_col][col] += val

                for (ri, ci), val in factor['W'].items():
                    out_row = out_row_offset + ri
                    out_col = ci
                    W[out_col * 4 + out_row][col] += val

                col += 1

    return U, V, W

def _generate_rank1_candidates(n, m, p):
    """
    Generate candidate rank-1 tensors with sparse integer/half-integer entries.
    A rank-1 term corresponds to:
      U[i*m+j] = a_{ij}, V[j*p+k] = b_{jk}, W[k*n+i] = c_{ki}
    for a single (i,j,k) triple, OR more general "sum-of-products" patterns.
    
    For a true rank-1 tensor in the multiplication tensor space, we need:
    T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    where the multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A
    and (j,k) entry of B contribute to (i,k) of C.
    
    Actually, the multiplication tensor is: T[i,j,k] = delta_{i,k} for the
    standard multiplication. Wait, no. The tensor is:
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor has T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n),
    j in [0,m), k in [0,p).
    
    A rank-1 decomposition term has the form:
    U[i,j] * V[j,k] * W[k,i]
    where U is n x m, V is m x p, W is p x n.
    
    For the standard multiplication tensor, we need:
    sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all (i,j,k)
    
    The schoolbook uses rank n*m*p with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    
    For Strassen-like patterns, we use combinations. Let's generate candidates
    based on small support patterns.
    """
    candidates = []
    
    # Generate all possible "simple" rank-1 patterns with small supports
    # For 2x2x2 Strassen, the patterns have supports of size 1-4 in each factor.
    # Let's generate patterns where each of U, V, W has at most 4 non-zero entries
    # with values in {-1, 0, 0.5, 1, 1.5, 2}
    
    # This is too large. Instead, let's use the known Strassen patterns as seeds
    # and try to compose them globally.
    
    # Alternative: use the 7 Strassen 2x2x2 patterns and the 12 2x2x3 patterns
    # (if they exist) as building blocks, then do a global search.
    
    # For now, let's just return the schoolbook and Strassen-tiled as fallbacks.
    # The real search will be done in _try_global_search.
    
    return candidates

def _try_global_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by searching over combinations
    of Strassen-like patterns applied to different block partitions.
    """
    best = None
    best_rank = n * m * p
    
    # Try different block partition strategies
    # Strategy 1: 2x2 blocks (current approach, rank 77)
    # Strategy 2: 2x3 blocks for the 4x6 part
    # Strategy 3: Mixed partitions
    
    # For a 4x4 * 4x6 multiplication:
    # We can partition A into 2x2 blocks (4 blocks) and B into 2x3 blocks (2 blocks)
    # or 4x3 blocks (1 block) etc.
    
    # Let's try: A as 2x2 blocks, B as 2x3 blocks
    # A: 2x2 blocks -> 4 blocks
    # B: 2x3 blocks -> 2 blocks (rows 0-1, 2-3; cols 0-2, 3-5)
    # But 4x4 * 4x6 doesn't align neatly with 2x3 B blocks.
    
    # Alternative: Use the fact that 4x4*4x6 = (2x2*2x3) repeated
    # A: 2x2 blocks (2x2 grid of blocks)
    # B: 2x3 blocks (2x1 grid of blocks, each 2 rows x 3 cols)
    # Output: 2x3 blocks (2x1 grid of blocks)
    
    # Block (i,j) of A (2x2) * Block (i,0) of B (2x3) -> Block (i,0) of C (2x3)
    # Block (i,j) of A (2x2) * Block (i,1) of B (2x3) -> Block (i,1) of C (2x3)
    
    # So we have 4*2 = 8 block products of type 2x2 * 2x3.
    # If we can find a rank-r decomposition for 2x2 * 2x3, total rank = 8*r.
    # Schoolbook for 2x2*2x3 has rank 12. Strassen-like might do better.
    
    # Let's try to find a low-rank decomposition for 2x2 * 2x3.
    # The 2x2*2x3 multiplication tensor has 2*2*3 = 12 entries.
    # Can we do it in fewer than 12 rank-1 terms?
    
    # For 2x2*2x2, Strassen gives rank 7 < 8.
    # For 2x2*2x3, let's try to extend Strassen ideas.
    
    # Actually, let's just try to use the known best bounds.
    # The rank of 2x2*2x3 is at most 2*2*3 = 12 (schoolbook).
    # With Strassen-like methods, we might get 2*2*3 - 1 = 11 or less.
    
    # For now, let's just return the Strassen-tiled result (rank 77) as it's verified.
    # A true global search would require solving a large linear system.
    
    # Let's try a quick linear algebra approach:
    # Represent the multiplication tensor as a vector in R^(n*m*p)
    # Each rank-1 term is a vector in R^(n*m*p)
    # We want the minimum number of rank-1 vectors that sum to the multiplication tensor.
    
    # This is NP-hard in general, but for small dimensions we can try.
    
    # For 4x4*4x6, the tensor has 96 entries.
    # Let's generate a set of candidate rank-1 vectors and do a greedy set cover.
    
    # Generate candidates: all rank-1 tensors with entries in {-1, 0, 1}
    # and support size <= 4 in each factor.
    
    # This is still too large. Let's just use the Strassen-tiled result.
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p

    # Try Strassen-tiled approach (49 + 28 = 77)
    try:
        strassen = _build_strassen_tiled_4x4_4x6()
        if strassen is not None:
            if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                rank = len(strassen["U"][0])
                if rank < best_rank:
                    best = strassen
                    best_rank = rank
    except Exception:
        pass

    # Try global search (may not find anything better, but worth a shot)
    try:
        if time.time() < deadline - 5:
            best_gs, best_rank_gs = _try_global_search(n, m, p, deadline)
            if best_gs is not None and best_rank_gs < best_rank:
                if _verify_decomposition(best_gs["U"], best_gs["V"], best_gs["W"], n, m, p):
                    best = best_gs
                    best_rank = best_rank_gs
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
