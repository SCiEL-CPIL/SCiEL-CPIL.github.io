import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 decomposition for 2x2 x 2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1

    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W


def _build_4x4x5_from_strassen():
    """Build 4x4x5 decomposition: Strassen 4x4x4 core + boundary terms."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 16
    rank_total = rank_core + rank_boundary

    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    for c in range(rank_core):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] = U44[i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_iter=200, tol=1e-9):
    """Attempt to reduce rank by removing provably-redundant rank-1 terms.
    Uses both individual term removal and pair-based linear combination checks."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])

    for iteration in range(max_iter):
        removed = False
        
        # Phase 1: Try removing individual terms
        for c in range(rank - 1, -1, -1):
            U_test = [row[:c] + row[c + 1:] for row in U]
            V_test = [row[:c] + row[c + 1:] for row in V]
            W_test = [row[:c] + row[c + 1:] for row in W]
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                removed = True
                break
        
        if removed:
            continue
        
        # Phase 2: Try merging pairs (c1, c2) into one term
        # If U[:,c1] = a*U[:,c2] + U_new and similar for V,W with W_new = W1/W2 scaling
        # More practically: check if removing c1 and adjusting c2 by a factor works
        # This is expensive, so only try a subset
        if rank > 1 and iteration % 3 == 0:
            merged = False
            for c1 in range(rank - 1, -1, -1):
                if merged:
                    break
                for c2 in range(rank - 1, c1, -1):
                    # Try: can we absorb term c1 into term c2?
                    # Check if there exists alpha such that
                    # U[:,c1] = alpha * U[:,c2], V[:,c1] = alpha * V[:,c2], W[:,c1] = alpha * W[:,c2]
                    # If so, we can just remove c1 (since it's a scalar multiple already captured)
                    # More generally: try small rational alphas
                    for alpha_num in range(-3, 4):
                        if alpha_num == 0:
                            continue
                        # Check if U[:,c1] ≈ alpha * U[:,c2]
                        match = True
                        for i in range(n * m):
                            if abs(U[i][c1] - alpha_num * U[i][c2]) > tol * 10:
                                match = False
                                break
                        if not match:
                            continue
                        for j in range(m * p):
                            if abs(V[j][c1] - alpha_num * V[j][c2]) > tol * 10:
                                match = False
                                break
                        if not match:
                            continue
                        for k in range(p * n):
                            if abs(W[k][c1] - alpha_num * W[k][c2]) > tol * 10:
                                match = False
                                break
                        if not match:
                            continue
                        # Exact scalar multiple: remove c1
                        U_test = [row[:c1] + row[c1 + 1:] for row in U]
                        V_test = [row[:c1] + row[c1 + 1:] for row in V]
                        W_test = [row[:c1] + row[c1 + 1:] for row in W]
                        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                            U = U_test
                            V = V_test
                            W = W_test
                            rank -= 1
                            removed = True
                            merged = True
                            break
            if merged:
                continue

        if not removed:
            break

    if rank < len(U[0]) or rank < len(V[0]) or rank < len(W[0]):
        return U, V, W, rank
    return None


def _try_merge_terms(U, V, W, n, m, p, tol=1e-9):
    """Try to find pairs of terms that can be merged (sum/diff gives valid decomposition).
    This is a limited search for algebraic simplifications."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    best_U, best_V, best_W, best_rank = U, V, W, rank

    # Try removing each term and checking
    for c in range(rank - 1, -1, -1):
        U_test = [row[:c] + row[c + 1:] for row in U]
        V_test = [row[:c] + row[c + 1:] for row in V]
        W_test = [row[:c] + row[c + 1:] for row in W]
        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
            U, V, W = U_test, V_test, W_test
            rank -= 1
            best_U, best_V, best_W, best_rank = U, V, W, rank

    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Build structured Strassen-based 4x4x5 decomposition
    if time.time() < deadline:
        U_new, V_new, W_new = _build_4x4x5_from_strassen()
        if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
            rank_total = len(U_new[0])
            best_rank = rank_total
            best_result = {"U": U_new, "V": V_new, "W": W_new}

            # Attempt rank reduction
            if time.time() < deadline:
                reduced = _try_reduce_rank(U_new, V_new, W_new, n, m, p, max_iter=300)
                if reduced is not None:
                    U_r, V_r, W_r, rank_r = reduced
                    if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                        best_rank = rank_r
                        best_result = {"U": U_r, "V": V_r, "W": W_r}

    # Try to find lower-rank solutions via structured search
    if time.time() < deadline and best_rank > 30:
        # Strategy: Try various rank values and search for valid decompositions
        # using a structured parameterization inspired by Strassen patterns.
        
        # Approach 1: Try to find a decomposition where the boundary terms
        # can be absorbed into the Strassen core terms.
        # The key idea: for the 5th column, C[:,4] = A @ B[:,4]
        # We need sum_c U[i,j,c] * V[j,4,c] * W[4,i,c] = 1 for all i,j
        # while also maintaining sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for k=0..3
        
        # Let's try a direct linear algebra approach:
        # Fix U and V from the Strassen core, solve for W.
        if time.time() < deadline:
            U_core, V_core, W_core = _build_4x4x4_strassen()
            rank_core = 28
            
            # For the 5th column, we need additional rank-1 terms
            # Try different numbers of additional terms
            for extra_rank in range(0, 20):
                if time.time() > deadline:
                    break
                
                total_rank = rank_core + extra_rank
                U_try = [row[:] for row in U_core] + [[0] * rank_core for _ in range(16)]
                V_try = [row[:] for row in V_core] + [[0] * rank_core for _ in range(20)]
                W_try = [row[:] for row in W_core] + [[0] * rank_core for _ in range(20)]
                
                # Extend to full size
                U_full = [[0] * total_rank for _ in range(16)]
                V_full = [[0] * total_rank for _ in range(20)]
                W_full = [[0] * total_rank for _ in range(20)]
                
                for c in range(rank_core):
                    for i in range(16):
                        U_full[i][c] = U_core[i][c]
                    for j in range(16):
                        V_full[j][c] = V_core[j][c]
                    for k in range(16):
                        W_full[k][c] = W_core[k][c]
                
                # For the extra terms, we need to fill in the 5th column
                # The constraint is: for each (i,j), sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
                # And for k=0..3: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                
                # This is complex; skip for now and try simpler approaches
                pass

    # WFTT-inspired structured search: embed into cyclic convolution over Z_11
    # For 4x4 x 4x5, we can use the fact that 4*4=16 and 4*5=20
    # Try to construct a decomposition using roots of unity in Z_11
    if time.time() < deadline and best_rank > 30:
        # Strategy: Use the 5x5 cyclic group structure
        # The 5th prime Z_11 has primitive 5th roots: 3^1=3, 3^2=9, 3^3=5, 3^4=4 (mod 11)
        # Actually in Z_11, the multiplicative group has order 10, so 5th roots are:
        # x^5 = 1 mod 11: x = 1, and solutions of x^5-1=0
        # 3^5 = 243 mod 11 = 243 - 22*11 = 243-242 = 1. So 3 is a primitive 5th root.
        # Roots: 1, 3, 9, 5, 4 (mod 11)
        
        # WFTT approach: 
        # For matrix mult (n,m,p), if we can find a ring R where n,m,p divide phi(R)
        # or use the convolution embedding
        # 
        # For 4x4x5: try to use the structure where we pad to 5x5x5
        # and use DFT over Z_11 (which has 5th roots of unity)
        #
        # The key insight: if we can write the 4x4x5 tensor as a sum of rank-1
        # terms derived from the DFT diagonalization, we might get rank < 28+16=44
        #
        # Let's try a more direct approach: solve the system of equations
        # for a given target rank using the structure of the problem
        
        import random
        rng = random.Random(123)
        
        # Try structured patterns based on Strassen-like block decomposition
        # but with the 5th column handled differently
        for target_rank in range(28, min(best_rank, 40)):
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            # Use a structured approach: 
            # Start from the 4x4x4 Strassen core (rank 28) and try to
            # handle the 5th column with fewer than 16 additional terms
            # by modifying the core terms
            
            # For each (i,j) pair, we need:
            # sum_c U[i,j,c] * V[j,4,c] * W[4,i,c] = 1
            # 
            # If we add r extra terms, we need to solve for their values
            # such that all 16 equations (i,j) are satisfied
            # while not breaking the existing 4x4x4 equations
            
            extra_rank = target_rank - 28
            if extra_rank < 1:
                continue
            
            # Try to find extra terms that satisfy the 5th column constraint
            # without disturbing the first 4 columns
            # This means: for k=0..3, sum_extra U[i,j,c]*V[j,k,c]*W[k,i,c] = 0
            # and for the 5th col: sum_extra U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
            
            # Simple approach: use delta terms where only specific (i,j) pairs are affected
            # Each extra term can handle at most one (i,j) pair cleanly
            # So we need at least ceil(16/1) = 16 terms if they're independent
            # But we can potentially do better with overlapping terms
            
            # Try: use the Strassen structure for the 5th column too
            # If we treat the 5th column as another "row" in a 5x5 problem
            # and use 5x5 Strassen-like decomposition
            
            # For now, try a greedy approach:
            # Start with the 28-term core, add terms one by one for the 5th column
            U_core, V_core, W_core = _build_4x4x4_strassen()
            
            U_try = [row[:] + [0] * extra_rank for row in U_core]
            V_try = [row[:] + [0] * extra_rank for row in V_core]
            # Extend V to have 20 rows (5 cols)
            V_full = []
            for j in range(4):
                for k in range(5):
                    if k < 4:
                        V_full.append(V_core[j * 4 + k] + [0] * extra_rank)
                    else:
                        V_full.append([0] * 28 + [0] * extra_rank)
            
            # Extend W to have 20 rows
            W_full = []
            for k in range(5):
                for i in range(4):
                    if k < 4:
                        W_full.append(W_core[k * 4 + i] + [0] * extra_rank)
                    else:
                        W_full.append([0] * 28 + [0] * extra_rank)
            
            # Now fill in the extra terms for the 5th column
            # We need: for each (i,j), sum_{c=28}^{28+extra_rank-1} U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
            # And: for each (i,j,k=0..3), sum_extra U[i,j,c]*V[j,k,c]*W[k,i,c] = 0
            
            # Simplest: use diagonal terms where each extra term handles one (i,j)
            # U[i,j,c] = 1, V[j,4,c] = 1, W[4,i,c] = 1 for term c corresponding to (i,j)
            # But this requires 16 terms and doesn't interfere with other columns
            # since V[j,k,c] = 0 for k!=4 and W[k,i,c] = 0 for k!=4
            
            # Let's try to do better: use fewer terms with clever overlap
            # Try random sparse patterns
            for attempt in range(min(20, int((deadline - time.time()) / 0.5))):
                if time.time() > deadline:
                    break
                
                U_rand = [row[:] for row in U_try]
                V_rand = [row[:] for row in V_full]
                W_rand = [row[:] for row in W_full]
                
                # Randomly assign values to extra terms
                for c in range(28, 28 + extra_rank):
                    for idx in range(16):
                        if rng.random() < 0.3:
                            U_rand[idx][c] = rng.choice([-1, 1])
                    for idx in range(20):
                        if rng.random() < 0.3:
                            V_rand[idx][c] = rng.choice([-1, 1])
                        if rng.random() < 0.3:
                            W_rand[idx][c] = rng.choice([-1, 1])
                
                if _verify_decomposition_fast(U_rand, V_rand, W_rand, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                    break

    # Final: try to reduce the best result further with more aggressive search
    if time.time() < deadline:
        U_best, V_best, W_best = best_result["U"], best_result["V"], best_result["W"]
        reduced = _try_reduce_rank(U_best, V_best, W_best, n, m, p, max_iter=200)
        if reduced is not None:
            U_r, V_r, W_r, rank_r = reduced
            if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                best_rank = rank_r
                best_result = {"U": U_r, "V": V_r, "W": W_r}
        
        # Try pair-merging: check if two terms can be combined
        if time.time() < deadline - 10:
            U_f = [row[:] for row in best_result["U"]]
            V_f = [row[:] for row in best_result["V"]]
            W_f = [row[:] for row in best_result["W"]]
            r = len(U_f[0])
            
            # Try removing each pair and checking
            for c1 in range(r - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                for c2 in range(c1 - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    # Remove both c1 and c2, check if remaining terms still work
                    # (This is very conservative - only works if both are redundant)
                    idxs = [i for i in range(r) if i != c1 and i != c2]
                    U_test = [[U_f[i][j] for j in idxs] for i in range(16)]
                    V_test = [[V_f[i][j] for j in idxs] for i in range(20)]
                    W_test = [[W_f[i][j] for j in idxs] for i in range(20)]
                    if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                        best_rank = r - 2
                        best_result = {"U": U_test, "V": V_test, "W": W_test}
                        break

    return best_result
