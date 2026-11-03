"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations_with_replacement
from scipy.linalg import solve

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

def _build_strassen_2x2_factors():
    """Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication."""
    factors = []
    factors.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    factors.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}})
    factors.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,1): 1, (1,1): 1}})
    factors.append({'U': {(1,1): 1}, 'V': {(1,1): 1, (0,1): -1}, 'W': {(0,0): 1, (1,0): 1}})
    factors.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): -1}})
    factors.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    factors.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
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
    """Build rank-49 Strassen decomposition for 4x4 * 4x4."""
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
    Correctly handles the rectangular 2x2 * 2x2 block structure.
    """
    factors = _strassen_2x2_factors()
    # 4x4 A split into 4 blocks of 2x2, each multiplied by 4x2 B (split into 2 blocks of 2x2)
    # Actually: A is 4x4, B is 4x2.
    # A blocks: 2x2 grid -> 4 blocks each 2x2
    # B blocks: 2x1 grid -> 2 blocks each 2x2
    # C blocks: 2x1 grid -> 2 blocks each 2x2
    # For each A block row (2 blocks) and each B block (2 blocks), we do 2x2*2x2 with Strassen.
    # Total: 2 (A row blocks) * 2 (B col blocks) * 7 (Strassen rank) = 28
    rank = 28
    U = [[0] * rank for _ in range(16)]  # 4x4 = 16 entries
    V = [[0] * rank for _ in range(8)]   # 4x2 = 8 entries
    W = [[0] * rank for _ in range(8)]   # 4x2 = 8 entries (output is 4x2)

    col = 0
    for a_block_row in range(2):  # A row block: 0 or 1
        for b_block_col in range(2):  # B col block: 0 or 1
            for f in range(7):
                factor = factors[f]
                # U: A block at (a_block_row, *) - but A is 4x4 so block row a_block_row covers rows 2*a_block_row..2*a_block_row+1
                # The A block is 2x2, positioned at rows [2*a_block_row, 2*a_block_row+1], cols [2*b_block_col, 2*b_block_col+1]
                for (ri, ci), val in factor['U'].items():
                    a_row = 2 * a_block_row + ri
                    a_col = 2 * b_block_col + ci
                    U[a_row * 4 + a_col][col] += val

                # V: B block at (a_block_row, b_block_col) - B is 4x2
                # B block row: 2*a_block_row..2*a_block_row+1, col: 2*b_block_col..2*b_block_col+1
                for (ri, ci), val in factor['V'].items():
                    b_row = 2 * a_block_row + ri
                    b_col = 2 * b_block_col + ci
                    V[b_row * 2 + b_col][col] += val

                # W: C block at (a_block_row, b_block_col) - C is 4x2
                # C block row: 2*a_block_row..2*a_block_row+1, col: 2*b_block_col..2*b_block_col+1
                for (ri, ci), val in factor['W'].items():
                    c_row = 2 * a_block_row + ri
                    c_col = 2 * b_block_col + ci
                    W[c_col * 4 + c_row][col] += val

                col += 1

    return U, V, W

def _build_rank1_tensor(n, m, p, u_support, v_support, w_support, u_vals, v_vals, w_vals):
    """
    Build a rank-1 tensor vector from sparse supports.
    u_support: list of (i,j) indices for U (n*m entries)
    v_support: list of (j,k) indices for V (m*p entries)
    w_support: list of (k,i) indices for W (p*n entries)
    Returns a flat vector of length n*m*p representing the rank-1 tensor T[i,j,k] = U[i,j]*V[j,k]*W[k,i]
    """
    vec = np.zeros(n * m * p, dtype=float)
    U_dict = {}
    for idx, (i, j) in enumerate(u_support):
        U_dict[(i, j)] = u_vals[idx]
    V_dict = {}
    for idx, (j, k) in enumerate(v_support):
        V_dict[(j, k)] = v_vals[idx]
    W_dict = {}
    for idx, (k, i) in enumerate(w_support):
        W_dict[(k, i)] = w_vals[idx]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_dict.get((i, j), 0)
                v_val = V_dict.get((j, k), 0)
                w_val = W_dict.get((k, i), 0)
                if u_val != 0 and v_val != 0 and w_val != 0:
                    vec[i * m * p + j * p + k] = u_val * v_val * w_val
    return vec

def _generate_candidate_patterns(n, m, p, rng, num_candidates=20000):
    """
    Generate random sparse rank-1 tensor patterns.
    Each pattern has small support in U, V, W with values from {-1, -0.5, 0.5, 1}.
    """
    candidates = []
    values = [-1.0, -0.5, 0.5, 1.0]

    # Also include the schoolbook unit patterns
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * p, dtype=float)
                vec[i * m * p + j * p + k] = 1.0
                candidates.append(vec)

    # Generate random sparse patterns
    max_support = 4
    for _ in range(num_candidates):
        # Random support sizes
        su = rng.randint(1, max_support + 1)
        sv = rng.randint(1, max_support + 1)
        sw = rng.randint(1, max_support + 1)

        # Random indices
        u_idx = rng.choice(n * m, size=su, replace=False)
        v_idx = rng.choice(m * p, size=sv, replace=False)
        w_idx = rng.choice(p * n, size=sw, replace=False)

        u_support = [(int(x) // m, int(x) % m) for x in u_idx]
        v_support = [(int(x) // p, int(x) % p) for x in v_idx]
        w_support = [(int(x) // n, int(x) % n) for x in w_idx]

        u_vals = [values[rng.randint(0, len(values))] for _ in range(su)]
        v_vals = [values[rng.randint(0, len(values))] for _ in range(sv)]
        w_vals = [values[rng.randint(0, len(values))] for _ in range(sw)]

        vec = _build_rank1_tensor(n, m, p, u_support, v_support, w_support, u_vals, v_vals, w_vals)
        if np.any(vec != 0):
            candidates.append(vec)

    return candidates

def _try_tensor_rank_search(n, m, p, deadline, max_rank_to_try=80):
    """
    Try to find a low-rank decomposition by:
    1. Generating a large pool of random rank-1 tensor candidates
    2. Solving a linear system to find a sparse combination that equals the multiplication tensor
    3. Using iterative reweighted least squares or greedy selection
    """
    rng = np.random.RandomState(42)
    total_dim = n * m * p

    # The multiplication tensor: T[i,j,k] = 1 for all valid (i,j,k)
    target = np.ones(total_dim, dtype=float)

    # Generate candidates
    print(f"Generating {50000} candidate rank-1 tensors...")
    candidates = _generate_candidate_patterns(n, m, p, rng, num_candidates=50000)
    print(f"Generated {len(candidates)} candidates")

    if len(candidates) == 0:
        return None

    # Stack into matrix: each row is a candidate
    C = np.array(candidates, dtype=float)
    n_cands = C.shape[0]

    # Try greedy set-cover-like approach with linear algebra
    # Start with target, iteratively pick the candidate that best reduces the residual
    # subject to integer/half-integer constraints

    # Alternative: solve for coefficients directly
    # We want: sum_r alpha_r * candidate_r = target
    # With alpha_r in {-1, -0.5, 0, 0.5, 1} (or integers)
    # And minimize the number of non-zero alphas

    # Use a greedy approach:
    # 1. Compute inner products of all candidates with target
    # 2. Pick the one with highest |inner product| / ||candidate||^2
    # 3. Subtract, repeat
    # 4. At the end, round coefficients to nearest half-integer and verify

    best_rank = n * m * p
    best_solution = None

    # Greedy with coefficient search
    residual = target.copy()
    chosen_indices = []
    coefficients = []

    # Precompute norms
    norms_sq = np.sum(C * C, axis=1)
    norms_sq[norms_sq < 1e-12] = 1.0

    max_iter = 100
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        if np.linalg.norm(residual) < 1e-10:
            break

        # Compute correlation with residual
        correlations = C @ residual
        # Score: |correlation| / norm_sq
        scores = np.abs(correlations) / norms_sq

        # Pick best
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < 1e-8:
            break

        # Determine coefficient: try to match the projection
        proj_coeff = correlations[best_idx] / norms_sq[best_idx]

        # Round to nearest half-integer
        rounded = np.round(proj_coeff * 2) / 2.0
        if rounded == 0:
            rounded = 1.0 if proj_coeff > 0 else -1.0

        # Check if using this coefficient helps
        new_residual = residual - rounded * C[best_idx]
        if np.linalg.norm(new_residual) < np.linalg.norm(residual) - 1e-10:
            residual = new_residual
            chosen_indices.append(best_idx)
            coefficients.append(rounded)
        else:
            # Try the opposite sign
            rounded = -rounded
            new_residual = residual - rounded * C[best_idx]
            if np.linalg.norm(new_residual) < np.linalg.norm(residual) - 1e-10:
                residual = new_residual
                chosen_indices.append(best_idx)
                coefficients.append(rounded)
            else:
                break

    if np.linalg.norm(residual) < 1e-8 and len(chosen_indices) > 0:
        rank_found = len(chosen_indices)
        if rank_found < best_rank:
            best_rank = rank_found
            best_solution = (chosen_indices, coefficients)

    # Also try a linear algebra approach: pick a subset and solve exactly
    # Use the first K candidates and solve least squares, then round
    for K in [20, 30, 40, 50, 60]:
        if time.time() > deadline - 10:
            break
        if K > n_cands:
            break

        # Select K candidates with highest correlation to target
        target_corrs = C @ target
        target_scores = np.abs(target_corrs) / norms_sq
        top_K = np.argsort(target_scores)[-K:]

        C_sub = C[top_K, :]
        # Solve least squares: C_sub^T * alpha = target
        try:
            alpha, residuals, rank_sv, sv = np.linalg.lstsq(C_sub, target, rcond=None)
        except:
            continue

        # Round alpha to half-integers
        alpha_rounded = np.round(alpha * 2) / 2.0

        # Check if the rounded solution is exact
        reconstructed = C_sub.T @ alpha_rounded
        if np.allclose(reconstructed, target, atol=1e-9):
            nonzero_mask = np.abs(alpha_rounded) > 1e-9
            rank_found = int(np.sum(nonzero_mask))
            if rank_found < best_rank:
                best_rank = rank_found
                best_indices = [top_K[i] for i in range(K) if nonzero_mask[i]]
                best_coeffs = alpha_rounded[nonzero_mask].tolist()
                best_solution = (best_indices, best_coeffs)

    if best_solution is None:
        return None

    chosen_indices, coefficients = best_solution
    rank_found = len(chosen_indices)

    # Reconstruct U, V, W from the chosen candidates
    U = [[0] * rank_found for _ in range(n * m)]
    V = [[0] * rank_found for _ in range(m * p)]
    W = [[0] * rank_found for _ in range(p * n)]

    for col, (cand_idx, coeff) in enumerate(zip(chosen_indices, coefficients)):
        cand_vec = candidates[cand_idx]
        # Reconstruct the factor matrices from the candidate vector
        # The candidate was built from u_support, v_support, w_support
        # We need to recover them. Instead, let's directly decompose:
        # cand_vec[i*m*p + j*p + k] = U[i,j] * V[j,k] * W[k,i]
        # This is hard to invert. Instead, store the candidate as a full tensor and
        # use it directly.

        # Actually, for the output format, we need U, V, W matrices.
        # Let's reconstruct from the tensor vector using a simple factorization.
        # Since our candidates are rank-1 tensors, we can extract factors.
        # But we didn't store the factors. Let's regenerate them.

    # This approach has a problem: we need the actual U, V, W factor matrices.
    # Let's take a different approach: generate candidates WITH their factor structure.

    return None  # Fallback: the above approach doesn't directly give us U,V,W

def _try_direct_linear_solve(n, m, p, deadline):
    """
    Direct approach: represent the problem as finding U, V, W with minimal rank.
    Use the fact that for a rank-r decomposition, we need:
      T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    For small ranks, we can set up the equations and solve.
    """
    total_dim = n * m * p
    target = np.ones(total_dim, dtype=float)

    # Try ranks from low to high
    for R in range(40, 80):
        if time.time() > deadline - 10:
            break

        # We need to find U (n*m x R), V (m*p x R), W (p*n x R)
        # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1

        # This is a system of n*m*p equations in (n*m + m*p + p*n) * R unknowns.
        # For n=4,m=4,p=6: 96 equations, (16+24+24)*R = 64R unknowns.
        # For R=40: 2560 unknowns, 96 equations -> very underdetermined.
        # We need additional constraints (integer/half-integer).

        # Alternative: fix U and V, solve for W (linear in W).
        # Or use a sequential approach.

        # Let's try: fix U to have a specific structure (e.g., from Strassen patterns)
        # and solve for V and W.

        # Actually, let's try a different tactic: use the known Strassen structure
        # but with optimized block assignments.

        break  # Skip this for now, too complex

    return None

def _build_strassen_tiled_v2(n, m, p):
    """
    Improved Strassen tiling with correct index handling.
    For 4x4 * 4x6:
    - Split B into two 4x3 blocks (cols 0-2 and 3-5)
    - For each 4x4 * 4x3 block, use a mixed approach:
      - 4x4 * 4x3 = (4x4 * 4x2) + (4x4 * 4x1)
      - Or: split A into 2x2 blocks, B into 2x3 blocks
    """
    # Let's use the 2x2 * 2x3 approach:
    # A: 2x2 blocks (2x2 grid) -> 4 blocks
    # B: 2x3 blocks (2x1 grid) -> 2 blocks (each 2 rows x 3 cols)
    # C: 2x3 blocks (2x1 grid) -> 2 blocks
    #
    # For each A block (a_i, a_j) and B block (b_i, b_k):
    #   a_i, b_i are row indices (0 or 1), a_j is A col block, b_k is B col block
    #   The product is 2x2 * 2x3 -> 2x3
    #
    # We need a rank-r decomposition for 2x2 * 2x3.
    # Schoolbook: rank 12.
    # Can we do better? Let's try to find one.

    # For 2x2 * 2x3, the tensor has 12 entries.
    # Let's try to find a rank < 12 decomposition.

    # Actually, let's just use the corrected 4x4*4x4 + 4x4*4x2 approach
    # which should give rank 49 + 28 = 77.

    return _build_strassen_tiled_4x4_4x6()

def _find_2x2_2x3_decomposition():
    """
    Try to find a low-rank decomposition for 2x2 * 2x3 matrix multiplication.
    The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    Total: 12 entries.

    A rank-1 term: U[i,j] * V[j,k] * W[k,i]
    """
    n, m, p = 2, 2, 3
    total_dim = n * m * p  # 12
    target = np.ones(total_dim, dtype=float)

    # Generate all possible rank-1 tensors with entries in {-1, 0, 1}
    # U has 4 entries, V has 6 entries, W has 6 entries
    # Each can be 0 or +/-1
    # That's 3^4 * 3^6 * 3^6 = 3^16 = 43 million - too many.

    # Instead, use sparse patterns with at most 2 non-zeros per factor
    candidates = []
    vals = [-1, 0, 1]

    # Generate all U patterns with at most 2 non-zeros
    U_patterns = []
    for nz in range(1, 3):
        for positions in combinations_with_replacement(range(4), nz):
            # Actually, we want all subsets of size nz
            from itertools import combinations
            for pos in combinations(range(4), nz):
                for val_combo in product([v for v in vals if v != 0], repeat=nz):
                    U_pat = [0] * 4
                    for p_idx, v in zip(pos, val_combo):
                        U_pat[p_idx] = v
                    U_patterns.append(tuple(U_pat))

    # Similarly for V (6 entries) and W (6 entries)
    V_patterns = []
    for nz in range(1, 3):
        from itertools import combinations
        for pos in combinations(range(6), nz):
            for val_combo in product([v for v in vals if v != 0], repeat=nz):
                V_pat = [0] * 6
                for p_idx, v in zip(pos, val_combo):
                    V_pat[p_idx] = v
                V_patterns.append(tuple(V_pat))

    W_patterns = []
    for nz in range(1, 3):
        from itertools import combinations
        for pos in combinations(range(6), nz):
            for val_combo in product([v for v in vals if v != 0], repeat=nz):
                W_pat = [0] * 6
                for p_idx, v in zip(pos, val_combo):
                    W_pat[p_idx] = v
                W_patterns.append(tuple(W_pat))

    # Generate all rank-1 tensors
    print(f"  U patterns: {len(U_patterns)}, V patterns: {len(V_patterns)}, W patterns: {len(W_patterns)}")
    total_combos = len(U_patterns) * len(V_patterns) * len(W_patterns)
    print(f"  Total combinations: {total_combos}")

    if total_combos > 5000000:
        # Too many, sample
        rng = np.random.RandomState(42)
        sample_size = 5000000
        for _ in range(sample_size):
            u = U_patterns[rng.randint(len(U_patterns))]
            v = V_patterns[rng.randint(len(V_patterns))]
            w = W_patterns[rng.randint(len(W_patterns))]
            vec = np.zeros(12, dtype=float)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        vec[i * 6 + j * 3 + k] = u[i * 2 + j] * v[j * 3 + k] * w[k * 2 + i]
            if np.any(vec != 0):
                candidates.append(vec)
    else:
        for u in U_patterns:
            for v in V_patterns:
                for w in W_patterns:
                    vec = np.zeros(12, dtype=float)
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                vec[i * 6 + j * 3 + k] = u[i * 2 + j] * v[j * 3 + k] * w[k * 2 + i]
                    if np.any(vec != 0):
                        candidates.append(vec)

    print(f"  Generated {len(candidates)} candidate rank-1 tensors")

    # Now find a combination that sums to target
    # Use greedy approach
    C = np.array(candidates, dtype=float)
    residual = target.copy()
    chosen = []
    coeffs = []

    norms_sq = np.sum(C * C, axis=1)
    norms_sq[norms_sq < 1e-12] = 1.0

    for it in range(20):
        if np.linalg.norm(residual) < 1e-10:
            break
        corrs = C @ residual
        scores = np.abs(corrs) / norms_sq
        best_idx = np.argmax(scores)
        if scores[best_idx] < 1e-8:
            break
        proj = corrs[best_idx] / norms_sq[best_idx]
        rounded = np.round(proj)
        if rounded == 0:
            rounded = 1 if proj > 0 else -1
        new_res = residual - rounded * C[best_idx]
        if np.linalg.norm(new_res) < np.linalg.norm(residual) - 1e-10:
            residual = new_res
            chosen.append(best_idx)
            coeffs.append(rounded)
        else:
            break

    if np.linalg.norm(residual) < 1e-8 and len(chosen) > 0:
        print(f"  Found 2x2*2x3 decomposition with rank {len(chosen)}")
        return chosen, coeffs, candidates

    print(f"  Failed to find 2x2*2x3 decomposition (residual norm: {np.linalg.norm(residual):.6f})")
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p

    # Try Strassen-tiled approach (should give rank 77 if correct)
    try:
        strassen = _build_strassen_tiled_4x4_4x6()
        if strassen is not None:
            if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                rank = len(strassen["U"][0])
                if rank < best_rank:
                    best = strassen
                    best_rank = rank
                    print(f"Strassen-tiled verified: rank {rank}")
            else:
                print("Strassen-tiled FAILED verification")
    except Exception as e:
        print(f"Strassen-tiled exception: {e}")

    # Try to find a 2x2 * 2x3 decomposition and tile it
    if time.time() < deadline - 30:
        try:
            result = _find_2x2_2x3_decomposition()
            if result is not None:
                chosen, coeffs, candidates = result
                r_2x2_2x3 = len(chosen)
                # Tile: 4x4 * 4x6 with 2x2 A blocks and 2x3 B blocks
                # A: 2x2 grid of 2x2 blocks
                # B: 2x1 grid of 2x3 blocks (rows 0-1, 2-3; cols 0-2, 3-5)
                # C: 2x1 grid of 2x3 blocks
                # Total block products: 2 (A row blocks) * 1 (A col block per B col) * 2 (B col blocks) = 4
                # Wait: A is 4x4 -> 2x2 grid of 2x2 blocks
                # B is 4x6 -> 2x2 grid of 2x3 blocks (rows 0-1,2-3; cols 0-2,3-5)
                # C is 4x6 -> 2x2 grid of 2x3 blocks
                # For each (a_row, a_col, b_col) where a_row in {0,1}, a_col in {0,1}, b_col in {0,1}:
                #   A[a_row, a_col] * B[a_row, b_col] -> C[a_row, b_col]
                # That's 2*2*2 = 8 block products, each of type 2x2 * 2x3
                # Total rank: 8 * r_2x2_2x3

                total_rank = 8 * r_2x2_2x3
                print(f"2x2*2x3 rank: {r_2x2_2x3}, tiled rank: {total_rank}")

                if total_rank < best_rank:
                    U = [[0] * total_rank for _ in range(n * m)]
                    V = [[0] * total_rank for _ in range(m * p)]
                    W = [[0] * total_rank for _ in range(p * n)]

                    col = 0
                    for a_row in range(2):
                        for a_col in range(2):
                            for b_col in range(2):
                                for idx, c in zip(chosen, coeffs):
                                    cand = candidates[idx]
                                    # cand is a 12-vector: cand[i*6+j*3+k] for i,j in {0,1}, k in {0,1,2}
                                    for i in range(2):
                                        for j in range(2):
                                            for k in range(3):
                                                val = cand[i * 6 + j * 3 + k]
                                                if val != 0:
                                                    # U index: (a_row*2+i, a_col*2+j) in 4x4 A
                                                    u_idx = (a_row * 2 + i) * 4 + (a_col * 2 + j)
                                                    # V index: (a_row*2+i, b_col*3+k) in 4x6 B
                                                    v_idx = (a_row * 2 + i) * 6 + (b_col * 3 + k)
                                                    # W index: (b_col*3+k, a_row*2+i) in 6x4 C (W[k*n+i] convention)
                                                    w_idx = (b_col * 3 + k) * 4 + (a_row * 2 + i)
                                                    U[u_idx][col] = val
                                                    V[v_idx][col] = val
                                                    W[w_idx][col] = val
                                    col += 1

                    # Verify
                    if _verify_decomposition(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = total_rank
                        print(f"Tiled 2x2*2x3 verified: rank {total_rank}")
                    else:
                        print("Tiled 2x2*2x3 FAILED verification")
        except Exception as e:
            print(f"2x2*2x3 tiling exception: {e}")

    # Final fallback: try a direct random search for rank-1 patterns
    if time.time() < deadline - 15 and best_rank == n * m * p:
        try:
            # Quick random search
            rng = np.random.RandomState(123)
            total_dim = n * m * p
            target = np.ones(total_dim, dtype=float)

            # Try to build a decomposition greedily
            U = [[0] * n * m * p for _ in range(n * m)]
            V = [[0] * n * m * p for _ in range(m * p)]
            W = [[0] * n * m * p for _ in range(p * n)]

            # This is just the schoolbook, already have it.
            pass
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
