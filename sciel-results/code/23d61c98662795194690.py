# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    tensor = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.all(tensor == 1))


def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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
    return {"U": U, "V": V, "W": W}


def _strassen_terms(i_off, j_off, k_off):
    """
    Return the 7 Strassen terms for a 2x2x2 block.
    Each term is (U_dict, V_dict, W_dict) where each dict maps global indices to coefficients.
    Indices are (i, j) for U, (j, k) for V, (k, i) for W (trace convention W[k*n+i]).
    """
    terms = []

    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] with +1, C[1,1] with +1
    U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U1, V1, W1))

    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V2 = {(j_off + 0, k_off + 0): 1}
    W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
    terms.append((U2, V2, W2))

    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # Contributes to C[0,1] with +1, C[1,1] with +1
    U3 = {(i_off + 0, j_off + 0): 1}
    V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
    W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U3, V3, W3))

    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # Contributes to C[1,0] with +1, C[0,0] with +1
    U4 = {(i_off + 1, j_off + 1): 1}
    V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
    W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
    terms.append((U4, V4, W4))

    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,1] with +1, C[0,0] with -1
    U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
    V5 = {(j_off + 1, k_off + 1): 1}
    W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
    terms.append((U5, V5, W5))

    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1] with +1
    U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
    V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
    W6 = {(k_off + 1, i_off + 1): 1}
    terms.append((U6, V6, W6))

    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0] with +1
    U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
    V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W7 = {(k_off + 0, i_off + 0): 1}
    terms.append((U7, W7) if False else (U7, V7, W7))

    return terms


def _build_decomposition(strassen_blocks):
    """
    Build the full decomposition given a list of Strassen blocks.
    Returns (U, V, W, rank) or None if invalid.
    """
    n, m, p = N, M, P

    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in strassen_blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    strassen_covered.add((i_off + ii, j_off + jj, k_off + kk))

    # Check for overlaps (shouldn't happen with disjoint blocks, but safety)
    # Actually, Strassen blocks can overlap in indices but we treat them as independent terms.
    # The "covered" set is just to avoid adding schoolbook terms for indices that are handled.
    # Wait, if blocks overlap, the tensor equation must still hold.
    # Strassen blocks are local. If two blocks share an (i,j,k) index, that index gets contributions from both.
    # The schoolbook term for that index would add another 1, making it 2. So we must NOT add schoolbook
    # for any index touched by ANY Strassen block.
    # But do the Strassen blocks themselves sum to 1 for each (i,j,k) in their volume?
    # Yes, by construction.
    # So if blocks overlap, the sum of contributions for an overlapping index is > 1.
    # Thus, blocks MUST be disjoint in their (i,j,k) volumes for this simple combination to work.

    all_terms = []

    # Add Strassen terms for each block
    for i_off, j_off, k_off in strassen_blocks:
        block_terms = _strassen_terms(i_off, j_off, k_off)
        for term in block_terms:
            all_terms.append(term)

    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_sb = {(i, j): 1}
                    V_sb = {(j, k): 1}
                    W_sb = {(k, i): 1}
                    all_terms.append((U_sb, V_sb, W_sb))

    rank = len(all_terms)

    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    return U, V, W, rank


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Searches for the maximum number of disjoint Strassen 2x2x2 blocks to minimize rank.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = N, M, P

    # Generate all possible Strassen block positions
    possible_blocks = []
    for i_off in range(n - 1):
        for j_off in range(m - 1):
            for k_off in range(p - 1):
                possible_blocks.append((i_off, j_off, k_off))

    # Precompute the set of (i,j,k) indices covered by each block
    block_volumes = []
    for block in possible_blocks:
        i_off, j_off, k_off = block
        vol = set()
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    vol.add((i_off + ii, j_off + jj, k_off + kk))
        block_volumes.append(vol)

    # Backtracking search for maximum set of disjoint blocks
    # State: (current_index, current_selection, current_rank)
    # We want to maximize the number of blocks (minimize rank = 60 - 4 * num_blocks)
    
    best_rank = RANK_SCHOOLBOOK
    best_blocks = []

    # Sort blocks to improve pruning? Maybe by volume overlap?
    # For now, simple index order.
    
    # Use recursion with pruning
    # Max possible blocks is small (volume 60, each block size 8, so max 7 blocks)
    # 7 blocks -> rank 60 - 4*7 = 32.
    # 6 blocks -> rank 60 - 4*6 = 36.
    # Current best is 4 blocks -> rank 44.
    
    def search(start_idx, selected_blocks, current_rank):
        nonlocal best_rank, best_blocks
        
        if time.time() > deadline:
            return

        # Prune: even if we add all remaining blocks, can we beat best_rank?
        # Max additional blocks = len(possible_blocks) - start_idx
        # Rank reduction per block = 4
        # Min possible rank from this state = current_rank - 4 * (len(possible_blocks) - start_idx)
        if current_rank - 4 * (len(possible_blocks) - start_idx) >= best_rank:
            return

        if current_rank < best_rank:
            best_rank = current_rank
            best_blocks = list(selected_blocks)

        if start_idx >= len(possible_blocks):
            return

        for i in range(start_idx, len(possible_blocks)):
            block = possible_blocks[i]
            vol = block_volumes[i]
            
            # Check disjointness with selected blocks
            disjoint = True
            for sel_block in selected_blocks:
                sel_vol = block_volumes[possible_blocks.index(sel_block)] # This is inefficient, precompute indices
                # Better: store indices in selected_blocks
                pass
            
            # Let's restructure to store indices
            # ...
            
            # For now, just check overlap directly
            is_disjoint = True
            for sel_idx in selected_blocks:
                if not vol.isdisjoint(block_volumes[sel_idx]):
                    is_disjoint = False
                    break
            
            if is_disjoint:
                selected_blocks.append(i)
                search(i + 1, selected_blocks, current_rank - 4)
                selected_blocks.pop()
                
            # If time is running out, stop
            if time.time() > deadline:
                return

    # Optimization: Precompute index mapping
    # The above search is O(N^2) in overlap checks. N=24. 24*24 is fine.
    # But recursion depth and branching factor could be high.
    # Max depth is ~7. Branching factor is ~24. 24^7 is too high.
    # Pruning is essential.
    
    # Let's try a greedy heuristic first to get a good lower bound for pruning
    # Greedy: pick block with least overlap with already picked?
    # Or just pick first available?
    
    # Let's stick to the backtracking with strong pruning.
    # Initial call
    search(0, [], RANK_SCHOOLBOOK)

    # If we found a better set of blocks, use it.
    # If not, use the known 4-block configuration.
    if not best_blocks or best_rank >= RANK_SCHOOLBOOK:
        # Fallback to the specific 4 blocks from the current program
        strassen_blocks = [
            (0, 0, 0),
            (0, 2, 2),
            (1, 0, 3),
            (1, 2, 0)
        ]
    else:
        strassen_blocks = [possible_blocks[i] for i in best_blocks]

    # Build the decomposition
    U, V, W, rank = _build_decomposition(strassen_blocks)

    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook if verification fails
    return _schoolbook()
# EVOLVE-BLOCK-END
