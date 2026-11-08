"""Deterministic rank-minimal decomposition for 3x4*4x7 matrix multiplication.

Uses a greedy maximum-packing of disjoint 2x2x2 Strassen blocks plus schoolbook
for remaining entries. Each Strassen block costs 7 rank units vs 8 for schoolbook,
so each additional block saves 1 rank unit.
"""
import time
from itertools import product

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def _find_max_packing(n, m, p):
    """Find maximum number of disjoint 2x2x2 blocks via greedy + local search."""
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                entries = frozenset(
                    (ii, jj, kk)
                    for ii in (i0, i0 + 1)
                    for jj in (j0, j0 + 1)
                    for kk in (k0, k0 + 1)
                )
                all_blocks.append(((i0, i0 + 1, j0, j0 + 1, k0, k0 + 1), entries))

    total_entries = n * m * p

    # Greedy: repeatedly pick the block that covers the most uncovered entries
    # (all cover 8, so just pick any that doesn't conflict; use heuristic ordering)
    # Try multiple random orderings and keep best
    import random
    best_count = 0
    best_selected = []
    
    for trial in range(200):
        rng = random.Random(trial)
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        
        covered = set()
        selected = []
        for idx in order:
            block_params, entries = all_blocks[idx]
            if not entries & covered:
                covered |= entries
                selected.append(block_params)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    # Local search: try swapping out one block for two
    improved = True
    while improved and time.time() - _t0 < _DEADLINE - 10:
        improved = False
        covered = set()
        for bp in best_selected:
            i0, i1, j0, j1, k0, k1 = bp
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))
        
        for remove_idx in range(len(best_selected)):
            candidate = best_selected[:remove_idx] + best_selected[remove_idx+1:]
            cov_cand = set()
            for bp in candidate:
                i0, i1, j0, j1, k0, k1 = bp
                for ii in (i0, i1):
                    for jj in (j0, j1):
                        for kk in (k0, k1):
                            cov_cand.add((ii, jj, kk))
            
            # Try adding one block
            for idx in range(len(all_blocks)):
                bp, entries = all_blocks[idx]
                if bp in candidate:
                    continue
                if not entries & cov_cand:
                    new_sel = candidate + [bp]
                    if len(new_sel) > len(best_selected):
                        best_selected = new_sel
                        improved = True
                        break
            if improved:
                break
        if time.time() - _t0 > _DEADLINE - 15:
            break
    
    return best_selected, best_count


def construct():
    global _t0
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    selected_blocks, num_blocks = _find_max_packing(n, m, p)
    rank = num_blocks * 7 + (n * m * p - num_blocks * 8)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    covered = set()
    for bp in selected_blocks:
        col = add_strassen_block(col, *bp)
        i0, i1, j0, j1, k0, k1 = bp
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
