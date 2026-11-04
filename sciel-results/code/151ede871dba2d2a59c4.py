"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 matrix multiplication."""
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 6]
TIME_LIMIT = 120.0
START = time.time()


def _remaining_budget():
    return TIME_LIMIT - (time.time() - START)


def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    total_cells = N * M * P  # 90

    # Precompute all nonempty cube subsets (u,v,w) with at least one 1.
    # Each cube covers cells (i,j,k) where u[i]=v[j]=w[k]=1.
    cubes = []
    for u in range(1 << n):
        for v in range(1 << m):
            for w in range(1 << p):
                if u == 0 or v == 0 or w == 0:
                    continue
                # covered cells as bitset
                mask = 0
                cnt = 0
                for i in range(n):
                    if not (u >> i) & 1:
                        continue
                    for j in range(m):
                        if not (v >> j) & 1:
                            continue
                        for k in range(p):
                            if (w >> k) & 1:
                                idx = (i * m + j) * p + k
                                mask |= (1 << idx)
                                cnt += 1
                cubes.append((mask, cnt, u, v, w))

    full_mask = (1 << total_cells) - 1
    num_cubes = len(cubes)

    # Build cell -> list of cube indices covering it
    cell_to_cubes = [[] for _ in range(total_cells)]
    for ci, (mask, cnt, u, v, w) in enumerate(cubes):
        mm = mask
        while mm:
            lb = mm & (-mm)
            c = lb.bit_length() - 1
            cell_to_cubes[c].append(ci)
            mm ^= lb

    # Rank lower bound: each cube covers at most max_cnt cells
    max_cnt = max(c[1] for c in cubes)
    import math
    lb = math.ceil(total_cells / max_cnt)

    # Also a simple bound: need at least 1 per row? Use max_cnt bound.

    best = None
    best_rank = total_cells  # schoolbook
    deadline = START + TIME_LIMIT - 3.0  # safety margin

    # Sort cells by number of covering cubes (most constrained first)
    cell_order = sorted(range(total_cells), key=lambda c: len(cell_to_cubes[c]))

    # We'll do a best-first / branch-and-bound search over cells, choosing a cube to cover the
    # currently-most-uncovered cell. State: (uncovered_mask, chosen_cubes, depth).
    # To keep it tractable, use a priority queue by (depth, heuristic) and an upper bound.

    import heapq
    # heuristic: greedy lower bound on remaining cells
    def greedy_lb(mask):
        # greedy set cover lower bound using max coverage per step
        # approximate: number of cells / max cube coverage within mask
        if mask == 0:
            return 0
        # compute max overlap of any cube with mask
        mx = 0
        for ci, (cmask, cnt, u, v, w) in enumerate(cubes):
            ov = (cmask & mask).bit_count()
            if ov > mx:
                mx = ov
        if mx == 0:
            return 10 ** 9
        return math.ceil(mask.bit_count() / mx)

    # Priority queue entries: (depth, heuristic, counter, uncovered_mask, chosen_cubes_tuple)
    counter = 0
    start_mask = full_mask
    start_heur = greedy_lb(start_mask)
    heap = [(0, start_heur, counter, start_mask, ())]
    counter += 1

    # Precompute cube masks list for fast access
    cube_masks = [c[0] for c in cubes]

    nodes = 0
    while heap and time.time() < deadline:
        depth, heur, _, mask, chosen = heapq.heappop(heap)
        nodes += 1
        if nodes % 20000 == 0 and _remaining_budget() < 5.0:
            break
        if mask == 0:
            rank = depth
            if rank < best_rank:
                best_rank = rank
                best = chosen
                if best_rank == lb:
                    break
            continue
        if depth + heur >= best_rank:
            continue
        # pick the most constrained uncovered cell
        # find uncovered cell with fewest covering cubes
        mm = mask
        best_cell = -1
        best_cnt = 10 ** 9
        while mm:
            lb2 = mm & (-mm)
            c = lb2.bit_length() - 1
            cov = 0
            for ci in cell_to_cubes[c]:
                if cube_masks[ci] & mask:
                    cov += 1
            if cov < best_cnt:
                best_cnt = cov
                best_cell = c
                if cov <= 1:
                    break
            mm ^= lb2
        # branch on cubes covering best_cell
        options = []
        for ci in cell_to_cubes[best_cell]:
            cm = cube_masks[ci]
            if cm & mask:
                new_mask = mask & ~cm
                new_depth = depth + 1
                if new_depth >= best_rank:
                    continue
                new_heur = greedy_lb(new_mask)
                if new_depth + new_heur >= best_rank:
                    continue
                heapq.heappush(heap, (new_depth, new_heur, counter, new_mask, chosen + (ci,)))
                counter += 1
        if nodes % 5000 == 0:
            # occasionally update greedy upper bound to prune
            pass

    # If no better than schoolbook found, fall back to schoolbook
    if best is None:
        rank = total_cells
        chosen = tuple(range(total_cells))
        # schoolbook: each cell is its own cube with single 1
        # But our cubes list doesn't have single-cell cubes necessarily; build manually.
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = len(best)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    for c, ci in enumerate(best):
        u, v, w = cubes[ci][2], cubes[ci][3], cubes[ci][4]
        for i in range(n):
            if (u >> i) & 1:
                for j in range(m):
                    if (v >> j) & 1:
                        U[i * m + j][c] = 1
        for j in range(m):
            if (v >> j) & 1:
                for k in range(p):
                    if (w >> k) & 1:
                        V[j * p + k][c] = 1
        for k in range(p):
            if (w >> k) & 1:
                for i in range(n):
                    if (u >> i) & 1:
                        W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
