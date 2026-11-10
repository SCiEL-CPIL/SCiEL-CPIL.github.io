import random
import time
import numpy as np

_START = time.time()
_DEADLINE = 150.0 - 3.0   # safety margin


def _edge_index(i, j):
    # 0-based index for pair (i,j), 0<=i<j<16, in lexicographic order
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)


def _build_k4s():
    # For each K4, store its 6 edge indices
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ))
    return k4s


_K4S = _build_k4s()
_K4S_ARR = np.array(_K4S, dtype=np.int32)  # shape (1820, 6)
_NUM_K4 = len(_K4S)

# Precompute the 15 pair indices within a K4's 6 edges
_PAIR_IDX = [(0,1),(0,2),(0,3),(0,4),(0,5),
             (1,2),(1,3),(1,4),(1,5),
             (2,3),(2,4),(2,5),
             (3,4),(3,5),
             (4,5)]


def _colour_counts(colours):
    cnt = [0] * 120
    for c in colours:
        cnt[c] += 1
    return cnt


def _validate_np(colours_arr):
    """Vectorized validation: return True if every K4 has >= 5 distinct colours."""
    edge_cols = colours_arr[_K4S_ARR]  # shape (1820, 6)
    num_equal = np.zeros(_NUM_K4, dtype=np.int32)
    for p, q in _PAIR_IDX:
        num_equal += (edge_cols[:, p] == edge_cols[:, q]).astype(np.int32)
    return np.all(num_equal <= 1)


def _try_merge(colours, cnt, a, b):
    # Return new colours list if merging colour b into a keeps validity, else None.
    if cnt[b] > cnt[a]:
        return None
    new = list(colours)
    for idx, c in enumerate(new):
        if c == b:
            new[idx] = a
    new_arr = np.array(new, dtype=np.int32)
    if not _validate_np(new_arr):
        return None
    newcnt = list(cnt)
    newcnt[a] += newcnt[b]
    newcnt[b] = 0
    return new, newcnt


def _num_colours(cnt):
    return sum(1 for x in cnt if x > 0)


def construct():
    # Start from the known valid 17-colour solution
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    cnt = _colour_counts(colours)
    best = list(colours)
    best_cnt = _num_colours(cnt)

    rng = random.Random(12345)

    # Deterministic greedy merge: repeatedly try to merge the least-used colour
    # into another colour, preferring to keep the palette small.
    def greedy(colours, cnt):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        progress = True
        while progress:
            progress = False
            used = [c for c in range(120) if cnt[c] > 0]
            if len(used) <= 1:
                break
            used.sort(key=lambda c: (cnt[c], c))
            b = used[0]  # least used
            if cnt[b] == 0:
                continue
            targets = [a for a in used if a != b]
            targets.sort(key=lambda c: (-cnt[c], c))
            for a in targets:
                res = _try_merge(colours, cnt, a, b)
                if res is not None:
                    colours, cnt = res
                    nc -= 1
                    progress = True
                    break
        return colours, cnt, nc

    colours, cnt, nc = greedy(colours, cnt)
    if nc < best_cnt:
        best = list(colours)
        best_cnt = nc

    # Simulated annealing local search with merge operations
    def sa_local_search(colours, cnt, iters=15000, t0=1.0, t_end=0.01, cooling=0.9999):
        colours = list(colours)
        cnt = list(cnt)
        nc = _num_colours(cnt)
        T = t0
        for _ in range(iters):
            if time.time() > _DEADLINE:
                break
            used = [c for c in range(120) if cnt[c] > 0]
            if len(used) <= 1:
                break
            # Pick a colour to try to eliminate
            r = rng.random()
            if r < 0.7:
                used_sorted = sorted(used, key=lambda c: cnt[c])
                b = used_sorted[rng.randrange(min(len(used_sorted) // 2 + 1, len(used_sorted)))]
            else:
                b = used[rng.randrange(len(used))]
            if cnt[b] == 0:
                continue
            # Pick a target
            targets = [a for a in used if a != b]
            r2 = rng.random()
            if r2 < 0.6:
                targets_sorted = sorted(targets, key=lambda c: -cnt[c])
                a = targets_sorted[rng.randrange(min(len(targets_sorted) // 2 + 1, len(targets_sorted)))]
            else:
                a = targets[rng.randrange(len(targets))]
            
            # Try the merge
            if cnt[b] > cnt[a]:
                # Swapped merge: merge a into b
                a, b = b, a
                if cnt[b] > cnt[a]:
                    continue
            
            res = _try_merge(colours, cnt, a, b)
            if res is not None:
                colours, cnt = res
                nc -= 1
            else:
                # Accept worse state with probability exp(-delta/T)
                # Here "worse" means we couldn't merge, so no change
                # But we can also try single-edge colour swaps
                if rng.random() < 0.3:
                    # Try a random single-edge colour change
                    idx = rng.randrange(120)
                    old_c = colours[idx]
                    new_c = used[rng.randrange(len(used))]
                    if old_c == new_c:
                        continue
                    colours[idx] = new_c
                    new_arr = np.array(colours, dtype=np.int32)
                    if _validate_np(new_arr):
                        newcnt = list(cnt)
                        newcnt[old_c] -= 1
                        newcnt[new_c] += 1
                        if newcnt[old_c] == 0:
                            nc_new = _num_colours(newcnt)
                        else:
                            nc_new = nc
                        if nc_new <= nc or rng.random() < np.exp(-(nc_new - nc) / max(T, 1e-10)):
                            cnt = newcnt
                            nc = nc_new
                        else:
                            colours[idx] = old_c
                    else:
                        colours[idx] = old_c
            
            T *= cooling
        return colours, cnt, nc

    # Try several random perturbations + SA local search
    for restart in range(300):
        if time.time() > _DEADLINE:
            break
        # Start from best, apply a few random single-edge colour changes to perturb
        cur = list(best)
        curcnt = _colour_counts(cur)
        used_colors = [c for c in range(120) if curcnt[c] > 0]
        num_changes = rng.randint(1, 3)
        for _ in range(num_changes):
            idx = rng.randrange(120)
            cur[idx] = used_colors[rng.randrange(len(used_colors))]
        # validate; if invalid, skip
        cur_arr = np.array(cur, dtype=np.int32)
        if not _validate_np(cur_arr):
            continue
        curcnt = _colour_counts(cur)
        # SA local search
        cur, curcnt, nc = sa_local_search(cur, curcnt, iters=10000)
        if nc < best_cnt:
            best = list(cur)
            best_cnt = nc
            # try greedy from here too
            gcol, gcnt, gnc = greedy(best, _colour_counts(best))
            if gnc < best_cnt:
                best = list(gcol)
                best_cnt = gnc

    # Final greedy polish
    gcol, gcnt, gnc = greedy(best, _colour_counts(best))
    if gnc < best_cnt:
        best = list(gcol)
        best_cnt = gnc

    return best
