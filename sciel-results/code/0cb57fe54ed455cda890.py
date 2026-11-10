import random
import time

_START = time.time()
_DEADLINE = 145.0  # 5s margin before 150s kill

def _edge_index(i, j):
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)

def _build_k4s():
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
_NUM_K4S = len(_K4S)

def _colour_counts(colours):
    cnt = [0] * 120
    for c in colours:
        cnt[c] += 1
    return cnt

def _num_colours(cnt):
    return sum(1 for x in cnt if x > 0)

def _is_valid(colours):
    for e in _K4S:
        s = {colours[e[0]], colours[e[1]], colours[e[2]], colours[e[3]], colours[e[4]], colours[e[5]]}
        if len(s) < 5:
            return False
    return True

def _try_merge(colours, cnt, a, b):
    """Try to merge color b into color a. Returns (new_colours, new_cnt) or None."""
    if cnt[b] == 0:
        return None
    # Prefer merging less-used into more-used to keep distribution balanced
    if cnt[b] > cnt[a]:
        return None
    new = list(colours)
    for idx in range(120):
        if new[idx] == b:
            new[idx] = a
    # Validate all K4s
    for e in _K4S:
        s = {new[e[0]], new[e[1]], new[e[2]], new[e[3]], new[e[4]], new[e[5]]}
        if len(s) < 5:
            return None
    newcnt = [0] * 120
    for c in new:
        newcnt[c] += 1
    return new, newcnt

def _greedy_merge(colours, cnt):
    """Repeatedly try to merge the least-used color into another, preferring valid merges."""
    colours = list(colours)
    cnt = list(cnt)
    nc = _num_colours(cnt)
    progress = True
    while progress:
        progress = False
        used = [c for c in range(120) if cnt[c] > 0]
        if len(used) <= 1:
            break
        # Sort by count ascending, then by color for determinism
        used.sort(key=lambda c: (cnt[c], c))
        for b in used:
            if cnt[b] == 0:
                continue
            # Try merging b into each other used color
            for a in used:
                if a == b:
                    continue
                res = _try_merge(colours, cnt, a, b)
                if res is not None:
                    colours, cnt = res
                    nc -= 1
                    progress = True
                    break
            if progress:
                break
    return colours, cnt, nc

def _local_search_merge(colours, cnt, rng, max_iters=5000):
    """Local search: try random merges, accept if valid, occasionally perturb."""
    colours = list(colours)
    cnt = list(cnt)
    nc = _num_colours(cnt)
    best_nc = nc
    best_colours = list(colours)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        if best_nc <= 13:
            break
        
        used = [c for c in range(120) if cnt[c] > 0]
        if len(used) <= 1:
            break
        
        # Pick a random color to try to eliminate
        b = used[rng.randrange(len(used))]
        if cnt[b] == 0:
            continue
        
        # Pick a random target color
        targets = [a for a in used if a != b]
        if not targets:
            break
        a = targets[rng.randrange(len(targets))]
        
        res = _try_merge(colours, cnt, a, b)
        if res is not None:
            colours, cnt = res
            nc -= 1
            if nc < best_nc:
                best_nc = nc
                best_colours = list(colours)
        else:
            # Occasionally perturb: change a random edge to a random color
            if rng.random() < 0.1:
                idx = rng.randrange(120)
                old_c = colours[idx]
                new_c = rng.randrange(120)
                if old_c != new_c:
                    colours[idx] = new_c
                    # Update counts
                    cnt[old_c] -= 1
                    cnt[new_c] += 1
                    # Check validity
                    if _is_valid(colours):
                        nc = _num_colours(cnt)
                        if nc < best_nc:
                            best_nc = nc
                            best_colours = list(colours)
                    else:
                        # Revert
                        colours[idx] = old_c
                        cnt[old_c] += 1
                        cnt[new_c] -= 1
    
    return best_colours, _colour_counts(best_colours), best_nc

def construct():
    # Start from the known valid 17-coloring
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    cnt = _colour_counts(colours)
    best = list(colours)
    best_cnt = _num_colours(cnt)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy merge
    gcol, gcnt, gnc = _greedy_merge(colours, cnt)
    if gnc < best_cnt:
        best = list(gcol)
        best_cnt = gnc
    
    # Phase 2: Local search with random restarts
    for restart in range(50):
        if time.time() > _DEADLINE:
            break
        
        # Start from best, apply small perturbation
        cur = list(best)
        curcnt = _colour_counts(cur)
        
        # Perturb: change 1-3 random edges
        num_perturb = rng.randint(1, 3)
        for _ in range(num_perturb):
            idx = rng.randrange(120)
            cur[idx] = rng.randrange(120)
        
        # Check validity
        if not _is_valid(cur):
            continue
        curcnt = _colour_counts(cur)
        
        # Local search
        cur, curcnt, nc = _local_search_merge(cur, curcnt, rng, max_iters=3000)
        if nc < best_cnt:
            best = list(cur)
            best_cnt = nc
            # Try greedy from here
            gcol, gcnt, gnc = _greedy_merge(best, _colour_counts(best))
            if gnc < best_cnt:
                best = list(gcol)
                best_cnt = gnc
    
    # Phase 3: Final polish - try more aggressive merging
    if time.time() < _DEADLINE - 5:
        final_colours, final_cnt, final_nc = _greedy_merge(best, _colour_counts(best))
        if final_nc < best_cnt:
            best = list(final_colours)
            best_cnt = final_nc
    
    return best
