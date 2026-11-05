"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    n = 128
    
    # Precompute 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # For each position, list of AP indices containing it
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(i)
    
    def check_valid_partial(colors, pos):
        """Check if assigning colors[pos] is valid given current colors."""
        c = colors[pos]
        for ai in pos_aps[pos]:
            p1, p2, p3, p4 = aps[ai]
            cs = [colors[p1], colors[p2], colors[p3], colors[p4]]
            if -1 in cs:
                continue
            if len(set(cs)) < 3:
                return False
        return True
    
    def greedy():
        colors = [-1] * n
        for pos in range(n):
            used = set(c for c in colors if c != -1)
            cands = sorted(used) + ([max(used) + 1] if used else [0])
            for c in cands:
                colors[pos] = c
                if check_valid_partial(colors, pos):
                    break
            else:
                colors[pos] = 0
        return colors
    
    def full_check(colors):
        for p1, p2, p3, p4 in aps:
            if len({colors[p1], colors[p2], colors[p3], colors[p4]}) < 3:
                return False
        return True
    
    def reduce_colors(colors):
        colors = colors[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            ds = sorted(set(colors))
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    a, b = ds[i], ds[j]
                    nc = [a if x == b else x for x in colors]
                    if full_check(nc):
                        colors = nc
                        changed = True
                        break
                if changed:
                    break
        return colors
    
    def ndistinct(colors):
        return len(set(colors))
    
    # Phase 1: Greedy
    best = greedy()
    best = reduce_colors(best)
    best_k = ndistinct(best)
    
    # Phase 2: Local search
    while time.time() < deadline - 15:
        prog = False
        ds = sorted(set(best))
        for pos in range(n):
            if time.time() > deadline - 15:
                break
            for c in ds:
                if c == best[pos]:
                    continue
                old = best[pos]
                best[pos] = c
                if full_check(best):
                    k = ndistinct(best)
                    if k < best_k:
                        best_k = k
                        prog = True
                    else:
                        best[pos] = old
                else:
                    best[pos] = old
        if not prog:
            break
    
    best = reduce_colors(best)
    
    # Remap
    ds = sorted(set(best))
    mp = {c: i for i, c in enumerate(ds)}
    return [mp[c] for c in best]
# EVOLVE-BLOCK-END
