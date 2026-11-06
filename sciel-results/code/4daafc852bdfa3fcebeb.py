import time
import random
import math

N = 128

# Precompute all nonconstant 4-term APs (0-indexed positions 0..127)
_aps = []
_pos_aps = [[] for _ in range(N)]
for _d in range(1, N // 3 + 1):
    for _a in range(0, N - 3 * _d):
        _ap = (_a, _a + _d, _a + 2 * _d, _a + 3 * _d)
        _idx = len(_aps)
        _aps.append(_ap)
        for _p in _ap:
            _pos_aps[_p].append(_idx)

_APS = _aps
_POS_APS = _pos_aps
_NUM_APS = len(_APS)


def _init_cost(c):
    """Initialize bad AP counts and total using bitmask-based distinct count."""
    ap_bad = [0] * _NUM_APS
    total = 0
    for idx, ap in enumerate(_APS):
        mask = 0
        for i in ap:
            mask |= (1 << c[i])
        # Count bits
        d = bin(mask).count('1')
        if d <= 2:
            ap_bad[idx] = 1
            total += 1
    return ap_bad, total


def _distinct_count(c, ap):
    """Count distinct colors in a 4-element AP using bitmask."""
    mask = 0
    for i in ap:
        mask |= (1 << c[i])
    return bin(mask).count('1')


def _sa_run(c, k, budget, rng):
    """Simulated annealing to minimize violations, then try to reduce colors."""
    ap_bad, total = _init_cost(c)
    best_c = list(c)
    best_total = total
    t0 = time.time()
    T = 2.0
    cool = 0.99998
    exp = math.exp
    while time.time() - t0 < budget:
        p = rng.randrange(N)
        newc = rng.randrange(k)
        if newc == c[p]:
            continue
        oldc = c[p]
        old_total = total
        c[p] = newc
        delta = 0
        for idx in _POS_APS[p]:
            ap = _APS[idx]
            d = _distinct_count(c, ap)
            bad = 1 if d <= 2 else 0
            if bad != ap_bad[idx]:
                delta += bad - ap_bad[idx]
                ap_bad[idx] = bad
        new_total = old_total + delta
        if new_total < best_total:
            best_total = total if new_total < total else new_total
            best_total = new_total
            best_c = list(c)
        if new_total <= old_total or rng.random() < exp(-delta / T):
            total = new_total
        else:
            c[p] = oldc
            for idx in _POS_APS[p]:
                ap = _APS[idx]
                d = _distinct_count(c, ap)
                bad = 1 if d <= 2 else 0
                ap_bad[idx] = bad
            total = old_total
        T *= cool
    return best_c, best_total


def _greedy_construct(k, rng):
    """Greedy construction: assign colors one position at a time."""
    n = N
    colors = [-1] * n
    
    for i in range(n):
        bad_colors = [False] * k
        
        for idx in _pos_aps[i]:
            ap = _APS[idx]
            others = [j for j in ap if j != i]
            # Check if all others are assigned
            if any(colors[j] == -1 for j in others):
                continue
            
            # For each candidate color, check if it would cause < 3 distinct
            for cc in range(k):
                mask = (1 << cc)
                for j in others:
                    mask |= (1 << colors[j])
                if bin(mask).count('1') < 3:
                    bad_colors[cc] = True
        
        available = [c for c in range(k) if not bad_colors[c]]
        if not available:
            return None
        colors[i] = rng.choice(available)
    
    # Verify
    ap_bad, total = _init_cost(colors)
    if total == 0:
        return colors
    return None


def _try_eliminate_color(c, k, rng, deadline):
    """Try to eliminate one color by reassigning its positions."""
    n = N
    used = sorted(set(c))
    if len(used) <= 3:
        return c, len(used)
    
    # Try to eliminate each color
    for target in used:
        if time.time() > deadline - 5.0:
            break
        positions_with_target = [i for i in range(n) if c[i] == target]
        if not positions_with_target:
            continue
        
        # Try to reassign each position to another used color
        other_colors = [x for x in used if x != target]
        improved = False
        for pos in positions_with_target:
            if time.time() > deadline - 5.0:
                break
            old_color = c[pos]
            for new_color in other_colors:
                c[pos] = new_color
                # Check validity
                valid = True
                for idx in _pos_aps[pos]:
                    ap = _APS[idx]
                    d = _distinct_count(c, ap)
                    if d < 3:
                        valid = False
                        break
                if valid:
                    # Check if we actually reduced colors
                    if c.count(target) == 0:
                        improved = True
                        break
                    # Keep the change even if target still exists, might help later
                    break
                else:
                    c[pos] = old_color
            if improved:
                break
        
        if improved:
            new_used = sorted(set(c))
            if len(new_used) < len(used):
                # Remap to consecutive
                remap = {v: i for i, v in enumerate(new_used)}
                c = [remap[x] for x in c]
                return c, len(new_used)
    
    return c, len(used)


def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    
    # Guaranteed valid fallback: each position a unique color
    base = list(range(N))
    best_valid = base
    best_distinct = N
    
    rng = random.Random()
    
    def _try_k(k, budget, starts):
        nonlocal best_valid, best_distinct
        t_end = time.time() + budget
        it = 0
        while time.time() < t_end and time.time() < deadline - 3.0:
            # Try greedy first
            if it == 0:
                g = _greedy_construct(k, rng)
                if g is not None:
                    used = sorted(set(g))
                    m = len(used)
                    if m < best_distinct:
                        remap = {v: i for i, v in enumerate(used)}
                        best_valid = [remap[x] for x in g]
                        best_distinct = m
                        if m <= 3:
                            return True
                    # Try to eliminate more colors
                    g2, m2 = _try_eliminate_color(g, k, rng, deadline)
                    if m2 < best_distinct:
                        remap = {v: i for i, v in enumerate(sorted(set(g2)))}
                        best_valid = [remap[x] for x in g2]
                        best_distinct = m2
                        if m2 <= 3:
                            return True
                    it += 1
                    continue
            
            # SA from random or structured start
            start = list(starts[it % len(starts)])
            remaining = max(0.3, min(3.0, t_end - time.time()))
            c, tot = _sa_run(start, k, remaining, rng)
            if tot == 0:
                used = sorted(set(c))
                m = len(used)
                if m < best_distinct:
                    remap = {v: i for i, v in enumerate(used)}
                    best_valid = [remap[x] for x in c]
                    best_distinct = m
                    if m <= 3:
                        return True
                # Try to eliminate more colors from valid solution
                c2, m2 = _try_eliminate_color(c, k, rng, deadline)
                if m2 < best_distinct:
                    remap = {v: i for i, v in enumerate(sorted(set(c2)))}
                    best_valid = [remap[x] for x in c2]
                    best_distinct = m2
                    if m2 <= 3:
                        return True
            it += 1
        return False
    
    # Palette sizes in ascending order; smaller is better
    plan = [
        (3, 20.0, [[i % 3 for i in range(N)],
                   [random.randrange(3) for _ in range(N)]]),
        (4, 15.0, [[i % 4 for i in range(N)],
                   [random.randrange(4) for _ in range(N)]]),
        (5, 10.0, [[i % 5 for i in range(N)],
                   [random.randrange(5) for _ in range(N)]]),
        (6, 8.0, [[i % 6 for i in range(N)],
                   [random.randrange(6) for _ in range(N)]]),
        (8, 5.0, [[i % 8 for i in range(N)],
                   [random.randrange(8) for _ in range(N)]]),
    ]
    
    for k, budget, starts in plan:
        if time.time() > deadline - 5.0:
            break
        b = min(budget, deadline - time.time())
        if b < 2.0:
            break
        if _try_k(k, b, starts):
            break
    
    return best_valid
