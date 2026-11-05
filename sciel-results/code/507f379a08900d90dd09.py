"""Maximum cardinality subset of [1,512] with pairwise lcm <= 512.

Structural change: use a Tomita-style Maximum Clique solver on the COMPATIBILITY
graph (where an edge means lcm(a,b) <= 512). The compatibility graph is much
denser than the conflict graph, making greedy coloring bounds tighter and the
branch-and-bound search far more efficient.

Key insight: numbers that divide 512 (powers of 2 up to 512) form a clique
in the compatibility graph. More generally, divisors of highly composite
numbers form large cliques. We exploit this structure.
"""
import time
from math import gcd
from itertools import combinations

LIMIT = 512
DEADLINE = 85.0  # safety margin before 90s kill


def _lcm(a, b):
    return a * b // gcd(a, b)


def _validate(s):
    s = list(set(s))
    if not s:
        return False
    for x in s:
        if x < 1 or x > LIMIT:
            return False
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if _lcm(s[i], s[j]) > LIMIT:
                return False
    return True


def _build_compatibility():
    """Build compatibility bitmask: compat[a] has bit b set if lcm(a,b) <= 512."""
    n = LIMIT
    compat = [0] * (n + 1)
    for a in range(1, n + 1):
        mask = 0
        for b in range(a + 1, n + 1):
            if _lcm(a, b) <= LIMIT:
                mask |= (1 << b)
                compat[b] |= (1 << a)
        compat[a] = mask
    return compat


def _greedy_color_compat(cand_mask, compat):
    """Greedy coloring of the candidate set in the compatibility graph.
    In a compatibility graph, independent set = clique in the complement (conflict).
    But we're solving max clique on compat. For max clique, we color the COMPAT graph
    to get an upper bound: vertices with the same color form an independent set in compat,
    so at most one per color can be in a clique.
    
    Returns (num_colors, ordered_vertices).
    """
    uncolored = cand_mask
    color = 0
    colors = {}
    while uncolored:
        color += 1
        avail = uncolored
        while avail:
            # Pick vertex with highest bit
            v_bit = 1 << (avail.bit_length() - 1)
            v = v_bit.bit_length() - 1
            colors[v] = color
            # Remove v and its non-neighbors in compat from avail
            # In compat graph, neighbors are those with bit set in compat[v]
            # Non-neighbors (in the graph) are those NOT in compat[v]
            # For coloring an independent set, we remove v and its neighbors
            avail &= ~(v_bit | (compat[v] & uncolored))
            uncolored &= ~v_bit
    ordered = sorted(colors.keys(), key=lambda v: (colors[v], v))
    return color, ordered


def _greedy_clique(cand_mask, compat):
    """Greedy construction of a clique in the compatibility graph."""
    clique = 0
    size = 0
    cand = cand_mask
    while cand:
        # Pick vertex with highest degree in cand
        best_v = 0
        best_deg = -1
        c = cand
        while c:
            v_bit = 1 << (c.bit_length() - 1)
            v = v_bit.bit_length() - 1
            deg = bin(compat[v] & cand).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c &= ~(1 << v)
        if best_v == 0:
            break
        clique |= (1 << best_v)
        size += 1
        # Restrict to neighbors of best_v
        cand &= compat[best_v]
    return clique, size


def _max_clique_tomita(compat, deadline):
    """Tomita-style maximum clique solver on the compatibility graph."""
    n = LIMIT
    full_mask = (1 << (n + 1)) - 2  # bits 1..n
    
    best_clique = 0
    best_size = 0
    
    aborted = [False]
    
    def color_sort(cand_mask):
        """Return (num_colors, ordered_vertices) for greedy coloring."""
        uncolored = cand_mask
        color = 0
        colors = {}
        while uncolored:
            color += 1
            avail = uncolored
            while avail:
                v_bit = 1 << (avail.bit_length() - 1)
                v = v_bit.bit_length() - 1
                colors[v] = color
                avail &= ~(v_bit | (compat[v] & uncolored))
                uncolored &= ~v_bit
        ordered = sorted(colors.keys(), key=lambda v: (colors[v], v))
        return color, ordered
    
    def expand(cand_mask, current_clique, current_size):
        nonlocal best_clique, best_size, aborted
        if aborted[0] or time.time() > deadline:
            aborted[0] = True
            return
        
        if cand_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique
            return
        
        # Greedy coloring upper bound
        bound, ordered = color_sort(cand_mask)
        if current_size + bound <= best_size:
            return
        
        # Try to extend greedily first for a quick improvement
        trial_clique = current_clique
        trial_size = current_size
        for v in ordered:
            if not (cand_mask >> v) & 1:
                continue
            trial_clique |= (1 << v)
            trial_size += 1
            cand_mask &= ~(1 << v)
            cand_mask &= compat[v]
            if trial_size > best_size:
                best_size = trial_size
                best_clique = trial_clique
        
        # Now do proper branch and bound
        # Process vertices in order of increasing color (tightest bound first)
        for v in reversed(ordered):
            if aborted[0] or time.time() > deadline:
                aborted[0] = True
                return
            if not (cand_mask >> v) & 1:
                continue
            
            # Include v
            new_cand = cand_mask & compat[v] & ~(1 << v)
            expand(new_cand, current_clique | (1 << v), current_size + 1)
            
            # Exclude v
            cand_mask &= ~(1 << v)
            
            if cand_mask == 0:
                break
            
            # Re-check bound
            cb, _ = color_sort(cand_mask)
            if current_size + cb <= best_size:
                break
    
    expand(full_mask, 0, 0)
    return {i for i in range(1, n + 1) if (best_clique >> i) & 1}


def _initial_clique(compat):
    """Get a good initial clique using structural insights."""
    n = LIMIT
    
    # Strategy: Start with divisors of 512 (powers of 2)
    # These form a clique since 2^a | 2^b for a <= b
    base = set()
    p = 1
    while p <= n:
        base.add(p)
        p *= 2
    # base = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    
    # Try to greedily add more vertices
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if v in base:
                continue
            # Check if v is compatible with all in base
            ok = True
            for u in base:
                if (compat[v] >> u) & 1 == 0:
                    ok = False
                    break
            if ok:
                base.add(v)
                improved = True
    
    # Also try starting from divisors of 360
    base2 = set()
    for d in range(1, 361):
        if 360 % d == 0:
            base2.add(d)
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if v in base2:
                continue
            ok = True
            for u in base2:
                if (compat[v] >> u) & 1 == 0:
                    ok = False
                    break
            if ok:
                base2.add(v)
                improved = True
    
    # Also try starting from all numbers <= 24 (they're all compatible with each other? Let's check)
    # lcm(a,b) <= 24*24 = 576 which may exceed 512, so not all pairs work
    base3 = set(range(1, 13))  # 1-12: lcm(11,12)=132<=512, lcm(7,11)=77<=512, etc.
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if v in base3:
                continue
            ok = True
            for u in base3:
                if (compat[v] >> u) & 1 == 0:
                    ok = False
                    break
            if ok:
                base3.add(v)
                improved = True
    
    candidates = [base, base2, base3]
    best = max(candidates, key=len)
    return best


def _local_search(compat, s, deadline):
    """Local search on the clique: try swaps and additions."""
    s = set(s)
    start = time.time()
    
    def compatible_with_all(v, base_set):
        for u in base_set:
            if v != u and not ((compat[v] >> u) & 1):
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try pure addition
        for y in range(1, LIMIT + 1):
            if y in s:
                continue
            if compatible_with_all(y, s):
                s.add(y)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove x, add y, then try to add another
        found = False
        s_list = list(s)
        for x in s_list:
            if time.time() > deadline:
                break
            base = s - {x}
            for y in range(1, LIMIT + 1):
                if y in base:
                    continue
                if not compatible_with_all(y, base):
                    continue
                s_trial = base | {y}
                # Try to add another
                for z in range(1, LIMIT + 1):
                    if z in s_trial:
                        continue
                    if compatible_with_all(z, s_trial):
                        s = s_trial | {z}
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            continue
    
    return s


def construct():
    start = time.time()
    compat = _build_compatibility()
    
    # Phase 1: Get initial solution
    best = _initial_clique(compat)
    best_size = len(best)
    
    # Phase 2: Tomita maximum clique
    mc_deadline = start + DEADLINE - 10.0
    if time.time() < mc_deadline:
        mc_result = _max_clique_tomita(compat, mc_deadline)
        if len(mc_result) > best_size:
            best = mc_result
            best_size = len(best)
    
    # Phase 3: Local search
    ls_deadline = start + DEADLINE - 1.0
    if time.time() < ls_deadline and best:
        best = _local_search(compat, best, ls_deadline)
    
    # Validate
    if not _validate(best):
        # Fallback: greedy
        best = set()
        for v in range(1, LIMIT + 1):
            if all(_lcm(u, v) <= LIMIT for u in best):
                best.add(v)
    
    if not best:
        best = {1}
    
    return list(best)
