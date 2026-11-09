"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations, permutations

# EVOLVE-BLOCK-START

_Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]

# Precompute for each Q3 edge, the other edges incident to each endpoint
_Q3_ADJ = {i: set() for i in range(8)}
for a, b in _Q3_EDGES:
    _Q3_ADJ[a].add(b)
    _Q3_ADJ[b].add(a)


def _has_q3_in_subset(adj_set, subset):
    """Check if the induced subgraph on 'subset' contains Q3."""
    sub = list(subset)
    # Check all 40320 permutations
    for perm in permutations(range(8)):
        ok = True
        for (a, b) in _Q3_EDGES:
            u, v = sub[perm[a]], sub[perm[b]]
            if u > v:
                u, v = v, u
            if (u, v) not in adj_set:
                ok = False
                break
        if ok:
            return True
    return False


def _add_edge_creates_q3(adj_set, u, v):
    """Check if adding edge (u,v) creates a Q3 subgraph."""
    # New Q3 must use edge (u,v)
    for (a, b) in _Q3_EDGES:
        # Try a->u, b->v
        if _check_partial(adj_set, {a: u, b: v}):
            return True
        # Try a->v, b->u
        if _check_partial(adj_set, {a: v, b: u}):
            return True
    return False


def _check_partial(adj_set, mapping):
    """Check if partial mapping can be extended to full Q3 embedding."""
    mapped_vals = list(mapping.values())
    if len(set(mapped_vals)) != len(mapped_vals):
        return False
    
    # Check edges among mapped
    for (a, b) in _Q3_EDGES:
        if a in mapping and b in mapping:
            x, y = mapping[a], mapping[b]
            if x > y:
                x, y = y, x
            if (x, y) not in adj_set:
                return False
    
    unmapped = [i for i in range(8) if i not in mapping]
    used = set(mapping.values())
    avail = [v for v in range(16) if v not in used]
    
    def bt(idx):
        if idx == len(unmapped):
            fm = dict(mapping)
            for i, qv in enumerate(unmapped):
                fm[qv] = asg[i]
            for (a, b) in _Q3_EDGES:
                x, y = fm[a], fm[b]
                if x > y:
                    x, y = y, x
                if (x, y) not in adj_set:
                    return False
            return True
        
        qv = unmapped[idx]
        for gv in avail:
            ok = True
            for nb in _Q3_ADJ[qv]:
                if nb in mapping:
                    x, y = gv, mapping[nb]
                    if x > y:
                        x, y = y, x
                    if (x, y) not in adj_set:
                        ok = False
                        break
            if not ok:
                continue
            asg.append(gv)
            if bt(idx + 1):
                return True
            asg.pop()
        return False
    
    asg = []
    return bt(0)


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    all_edges = set()
    for u in range(16):
        for v in range(u + 1, 16):
            all_edges.add((u, v))
    
    # Start with K3,13
    current = set()
    for a in range(3):
        for b in range(3, 16):
            current.add((a, b))
    
    best = current.copy()
    best_n = len(best)
    
    remaining = list(all_edges - current)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        random.shuffle(remaining)
        trial_set = current.copy()
        for e in remaining:
            if time.time() > deadline:
                break
            u, v = e
            if _add_edge_creates_q3(trial_set, u, v):
                continue
            trial_set.add(e)
        if len(trial_set) > best_n:
            best_n = len(trial_set)
            best = trial_set.copy()
    
    return [[u, v] for (u, v) in best]

# EVOLVE-BLOCK-END
