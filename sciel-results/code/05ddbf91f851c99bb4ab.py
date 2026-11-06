import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

def find_violation_fast(adj_masks, n, prev_viol=None):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None."""
    # Check size 6 first
    if prev_viol is not None:
        # Re-check only the previous violation first (it may still be violating)
        combo = prev_viol
        c = combo
        mask = (1 << c[0]) | (1 << c[1]) | (1 << c[2]) | (1 << c[3]) | (1 << c[4]) | (1 << c[5])
        d0 = (adj_masks[c[0]] & mask).bit_count()
        if ((adj_masks[c[1]] & mask).bit_count() == d0 and
            (adj_masks[c[2]] & mask).bit_count() == d0 and
            (adj_masks[c[3]] & mask).bit_count() == d0 and
            (adj_masks[c[4]] & mask).bit_count() == d0 and
            (adj_masks[c[5]] & mask).bit_count() == d0):
            return combo
    
    for combo in itertools.combinations(range(n), 6):
        c = combo
        mask = (1 << c[0]) | (1 << c[1]) | (1 << c[2]) | (1 << c[3]) | (1 << c[4]) | (1 << c[5])
        d0 = (adj_masks[c[0]] & mask).bit_count()
        if ((adj_masks[c[1]] & mask).bit_count() == d0 and
            (adj_masks[c[2]] & mask).bit_count() == d0 and
            (adj_masks[c[3]] & mask).bit_count() == d0 and
            (adj_masks[c[4]] & mask).bit_count() == d0 and
            (adj_masks[c[5]] & mask).bit_count() == d0):
            return combo

    # Check size 7
    for combo in itertools.combinations(range(n), 7):
        mask = 0
        for v in combo: mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok: return combo
        
    # Check size 8+ (less likely, but must check)
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo: mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok: return combo
            
    return None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def local_search(adj_masks, n, rng, deadline):
    cur = adj_masks[:]
    
    # Initial check
    viol = find_violation_fast(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    max_iters = 800
    it = 0
    
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation_fast(cur, n, prev_viol=viol)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = list(viol)
        internal_edges = []
        internal_nonedges = []
        for i in range(len(vset)):
            for j in range(i + 1, len(vset)):
                u, v = vset[i], vset[j]
                if cur[u] & (1 << v):
                    internal_edges.append((u, v))
                else:
                    internal_nonedges.append((u, v))
        
        # Try flipping edges/nonedges in the violating set
        # Prefer removing edges if there are many, adding if few
        if len(internal_edges) > len(internal_nonedges):
            # More edges than nonedges, try removing an edge
            candidates = internal_edges if internal_edges else internal_nonedges
        else:
            # More nonedges, try adding an edge
            candidates = internal_nonedges if internal_nonedges else internal_edges
        
        if not candidates:
            candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        u, v = rng.choice(candidates)
        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1
        
    viol = find_violation_fast(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(42)

    # Try n from 21 down to 6
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break

        if best_valid and best_n >= n:
            continue

        # More attempts for larger n
        if n >= 21:
            max_attempts = 80
        elif n >= 19:
            max_attempts = 50
        else:
            max_attempts = 30

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random graph with varying density
            p = rng.choice([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
            
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break

            # Local search
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

    if not best_valid or best_n < 6:
        return {'n': 6, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
