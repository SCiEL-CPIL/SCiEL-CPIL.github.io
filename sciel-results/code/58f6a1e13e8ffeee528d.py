"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _build_edges():
    d = 7
    n = 1 << d
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    return edges, d

def _build_face_conflicts(edges):
    # For each edge, find which other edges form a C4 with it.
    # Edge (u,v) with direction i. For each other direction j,
    # the C4 is: u, v, u^j, v^j. The other three edges are:
    #   (u, u^j), (v, v^j), (u^j, v^j)
    d = 7
    n = 1 << d
    # Map edge -> index
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    m = len(edges)
    # conflicts[i] = list of edge indices that conflict with edge i
    conflicts = [[] for _ in range(m)]
    for idx, (u, v) in enumerate(edges):
        # determine direction
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            # three other edges of the face
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            for ek in (e1, e2, e3):
                k = edge_idx[ek]
                if k != idx:
                    conflicts[idx].append(k)
    return conflicts

def _greedy_random(edges, conflicts, rng, time_budget):
    """Randomized greedy: shuffle edges, add if no conflict with already chosen."""
    d = 7
    m = len(edges)
    chosen = [False] * m
    count = 0
    order = list(range(m))
    rng.shuffle(order)
    start = time.time()
    for idx in order:
        if time.time() - start > time_budget:
            break
        # Check if adding idx conflicts with any already chosen
        ok = True
        for k in conflicts[idx]:
            if chosen[k]:
                ok = False
                break
        if ok:
            chosen[idx] = True
            count += 1
    return chosen, count

def _local_improve(edges, conflicts, chosen, count, rng, time_budget):
    """Try to add rejected edges by removing conflicting chosen edges if net gain."""
    d = 7
    m = len(edges)
    start = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        # Try to add each unchosen edge
        unchosen = [i for i in range(m) if not chosen[i]]
        rng.shuffle(unchosen)
        for idx in unchosen:
            if time.time() - start > time_budget:
                break
            # Find conflicting chosen edges
            conf = [k for k in conflicts[idx] if chosen[k]]
            if len(conf) < 1:
                # Can add directly
                chosen[idx] = True
                count += 1
                improved = True
            elif len(conf) == 1:
                # Swapping: remove 1, add 1 -> no net gain, skip
                pass
            elif len(conf) >= 2:
                # Try removing all conf and adding idx: net = 1 - len(conf) <= -1, skip
                pass
    return chosen, count

def _swap_improve(edges, conflicts, chosen, count, rng, time_budget):
    """Try swaps: remove a chosen edge, add two unchosen edges that were blocked by it."""
    d = 7
    m = len(edges)
    start = time.time()
    # For each chosen edge, find unchosen edges that conflict with it
    # If removing it allows adding 2+ edges, do the swap
    chosen_set = set(i for i in range(m) if chosen[i])
    # Build reverse: for each unchosen edge, which chosen edges block it
    # This is expensive; do a lighter version
    for idx in range(m):
        if time.time() - start > time_budget:
            break
        if chosen[idx]:
            continue
        conf = [k for k in conflicts[idx] if chosen[k]]
        if len(conf) == 2:
            # Try swapping: remove both conf, add idx
            # Net: -2 + 1 = -1, not helpful
            pass
    return chosen, count

def construct():
    deadline = time.time() + 110  # safety margin from 120s
    edges, d = _build_edges()
    conflicts = _build_face_conflicts(edges)
    m = len(edges)
    
    rng = random.Random(42)
    best_chosen = None
    best_count = 0
    
    # Multiple randomized greedy restarts
    while time.time() < deadline - 10:
        chosen, count = _greedy_random(edges, conflicts, rng, 5.0)
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
        
        # Local improvement
        if time.time() < deadline - 5:
            chosen, count = _local_improve(edges, conflicts, chosen, count, rng, 5.0)
            if count > best_count:
                best_count = count
                best_chosen = chosen[:]
    
    # Final: try to squeeze more with a longer local improvement pass
    if best_chosen is not None and time.time() < deadline - 2:
        chosen, count = _local_improve(edges, conflicts, best_chosen[:], best_count, rng, 3.0)
        if count > best_count:
            best_count = count
            best_chosen = chosen
    
    if best_chosen is None:
        # Fallback: simple greedy
        chosen, count = _greedy_random(edges, conflicts, random.Random(0), 1.0)
        best_chosen = chosen
        best_count = count
    
    out = []
    for i in range(m):
        if best_chosen[i]:
            out.append(list(edges[i]))
    return out
# EVOLVE-BLOCK-END
