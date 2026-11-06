"""Improved program: Local search for max edges on triangular lattice."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def _swap_delta(Sset, p_out, p_in):
    """Compute change in edge count when removing p_out and adding p_in."""
    # Edges lost: neighbors of p_out that are in Sset
    lost = 0
    for q in _neighbors(p_out):
        if q in Sset and q != p_in:
            lost += 1
    # Edges gained: neighbors of p_in that are in Sset (excluding p_out which is being removed)
    gained = 0
    for q in _neighbors(p_in):
        if q in Sset and q != p_out:
            gained += 1
    return gained - lost

def _two_swap_delta(Sset, p_out1, p_out2, p_in1, p_in2):
    """Compute change in edge count for a 2-swap move."""
    # Build the set without the two removed points
    S_without = Sset - {p_out1, p_out2}
    
    # Edges lost: edges involving p_out1 or p_out2
    lost = 0
    for p in (p_out1, p_out2):
        for q in _neighbors(p):
            if q in S_without:
                lost += 1
    # Edge between p_out1 and p_out2 (if they are neighbors)
    if p_out2 in _neighbors(p_out1):
        lost += 1
    
    # Edges gained: edges involving p_in1 or p_in2 with S_without
    gained = 0
    for p in (p_in1, p_in2):
        for q in _neighbors(p):
            if q in S_without:
                gained += 1
    # Edge between p_in1 and p_in2 (if they are neighbors)
    if p_in2 in _neighbors(p_in1):
        gained += 1
    
    return gained - lost

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Generate layers
    l0 = [(0, 0)]
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    
    for a in range(-5, 6):
        for b in range(-5, 6):
            d = (abs(a) + abs(a + b) + abs(b)) // 2
            if d == 1:
                l1.append((a, b))
            elif d == 2:
                l2.append((a, b))
            elif d == 3:
                l3.append((a, b))
            elif d == 4:
                l4.append((a, b))
            elif d == 5:
                l5.append((a, b))
    
    # Start with hexagon radius 2 (19 points)
    base_set = set(l0 + l1 + l2)
    
    # Greedily add 13 points from L3, L4, L5 to maximize edges
    current_set = set(base_set)
    candidates = list(l3) + list(l4) + list(l5)
    
    for _ in range(13):
        best_p = None
        best_score = -1
        for p in candidates:
            score = sum(1 for q in _neighbors(p) if q in current_set)
            if score > best_score:
                best_score = score
                best_p = p
        if best_p:
            current_set.add(best_p)
            candidates.remove(best_p)
    
    best_set = set(current_set)
    best_score = _edge_count(best_set)
    
    # Build swap pool: all points in radius 5 not in best_set
    pool = set()
    for a in range(-5, 6):
        for b in range(-5, 6):
            d = (abs(a) + abs(a + b) + abs(b)) // 2
            if d <= 5:
                pool.add((a, b))
    pool.difference_update(best_set)
    
    # Phase 1: Single swap local search with incremental delta
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        # Randomize order
        items = list(best_set)
        random.shuffle(items)
        for p_out in items:
            if time.time() > deadline - 30:
                break
            # Find best p_in to swap with
            best_delta = 0
            best_p_in = None
            # Sample a subset of pool for speed
            pool_list = list(pool)
            random.shuffle(pool_list)
            for p_in in pool_list:
                delta = _swap_delta(best_set, p_out, p_in)
                if delta > best_delta:
                    best_delta = delta
                    best_p_in = p_in
            if best_p_in is not None:
                best_set.remove(p_out)
                best_set.add(best_p_in)
                pool.remove(best_p_in)
                pool.add(p_out)
                best_score += best_delta
                improved = True
    
    # Phase 2: 2-swap moves to escape local optima
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        items = list(best_set)
        random.shuffle(items)
        for i in range(len(items)):
            if time.time() > deadline - 10:
                break
            p_out1 = items[i]
            for j in range(i + 1, len(items)):
                p_out2 = items[j]
                # Sample a few 2-swap candidates
                pool_list = list(pool)
                random.shuffle(pool_list)
                for p_in1 in pool_list[:20]:
                    if time.time() > deadline - 10:
                        break
                    for p_in2 in pool_list[:20]:
                        if p_in2 == p_in1:
                            continue
                        delta = _two_swap_delta(best_set, p_out1, p_out2, p_in1, p_in2)
                        if delta > 0:
                            best_set.remove(p_out1)
                            best_set.remove(p_out2)
                            best_set.add(p_in1)
                            best_set.add(p_in2)
                            pool.remove(p_in1)
                            pool.remove(p_in2)
                            pool.add(p_out1)
                            pool.add(p_out2)
                            best_score += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Phase 3: Final single-swap polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        items = list(best_set)
        random.shuffle(items)
        for p_out in items:
            if time.time() > deadline:
                break
            best_delta = 0
            best_p_in = None
            pool_list = list(pool)
            random.shuffle(pool_list)
            for p_in in pool_list:
                delta = _swap_delta(best_set, p_out, p_in)
                if delta > best_delta:
                    best_delta = delta
                    best_p_in = p_in
            if best_p_in is not None:
                best_set.remove(p_out)
                best_set.add(best_p_in)
                pool.remove(best_p_in)
                pool.add(p_out)
                best_score += best_delta
                improved = True
    
    # Verify and return
    final_score = _edge_count(best_set)
    return [[a, b] for a, b in best_set]
# EVOLVE-BLOCK-END
