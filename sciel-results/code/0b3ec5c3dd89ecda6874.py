"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

DEADLINE = time.time() + 290  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = np.zeros((NUM_PTS, NUM_PTS), dtype=np.int32)
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i, j] = d
        dist[j, i] = d

# Precompute conflict bitsets using numpy for speed
# conflict[p][a] is a bitmask of all points b such that (p, a, b) forms an isosceles triangle
# We'll store as list of Python ints for efficient bitwise ops
conflict = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for p in range(NUM_PTS):
    for a in range(NUM_PTS):
        if p == a:
            continue
        mask = 0
        d_pa = dist[p, a]
        for b in range(NUM_PTS):
            if b == p or b == a:
                continue
            d_pb = dist[p, b]
            d_ab = dist[a, b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << b)
        conflict[p][a] = mask

# Precompute "blocker count" for each point: how many other points are in its conflict with some reference
# This helps prioritize during completion - points with fewer conflicts first
conflict_counts = []
for p in range(NUM_PTS):
    total = 0
    for a in range(NUM_PTS):
        if p != a:
            total += bin(conflict[p][a]).count('1')
    conflict_counts.append(total)


def can_add_fast(p, chosen_indices):
    """Check if point p can be added given chosen points as a list of indices."""
    for a in chosen_indices:
        if conflict[p][a] and any((conflict[p][a] >> b) & 1 for b in chosen_indices if b != p and b != a):
            return False
    return True


def can_add_mask(p, chosen_mask):
    """Check if point p can be added given chosen points represented as bitmask."""
    cm = chosen_mask
    while cm:
        lb = cm & (-cm)
        a = lb.bit_length() - 1
        if conflict[p][a] & chosen_mask:
            return False
        cm ^= lb
    return True


def complete_to_maximal(chosen_list, chosen_mask, rng, order_mode='random'):
    """Add points until the set is maximal (no more can be added)."""
    candidates = [p for p in range(NUM_PTS) if not (chosen_mask >> p) & 1]
    
    if order_mode == 'conflict_first':
        # Sort by conflict count ascending (fewer conflicts first = less likely to force more additions)
        candidates.sort(key=lambda p: conflict_counts[p])
    else:
        rng.shuffle(candidates)
    
    for p in candidates:
        if can_add_mask(p, chosen_mask):
            chosen_list.append(p)
            chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def greedy_construct(order, chosen_mask=0, chosen_list=None):
    """Greedy construction following given order of points."""
    if chosen_list is None:
        chosen_list = []
    for p in order:
        if not (chosen_mask >> p) & 1:
            if can_add_mask(p, chosen_mask):
                chosen_list.append(p)
                chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def randomized_greedy(seed, use_conflict_order=False):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    if use_conflict_order:
        order.sort(key=lambda p: (conflict_counts[p], rng.random()))
    else:
        rng.shuffle(order)
    chosen, mask = greedy_construct(order)
    # Ensure maximality
    chosen, mask = complete_to_maximal(chosen, mask, rng, order_mode='random')
    return chosen, mask


def local_search(chosen_list, chosen_mask, max_iterations=50):
    """Local search: try removing a point and refilling to maximality."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point in the current best set
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try multiple completion strategies
            best_new_list = None
            best_new_mask = 0
            best_new_len = float('inf')
            
            for attempt in range(3):
                rng = random.Random(iteration * 10000 + i * 7 + attempt * 13 + 42)
                mode = 'random' if attempt < 2 else 'conflict_first'
                nl, nm = complete_to_maximal(list(new_list), new_mask, rng, order_mode=mode)
                if len(nl) < best_new_len:
                    best_new_list = nl
                    best_new_mask = nm
                    best_new_len = len(nl)
            
            if best_new_list is not None and best_new_len < best_len:
                best = best_new_list
                best_mask = best_new_mask
                best_len = best_new_len
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best, best_mask


def shrink_search(chosen_list, chosen_mask):
    """Try to shrink the set by removing points one at a time and re-completing."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    changed = True
    while changed and time.time() < DEADLINE - 3:
        changed = False
        for i in range(len(best)):
            if time.time() > DEADLINE - 3:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try multiple random completions to find the smallest maximal superset
            best_completion = None
            best_completion_mask = 0
            best_completion_len = float('inf')
            
            for attempt in range(4):
                rng = random.Random(attempt * 1000 + i + 12345)
                mode = 'random' if attempt < 3 else 'conflict_first'
                cl, cm = complete_to_maximal(list(new_list), new_mask, rng, order_mode=mode)
                if len(cl) < best_completion_len:
                    best_completion = cl
                    best_completion_mask = cm
                    best_completion_len = len(cl)
            
            if best_completion is not None and best_completion_len < best_len:
                best = best_completion
                best_mask = best_completion_mask
                best_len = best_completion_len
                changed = True
                break
    
    return best, best_mask


def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best = None
    best_mask = 0
    best_len = float('inf')
    
    # Baseline: deterministic row-major greedy
    baseline_order = list(range(NUM_PTS))
    baseline, baseline_mask = greedy_construct(baseline_order)
    baseline, baseline_mask = complete_to_maximal(baseline, baseline_mask, random.Random(0))
    best = baseline
    best_mask = baseline_mask
    best_len = len(best)
    
    # Also try conflict-ordered baseline
    conflict_order = sorted(range(NUM_PTS), key=lambda p: conflict_counts[p])
    c_baseline, c_baseline_mask = greedy_construct(conflict_order)
    c_baseline, c_baseline_mask = complete_to_maximal(c_baseline, c_baseline_mask, random.Random(1), order_mode='conflict_first')
    if len(c_baseline) < best_len:
        best = c_baseline
        best_mask = c_baseline_mask
        best_len = len(best)
    
    # Randomized restarts with local search
    seed = 42
    while time.time() < DEADLINE - 10:
        use_conflict = (seed % 3 == 0)  # Every 3rd restart uses conflict ordering
        chosen, mask = randomized_greedy(seed, use_conflict_order=use_conflict)
        if len(chosen) < best_len:
            best = chosen
            best_mask = mask
            best_len = len(best)
        
        # Local search on this solution
        if len(chosen) <= best_len:
            improved, improved_mask = local_search(chosen, mask, max_iterations=20)
            if len(improved) < best_len:
                best = improved
                best_mask = improved_mask
                best_len = len(best)
        
        seed += 1
    
    # Final polish: shrink search on the best solution found
    if time.time() < DEADLINE - 3:
        polished, polished_mask = shrink_search(best, best_mask)
        if len(polished) < best_len:
            best = polished
            best_mask = polished_mask
            best_len = len(best)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
