"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

DEADLINE = time.time() + 280  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict bitsets:
# conflict[p][a] is a bitmask of all points b such that (p, a, b) forms an isosceles triangle
# We use Python integers as bitsets for fast AND operations
conflict = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for p in range(NUM_PTS):
    for a in range(NUM_PTS):
        if p == a:
            continue
        mask = 0
        for b in range(NUM_PTS):
            if b == p or b == a:
                continue
            d_pa = dist[p][a]
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << b)
        conflict[p][a] = mask

def can_add(p, chosen_mask):
    """Check if point p can be added given chosen points represented as bitmask."""
    # For each a in chosen, check if any b in chosen is in conflict[p][a]
    # We iterate over set bits in chosen_mask
    cm = chosen_mask
    while cm:
        # Get lowest set bit
        lb = cm & (-cm)
        a = lb.bit_length() - 1
        # Check if any chosen point b conflicts
        if conflict[p][a] & chosen_mask:
            return False
        cm ^= lb
    return True

def greedy_construct(order, chosen_mask=0, chosen_list=None):
    """Greedy construction following given order of points."""
    if chosen_list is None:
        chosen_list = []
    for p in order:
        if not (chosen_mask >> p) & 1:  # p not already chosen
            if can_add(p, chosen_mask):
                chosen_list.append(p)
                chosen_mask |= (1 << p)
    return chosen_list, chosen_mask

def local_search(chosen_list, chosen_mask, max_iterations=200):
    """Local search: try removing a point and refilling greedily."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            # Remove point i
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try to add other points greedily (in random order for diversity)
            rng = random.Random(iteration * 1000 + i)
            candidates = [p for p in range(NUM_PTS) if not (new_mask >> p) & 1]
            rng.shuffle(candidates)
            
            for p in candidates:
                if can_add(p, new_mask):
                    new_list.append(p)
                    new_mask |= (1 << p)
            
            # If we got a smaller set, keep it
            if len(new_list) < best_len:
                best = new_list
                best_mask = new_mask
                best_len = len(best)
                improved = True
                break  # Restart local search with new best
    
    return best, best_mask

def randomized_greedy(seed):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    chosen, mask = greedy_construct(order)
    return chosen, mask

def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best = None
    best_mask = 0
    best_len = float('inf')
    
    # First, do a deterministic row-major greedy as baseline
    baseline_order = list(range(NUM_PTS))
    baseline, baseline_mask = greedy_construct(baseline_order)
    best = baseline
    best_mask = baseline_mask
    best_len = len(best)
    
    # Try randomized restarts
    seed = 42
    while time.time() < DEADLINE:
        # Randomized greedy
        chosen, mask = randomized_greedy(seed)
        if len(chosen) < best_len:
            best = chosen
            best_mask = mask
            best_len = len(best)
        
        # Local search improvement
        improved, improved_mask = local_search(chosen, mask, max_iterations=50)
        if len(improved) < best_len:
            best = improved
            best_mask = improved_mask
            best_len = len(best)
        
        seed += 1
    
    # Convert indices back to coordinates
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
