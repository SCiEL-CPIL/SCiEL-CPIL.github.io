import random
import time
import itertools

N = 16
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict bitsets:
# conflict[p][a] = bitmask of all b such that (p, a, b) is isosceles
conflict = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for p in range(NUM_PTS):
    for a in range(NUM_PTS):
        if p == a:
            continue
        mask = 0
        d_pa = dist[p][a]
        for b in range(NUM_PTS):
            if b == p or b == a:
                continue
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << b)
        conflict[p][a] = mask


def can_add(p, chosen_mask):
    """Check if point p can be added given chosen points as bitmask."""
    cm = chosen_mask
    while cm:
        lb = cm & (-cm)
        a = lb.bit_length() - 1
        if conflict[p][a] & chosen_mask:
            return False
        cm ^= lb
    return True


def greedy_from_order(order, chosen_mask=0, chosen_list=None):
    """Greedy construction following given order."""
    if chosen_list is None:
        chosen_list = []
    for p in order:
        if not (chosen_mask >> p) & 1:
            if can_add(p, chosen_mask):
                chosen_list.append(p)
                chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def is_maximal(chosen_mask, chosen_list):
    """Check if set is maximal (no further point can be added)."""
    for p in range(NUM_PTS):
        if not (chosen_mask >> p) & 1:
            if can_add(p, chosen_mask):
                return False
    return True


def local_search_minimize(chosen_list, chosen_mask, deadline, max_iters=30):
    """Local search: try removing points and greedily refilling to minimize size."""
    best_list = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best_list)
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try removing each point
        for idx in range(len(best_list)):
            if time.time() > deadline - 5:
                break
            
            p_remove = best_list[idx]
            candidate_list = [x for i, x in enumerate(best_list) if i != idx]
            candidate_mask = best_mask & ~(1 << p_remove)
            
            # Greedily refill with random order to make maximal
            fill_order = list(range(NUM_PTS))
            random.shuffle(fill_order)
            
            new_list, new_mask = greedy_from_order(fill_order, candidate_mask, candidate_list)
            
            # Check if maximal
            if is_maximal(new_mask, new_list):
                if len(new_list) < best_len:
                    best_list = new_list
                    best_mask = new_mask
                    best_len = len(new_list)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_list, best_mask


def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin
    
    best_list = None
    best_mask = 0
    best_len = NUM_PTS + 1
    
    seed = 0
    rng = random.Random(seed)
    
    while time.time() < deadline - 10:
        # Generate random order
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        
        # Greedy construction
        chosen_list, chosen_mask = greedy_from_order(order)
        
        # Local search minimization
        improved_list, improved_mask = local_search_minimize(chosen_list, chosen_mask, deadline, max_iters=15)
        
        if len(improved_list) < best_len:
            best_list = improved_list
            best_mask = improved_mask
            best_len = len(improved_list)
        
        seed += 1
    
    # Final polish: try more aggressive local search on best found
    if best_list is not None and time.time() < deadline - 5:
        final_list, final_mask = local_search_minimize(best_list, best_mask, deadline, max_iters=50)
        if len(final_list) < best_len:
            best_list = final_list
            best_mask = final_mask
            best_len = len(final_list)
    
    if best_list is None:
        # Fallback: deterministic greedy
        order = list(range(NUM_PTS))
        best_list, best_mask = greedy_from_order(order)
    
    return [POINTS[i] for i in best_list]
