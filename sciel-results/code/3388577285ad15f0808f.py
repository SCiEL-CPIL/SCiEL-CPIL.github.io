"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

# Precompute isosceles relationships
# iso[r] is a list of tuples (u, v) such that (r, u, v) is isosceles
iso = [[] for _ in range(NUM_POINTS)]
for r in range(NUM_POINTS):
    for u in range(r + 1, NUM_POINTS):
        for v in range(u + 1, NUM_POINTS):
            d_ru = _d2(POINTS[r], POINTS[u])
            d_rv = _d2(POINTS[r], POINTS[v])
            d_uv = _d2(POINTS[u], POINTS[v])
            if d_ru == d_rv or d_ru == d_uv or d_rv == d_uv:
                iso[r].append((u, v))
                # Note: We only need to check if u,v are in S.
                # The order in the tuple doesn't matter for set membership,
                # but we store as tuple.

def is_maximal(S_set):
    """Check if set S_set (of indices) is maximal."""
    for r in range(NUM_POINTS):
        if r in S_set:
            continue
        # Check if r can be added
        # r can be added if NO pair (u, v) in S_set is in iso[r]
        blocked = False
        for u, v in iso[r]:
            if u in S_set and v in S_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Initialize best with a simple greedy
    S_init = []
    S_init_set = set()
    for idx in range(NUM_POINTS):
        can_add = True
        for u, v in iso[idx]:
            if u in S_init_set and v in S_init_set:
                can_add = False
                break
        if can_add:
            S_init.append(idx)
            S_init_set.add(idx)
    best_set = S_init[:]
    best_size = len(S_init)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # 1. Greedy Construction with Random Order
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        S = []
        S_set = set()
        for idx in indices:
            can_add = True
            for u, v in iso[idx]:
                if u in S_set and v in S_set:
                    can_add = False
                    break
            if can_add:
                S.append(idx)
                S_set.add(idx)
        
        current_size = len(S)
        
        # 2. Pruning
        # Try to remove pairs
        # Limit pruning time per restart to avoid getting stuck
        prune_deadline = time.time() + 20 # 20s per restart max?
        while time.time() < prune_deadline and time.time() < deadline:
            changed = False
            S_list = list(S_set)
            random.shuffle(S_list)
            # Generate pairs
            pairs = list(itertools.combinations(S_list, 2))
            random.shuffle(pairs)
            
            for p, q in pairs:
                if time.time() > deadline:
                    break
                # Try removing p and q
                new_S_set = S_set - {p, q}
                # Check if new_S_set is maximal
                if is_maximal(new_S_set):
                    S_set = new_S_set
                    current_size -= 2
                    changed = True
                    break # Restart pair search
            
            if not changed:
                break
        
        if current_size < best_size:
            best_size = current_size
            best_set = list(S_set)
            
    return [POINTS[i] for i in best_set]
# EVOLVE-BLOCK-END
