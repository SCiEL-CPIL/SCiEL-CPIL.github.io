import time
import random
import itertools
import sys

DEADLINE = 110 # Safety margin from 120s
START_TIME = None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def is_regular_subset(adj_masks, combo):
    """Check if the induced subgraph on 'combo' is regular."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    d0 = (adj_masks[combo[0]] & mask).bit_count()
    for v in combo[1:]:
        if (adj_masks[v] & mask).bit_count() != d0:
            return False
    return True

def find_one_6_violation(adj_masks, n, rng):
    """Find one 6-vertex subset that induces a regular subgraph. Returns None if none found after a few random checks or full scan."""
    # Try a few random samples first
    for _ in range(100):
        combo = tuple(rng.sample(range(n), 6))
        if is_regular_subset(adj_masks, combo):
            return combo
    
    # If random fails, do a full scan but break early
    for combo in itertools.combinations(range(n), 6):
        if is_regular_subset(adj_masks, combo):
            return combo
            
    return None

def check_full_validity(adj_masks, n):
    """Check if any subset of size >= 6 is regular. Returns (True, None) if valid, else (False, violating_combo)."""
    # Check size 6
    for combo in itertools.combinations(range(n), 6):
        if is_regular_subset(adj_masks, combo):
            return False, combo
    # Check size 7
    for combo in itertools.combinations(range(n), 7):
        if is_regular_subset(adj_masks, combo):
            return False, combo
    # Check size 8 and up
    # Note: This is expensive. Only call when necessary.
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            if is_regular_subset(adj_masks, combo):
                return False, combo
    return True, None

def local_search(adj_masks, n, rng, deadline):
    cur = adj_masks[:]
    
    # Initial check for 6-violations
    # We don't do full validity check initially to save time, just assume it's invalid if 6-violations exist
    
    max_iters = 5000
    it = 0
    stuck_counter = 0
    last_full_check_time = 0
    
    while it < max_iters:
        if time.time() > deadline:
            break

        # 1. Try to find a 6-vertex violation
        viol_combo = find_one_6_violation(cur, n, rng)
        
        if viol_combo is None:
            # No 6-vertex violations found. 
            # We need to check for larger violations.
            # To avoid timeout, only do full check if we haven't done it recently or if we are stuck.
            current_time = time.time()
            if current_time - last_full_check_time > 2.0 or stuck_counter > 50:
                valid, viol_large = check_full_validity(cur, n)
                last_full_check_time = current_time
                if valid:
                    return n, get_edges_from_masks(cur, n), True
                else:
                    viol_combo = viol_large
                    stuck_counter = 0
            else:
                # If we can't check now, make a random move to escape potential local minimum
                # or just break and hope for the best? 
                # Better: make a random move and continue
                u, v = rng.randint(0, n-1), rng.randint(0, n-1)
                if u == v:
                    it += 1
                    continue
                cur[u] ^= (1 << v)
                cur[v] ^= (1 << u)
                it += 1
                stuck_counter += 1
                continue
        else:
            stuck_counter = 0

        # 2. Flip an edge within the violating subset
        # Choose a random pair from the violation
        i, j = rng.sample(range(len(viol_combo)), 2)
        u, v = viol_combo[i], viol_combo[j]
        
        # Flip edge (u, v)
        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1

    # Final check
    valid, _ = check_full_validity(cur, n)
    if valid:
        return n, get_edges_from_masks(cur, n), True
        
    return n, get_edges_from_masks(cur, n), False

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(42)

    # Try n from 21 down to 6
    # We prioritize higher n
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break

        if best_valid and best_n >= n:
            continue

        # Limit attempts for large n to save time
        if n >= 21:
            max_attempts = 5
        elif n >= 19:
            max_attempts = 3
        else:
            max_attempts = 2

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random graph with varying density
            # Densities around 0.5 are common for such graphs
            p = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                # If we found a valid graph for this n, we can stop trying this n
                break

    if not best_valid or best_n < 6:
        # Fallback: Return a known small valid graph
        # P6 is non-regular
        return {'n': 6, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,5]]}

    return {'n': best_n, 'edges': best_edges}
