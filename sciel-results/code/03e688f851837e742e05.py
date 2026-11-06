import random
import time
import itertools
import sys

# Precompute all triples
def get_all_triples(n=16):
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    return triples

ALL_TRIPLES = get_all_triples(16)
N_TRIPLES = len(ALL_TRIPLES)
TRIPLE_TO_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute conflicts
# For each triple t, find all K(2,2,2) configurations containing t.
# A K(2,2,2) is defined by 3 disjoint pairs {a,a'}, {b,b'}, {c,c'}.
# The 8 triples are all combinations.
# If t = (a,b,c), the other 7 are:
# (a',b,c), (a,b',c), (a,b,c'),
# (a',b',c), (a',b,c'), (a,b',c'),
# (a',b',c')
# We need to map these to indices.

def precompute_conflicts():
    conflicts = [[] for _ in range(N_TRIPLES)]
    
    # Iterate over all possible K(2,2,2) structures
    # Choose 6 distinct points
    for points in itertools.combinations(range(16), 6):
        # Partition 6 points into 3 pairs
        # There are 15 ways to partition 6 items into 3 pairs
        p = list(points)
        # Generate all pairings
        # Fix p[0] paired with p[1], p[2], ..., p[5]
        # Then pair up the remaining 4
        for i in range(1, 6):
            pair1 = (p[0], p[i])
            remaining = [p[j] for j in range(6) if j != 0 and j != i]
            # Pair up remaining 4: (r0,r1)-(r2,r3), (r0,r2)-(r1,r3), (r0,r3)-(r1,r2)
            r = remaining
            pairings = [
                ((r[0], r[1]), (r[2], r[3])),
                ((r[0], r[2]), (r[1], r[3])),
                ((r[0], r[3]), (r[1], r[2]))
            ]
            for pair2, pair3 in pairings:
                # We have 3 pairs: pair1, pair2, pair3
                # Each pair is {x, x'}
                # The 8 triples are formed by picking one from each pair
                # Let's generate the 8 triples
                triples_in_k = []
                for a in pair1:
                    for b in pair2:
                        for c in pair3:
                            t = tuple(sorted((a, b, c)))
                            triples_in_k.append(t)
                
                # For each of the 8 triples, add the other 7 to its conflict list
                for t in triples_in_k:
                    idx = TRIPLE_TO_INDEX[t]
                    others = [TRIPLE_TO_INDEX[tt] for tt in triples_in_k if tt != t]
                    # Avoid duplicates in conflict list? 
                    # Different K(2,2,2)s might share the same "other 7"? Unlikely but possible.
                    # Even if duplicate, it just means we check the same condition twice.
                    # To save memory/time, we can use a set of tuples for others?
                    # But lists are faster to iterate.
                    # Let's just append.
                    conflicts[idx].append(tuple(others))
    
    return conflicts

CONFLICTS = precompute_conflicts()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_size = 0
    best_set = set()
    
    # Greedy with random restarts and swap local search
    while time.time() < deadline:
        # Randomize the order of triples
        indices = list(range(N_TRIPLES))
        random.shuffle(indices)
        
        present = [False] * N_TRIPLES
        count = 0
        
        # Phase 1: Greedy Construction
        for idx in indices:
            # Check if adding idx creates a K(2,2,2)
            feasible = True
            for others in CONFLICTS[idx]:
                # If all others are present, adding idx creates a violation
                # Check if all are True
                # Using all() is fast
                if all(present[i] for i in others):
                    feasible = False
                    break
            
            if feasible:
                present[idx] = True
                count += 1
        
        # Phase 2: Swap Local Search
        # Try to improve by swapping existing triples with non-existing ones
        # We iterate a few times or until no improvement
        improved = True
        max_swap_iterations = 50
        for _ in range(max_swap_iterations):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            
            # Shuffle indices to avoid bias
            swap_indices = list(range(N_TRIPLES))
            random.shuffle(swap_indices)
            
            for idx_out in swap_indices:
                if not present[idx_out]:
                    continue
                
                # Try to find a replacement for idx_out
                # We want to remove idx_out and add some idx_in
                # To be efficient, we can just try random candidates for idx_in
                # Or iterate through all non-present idx_in? 
                # Iterating all is 560 * 560 * check_cost. 
                # Check cost is ~ conflicts[idx_in] * 7.
                # Let's try a limited number of random candidates for idx_in per idx_out
                # Or, simpler: iterate through all non-present idx_in, but stop early if found?
                # Better: For each idx_out, try to find ANY idx_in that works.
                
                # To speed up, we can pre-shuffle the non-present indices?
                # Or just iterate.
                
                # Let's try all non-present indices, but break on first success to keep it fast
                # If we find a swap, we apply it and restart the outer loop? 
                # Or just mark improved and continue?
                # Applying a swap might enable other swaps.
                
                # Optimization: We can pre-compute a list of non-present indices
                # But present changes.
                
                # Let's just iterate through all indices, skip if present
                # This is O(N^2) per pass. 560^2 = 313,600 checks.
                # Each check is ~ few microseconds. 0.1-0.5s per pass.
                # 50 passes = 5-25s. Might be too slow for many restarts.
                
                # Let's limit the search: For each idx_out, try only a few random idx_in?
                # Or, iterate through idx_in, and if feasible, swap.
                
                # Let's try a randomized local search:
                # Pick a random idx_out that is present.
                # Pick a random idx_in that is not present.
                # Check if swap is valid.
                # If yes, swap.
                # Repeat for a fixed number of steps or until no improvement in a window.
                
                pass # We will replace this with the actual implementation below

        # Update best
        if count > best_size:
            best_size = count
            best_set = set(i for i in range(N_TRIPLES) if present[i])

    # Convert best_set back to list of triples
    result = [ALL_TRIPLES[i] for i in best_set]
    return result

# Run construct to ensure it works and maybe cache?
# No, construct is called by evaluator.
