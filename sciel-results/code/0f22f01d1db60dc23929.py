"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55
    
    TARGET = 200
    # Focus on the upper half where max subset size for sum 200 is small
    # Elements in [51, 100] can form sums of 200 with at most 3 elements.
    # Elements < 51 can form sums with 4+ elements, which is harder to manage.
    # We start with [51, 100] and find a minimum hitting set for bad triples.
    
    UNIVERSE = list(range(51, 101))
    
    # Generate all bad triples (a, b, c) with a < b < c, a+b+c = 200
    bad_triples = []
    for i in range(len(UNIVERSE)):
        a = UNIVERSE[i]
        for j in range(i + 1, len(UNIVERSE)):
            b = UNIVERSE[j]
            rem = TARGET - a - b
            # rem must be > b and <= 100
            if rem <= b:
                continue
            if rem > 100:
                continue
            # Check if rem is in UNIVERSE
            # Since UNIVERSE is sorted, we can check index
            # rem is in [51, 100]
            if rem in UNIVERSE:
                bad_triples.append((a, b, rem))
                
    # If no bad triples, return all
    if not bad_triples:
        return UNIVERSE

    # Greedy Hitting Set with randomization
    def greedy_hitting_set(triples, rng):
        H = set()
        active = list(triples)
        
        while active:
            # Count frequency of each element in active triples
            counts = {}
            for t in active:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            
            # Find max count
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            
            # Pick one randomly to break symmetry
            best_elem = rng.choice(candidates)
            H.add(best_elem)
            
            # Remove all triples containing best_elem
            active = [t for t in active if best_elem not in t]
            
            # Time check
            if time.time() > deadline - 2:
                break
                
        return H

    def is_valid_hitting_set(H, triples):
        for t in triples:
            if not any(v in H for v in t):
                return False
        return True

    best_H = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple greedy iterations with different random seeds/orders
    num_trials = 200
    for i in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Slightly perturb the frequency calculation or just rely on random choice in ties
        H = greedy_hitting_set(bad_triples, rng)
        
        # Local Search: Try to remove elements from H
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for h in list(H):
                # Try removing h
                H_test = H - {h}
                if is_valid_hitting_set(H_test, bad_triples):
                    H = H_test
                    improved = True
                    break # Restart the while loop as H changed
        
        if len(H) < best_size:
            best_size = len(H)
            best_H = H.copy()
            
    # If we found a good H, construct the result
    if best_H is not None:
        result = [x for x in UNIVERSE if x not in best_H]
    else:
        # Fallback to simple greedy descending if something went wrong
        result = []
        reachable = {0}
        for x in range(100, 0, -1):
            if 200 - x in reachable:
                continue
            result.append(x)
            new_sums = {s + x for s in reachable if s + x < 200}
            reachable |= new_sums
            if time.time() > deadline:
                break

    return result
# EVOLVE-BLOCK-END
