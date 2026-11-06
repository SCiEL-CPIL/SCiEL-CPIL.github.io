import time
import random
from math import gcd

def construct():
    start_time = time.time()
    # 85s limit to be safe, target 80s
    deadline = start_time + 80.0
    
    N = 128
    
    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute bad_pairs_for[x]
    # bad_pairs_for[x] is a set of tuples (a, b) with a < b
    # such that (a, b, x) is a bad triple
    bad_pairs_for = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                # Optimization: if lab is already determined, check c
                # We need lcm(a,c) == lab and lcm(b,c) == lab
                # Calculate lcm(a,c)
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    # Bad triple found
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))

    best_S = set()
    best_size = 0
    
    # Helper functions for set operations
    def can_add(x, s_pairs):
        return s_pairs.isdisjoint(bad_pairs_for[x])

    def add(x, s_pairs, s_set):
        s_set.add(x)
        for y in s_set:
            if x < y:
                s_pairs.add((x, y))
            else:
                s_pairs.add((y, x))

    def remove(x, s_pairs, s_set):
        s_set.remove(x)
        for y in s_set:
            if x < y:
                s_pairs.remove((x, y))
            else:
                s_pairs.remove((y, x))

    # Run multiple random restarts
    # Each restart: Greedy init, then Local Search
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        current_S = set()
        current_pairs = set()
        
        # Greedy Initialization
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, current_pairs):
                add(x, current_pairs, current_S)
                
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
            
        # Local Search: Remove 1, Greedy Refill
        # We do this until we hit time limit or get stuck for a while?
        # Just run as many as possible.
        
        # To avoid infinite loops if no improvement, we can just keep going.
        # The state space is large enough.
        
        while time.time() < deadline - 2.0:
            if not current_S:
                break
                
            # Pick a random element to remove
            # Using list() is O(N), which is fine.
            rem = random.choice(list(current_S))
            remove(rem, current_pairs, current_S)
            
            # Greedy Refill
            # We want to add elements that fit.
            # To maximize chance of finding a larger set, we should prioritize
            # elements that allow more additions? 
            # Simple random shuffle is a good heuristic.
            # We can also sort by "degree" (number of conflicts)?
            # Elements with fewer conflicts are "better" to add.
            # Let's stick to random shuffle for simplicity and speed.
            
            candidates = [x for x in range(1, N + 1) if x not in current_S]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, current_pairs):
                    add(x, current_pairs, current_S)
                    
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
                
            # Optional: If we didn't improve in last K steps, restart?
            # For now, just keep going. The random perturbation will eventually escape.

    return list(best_S)
