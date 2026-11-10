import random
from itertools import combinations
from math import lcm

# Precompute bad triples and conflict structures
# conflicts_pair[u][v] is a set of w such that {u, v, w} is a bad triple
# We only need u < v to save space, but sets are unordered.
conflicts_pair = {}
for u in range(1, 129):
    conflicts_pair[u] = {}
    for v in range(u + 1, 129):
        conflicts_pair[u][v] = set()
        conflicts_pair[v][u] = set()

# List of all bad triples for quick lookup or degree calculation
bad_triples = []
for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            l_ab = lcm(a, b)
            l_ac = lcm(a, c)
            l_bc = lcm(b, c)
            if l_ab == l_ac == l_bc:
                bad_triples.append((a, b, c))
                conflicts_pair[a][b].add(c)
                conflicts_pair[a][c].add(b)
                conflicts_pair[b][a].add(c)
                conflicts_pair[b][c].add(a)
                conflicts_pair[c][a].add(b)
                conflicts_pair[c][b].add(a)

# Calculate conflict degree for each element
# degree[x] = number of bad triples containing x
degree = [0] * 129
for a, b, c in bad_triples:
    degree[a] += 1
    degree[b] += 1
    degree[c] += 1

def construct():
    best_set = set()
    
    # Run multiple iterations with different randomization strategies
    for _ in range(50):
        S = set()
        blocked = set() # Elements that cannot be added to S due to conflicts with pairs in S
        
        # Create a list of elements 1..128
        elements = list(range(1, 129))
        
        # Strategy: Sort by degree, but add randomness to break ties and diversify
        # We want to pick "safe" elements (low degree) first? 
        # Or "central" elements? 
        # Usually, picking elements that conflict with fewer others is safer for building a large set.
        # Let's sort by degree ascending, with random jitter.
        
        # Randomized sort key
        def sort_key(x):
            return degree[x] + random.uniform(0, 5)
            
        elements.sort(key=sort_key)
        
        for x in elements:
            if x in S:
                continue
            if x in blocked:
                continue
            
            # Check if x can be added
            # If x is not in blocked, it means no pair in S conflicts with x.
            # We can add x.
            S.add(x)
            
            # Update blocked set
            # For every y in S (y != x), x and y form a pair.
            # Any z in conflicts_pair[x][y] is now blocked.
            for y in S:
                if y == x:
                    continue
                # Add all z that conflict with pair (x, y)
                # Note: conflicts_pair[x][y] contains z such that {x,y,z} is bad.
                blocked.update(conflicts_pair[x][y])
        
        # Local Search: Try to improve
        # Simple hill climbing: try to swap elements to potentially unblock others?
        # Since the greedy step is deterministic given the order, and we randomized the order,
        # multiple runs should cover different local optima.
        
        # One more pass: try to add any element that might have been blocked incorrectly?
        # No, blocked is accurate for the current S.
        
        # Try single swaps to see if we can increase size? No, swap keeps size same.
        # The goal is to maximize size. Greedy adds if possible.
        # If we are stuck, we might need to remove an element to add two.
        # Let's try a "remove and re-add" heuristic for local improvement.
        
        improved = True
        while improved:
            improved = False
            # Try removing each element in S and see if we can add more
            # This is expensive. Let's just try random swaps that maintain validity and hope for the best?
            # Actually, a simpler improvement:
            # Check if any element outside S can be added by removing ONE element from S.
            # If so, do it if it allows adding MORE than 1 element? 
            # Or just try to find a valid set of size |S|+1.
            
            # Let's try to add one element from complement by removing one from S.
            # If we remove x and add y, size is same.
            # If we remove x and add y, z, size increases by 1.
            
            # Heuristic: Pick a random x from S. Remove it.
            # Recalculate blocked for S \ {x}.
            # Try to add as many elements as possible from complement greedily.
            # If new_size > old_size, accept.
            
            if len(S) >= 128: break
            
            # Sample a few removals
            for _ in range(10):
                if not S: break
                x_to_remove = random.choice(list(S))
                
                S_trial = S.copy()
                S_trial.remove(x_to_remove)
                
                # Recalculate blocked for S_trial
                # This is O(|S|^2). |S| is ~60-100. 100*100*10 = 100k ops. Fast.
                blocked_trial = set()
                for u in S_trial:
                    for v in S_trial:
                        if u < v:
                            blocked_trial.update(conflicts_pair[u][v])
                
                # Try to add elements from complement
                added = 0
                candidates = [z for z in range(1, 129) if z not in S_trial]
                random.shuffle(candidates)
                
                for z in candidates:
                    if z in blocked_trial:
                        continue
                    # Add z
                    S_trial.add(z)
                    added += 1
                    # Update blocked_trial
                    for y in S_trial:
                        if y == z: continue
                        blocked_trial.update(conflicts_pair[z][y])
                    
                    if len(S_trial) > len(S):
                        # We found a better set
                        S = S_trial
                        improved = True
                        break
                
                if improved:
                    break
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
    return list(best_set)
