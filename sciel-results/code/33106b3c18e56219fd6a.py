"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Precompute for each element, the set of pairs that form a bad triple with it
    # bad_pairs_for[x] = set of (a, b) such that (x, a, b) is a bad triple
    bad_pairs_for = {i: set() for i in range(1, n + 1)}
    for (a, b, c) in bad_triples:
        bad_pairs_for[a].add((b, c))
        bad_pairs_for[b].add((a, c))
        bad_pairs_for[c].add((a, b))
    
    # Also precompute list of bad triples containing each element for faster removal updates?
    # For the greedy removal, we need to know how many *remaining* bad triples each element is in.
    # We can maintain a set of remaining bad triples.
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        # A bad triple is created if there exists (a, b) in bad_pairs_for[x]
        # such that a in current_set and b in current_set.
        for (a, b) in bad_pairs_for[x]:
            if a in current_set and b in current_set:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Number of restarts
    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Hitting Set
        in_set = [True] * (n + 1)
        current_set = set(range(1, n + 1))
        
        # We need to remove elements to hit all bad triples.
        # remaining_bad: set of bad triples not yet hit.
        remaining_bad = set(bad_triples)
        
        # To speed up finding the element with max count, we can just scan.
        # N=128, so scanning is cheap.
        
        while remaining_bad:
            if time.time() > deadline:
                break
            
            # Count occurrences of each element in remaining_bad
            counts = defaultdict(int)
            for t in remaining_bad:
                for x in t:
                    if in_set[x]:
                        counts[x] += 1
            
            if not counts:
                break
                
            # Find max count
            max_count = max(counts.values())
            # Get all elements with max count
            candidates = [x for x, c in counts.items() if c == max_count]
            
            # Randomly pick one
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            current_set.remove(x)
            
            # Remove all bad triples containing x from remaining_bad
            remaining_bad = {t for t in remaining_bad if x not in t}
        
        # current_set is now a valid set.
        
        # Local Search: Add back removed elements
        removed = [i for i in range(1, n + 1) if not in_set[i]]
        random.shuffle(removed)
        
        # Try to add back
        for x in removed:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.add(x)
                in_set[x] = True
                
        # Second pass of add-back? 
        # Since adding an element only restricts others, one pass is enough 
        # if we process in a fixed order. But shuffling helps find different outcomes.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # Early exit if we found a very large set? 
            # Max possible is 128. If we get close, maybe stop?
            # But let's run all restarts to be sure.
            
    # If no valid set found (shouldn't happen), return [1]
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
