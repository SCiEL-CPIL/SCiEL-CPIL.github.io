import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
            
    # Find all bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # and a,b,c distinct.
    bad_triples = []
    for a, b, c in combinations(nums, 3):
        if gcd_mat[a][b] == gcd_mat[a][c] == gcd_mat[b][c]:
            bad_triples.append((a, b, c))
            
    # Heuristic: Remove elements involved in most bad triples
    # Use a priority queue or just sort each time
    # To be fast, we can just iterate.
    
    def solve():
        current_set = set(nums)
        # Count occurrences in bad triples
        # We need to recompute or maintain counts.
        # Since N=128, number of triples is ~ 350k.
        # Removing one element removes ~ O(N^2) triples? No, triples containing x.
        # Number of triples containing x is at most C(127, 2) ~ 8k.
        # So we can maintain a count.
        
        counts = {x: 0 for x in nums}
        # Initial counts
        for a, b, c in bad_triples:
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
            
        # We need to remove elements until no bad triples remain.
        # We will remove the element with max count.
        # But removing an element invalidates all triples containing it.
        # We need to update counts of the other elements in those triples.
        
        # This is expensive to do naively.
        # Alternative: Just recompute the set of bad triples in current_set each step?
        # C(128,3) is 341,376.
        # If we do this 100 times, it's 34M operations. Might be slow in Python.
        
        # Better: Keep a list of bad triples.
        # When we remove x, we mark all triples containing x as invalid.
        # And we decrement counts of the other elements in those triples.
        
        active_triples = [True] * len(bad_triples)
        # Map from number to list of indices of triples containing it
        num_to_triples = {x: [] for x in nums}
        for i, (a, b, c) in enumerate(bad_triples):
            num_to_triples[a].append(i)
            num_to_triples[b].append(i)
            num_to_triples[c].append(i)
            
        current_set = set(nums)
        
        # Greedy removal
        # While there is an active bad triple
        # Find max count element
        # Remove it
        # Update counts
        
        # To avoid infinite loops or slow execution, we can just run this once.
        # But greedy might get stuck in local minima.
        # We can run multiple times with random tie-breaking.
        
        best_size = 0
        best_set = set()
        
        for _ in range(20): # 20 restarts
            counts = {x: 0 for x in nums}
            for a, b, c in bad_triples:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
                
            current_set = set(nums)
            active_triples = [True] * len(bad_triples)
            
            # We need to know if there are any active triples.
            # We can just check if any count > 0? No, count > 0 means it's in some triple,
            # but that triple might be inactive if one of its other elements was removed.
            # So we must maintain active_triples.
            
            # Optimization: Instead of checking all triples, we can just try to remove
            # elements with high counts.
            # Stop when no active triples remain.
            
            # How to check if no active triples remain?
            # We can maintain a count of active triples.
            active_count = len(bad_triples)
            
            while active_count > 0:
                # Find element in current_set with max count
                # Note: counts might be stale if we don't update carefully.
                # We update counts when we remove an element.
                
                # To find max, we scan current_set.
                max_x = None
                max_c = -1
                for x in current_set:
                    if counts[x] > max_c:
                        max_c = counts[x]
                        max_x = x
                        
                if max_x is None or max_c == 0:
                    # No element in high conflict? But active_count > 0.
                    # This means we have a bug in count updating.
                    # Or we just pick a random one from an active triple.
                    # Let's find an active triple and pick one.
                    for i, t in enumerate(bad_triples):
                        if active_triples[i]:
                            a, b, c = t
                            # Pick one randomly
                            max_x = random.choice([a, b, c])
                            break
                    
                if max_x is None:
                    break
                    
                # Remove max_x
                current_set.remove(max_x)
                
                # Update counts for triples containing max_x
                for idx in num_to_triples[max_x]:
                    if active_triples[idx]:
                        active_triples[idx] = False
                        active_count -= 1
                        a, b, c = bad_triples[idx]
                        for y in (a, b, c):
                            if y != max_x and y in current_set:
                                counts[y] -= 1
                                
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        return list(best_set)

    result = solve()
    return result
