import random
import time
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, leaving 10s margin
    
    max_val = 4096
    n = max_val
    
    # Build conflict graph: two numbers conflict if they share a triple sum
    # For each pair (a,b), compute all triple sums involving both a and b
    # Actually, we need: two numbers x,y conflict if there exist triples
    # {x,x,x}, {x,x,y}, {x,y,y}, {x,y,z} that collide with sums from other numbers
    
    # Better approach: For each number x, it participates in triples:
    # {x,x,x} -> sum 3x
    # {x,x,y} -> sum 2x+y for each y
    # {x,y,y} -> sum x+2y for each y
    # {x,y,z} -> sum x+y+z for each y<z
    
    # Two numbers x and y conflict if any triple sum from x's involvement
    # equals any triple sum from y's involvement, where the triples are "different"
    
    # Simpler: use greedy with local search improvement
    # The current greedy approach is O(n^3) per trial which is too slow
    
    # Let's use a faster approach: precompute conflicts using bitsets
    
    # First, let's try a smarter greedy with local swap improvement
    best_set = []
    best_size = 0
    
    # Precompute for each candidate x, which other candidates conflict with it
    # Conflict: x and y conflict if there exist triples (allowing reps) from {x} and {y}
    # that produce the same sum, where the triples are not identical
    
    # Actually, let's just do multiple random greedy trials with local improvement
    
    def can_add_fast(x, current_set, sum_set):
        """Check if x can be added to current_set without creating duplicate triple sums."""
        # Generate all new triple sums involving x
        # {x,x,x}
        # {x,x,y} for y in current_set
        # {x,y,y} for y in current_set  
        # {x,y,z} for y<z in current_set
        
        new_sums = [3*x]
        for y in current_set:
            new_sums.append(2*x + y)
        for y in current_set:
            new_sums.append(x + 2*y)
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                new_sums.append(x + current_set[i] + current_set[j])
        
        # Check for internal duplicates
        if len(set(new_sums)) != len(new_sums):
            return False, None
            
        # Check against existing sums
        new_sums_set = set(new_sums)
        if not new_sums_set.isdisjoint(sum_set):
            return False, None
            
        return True, new_sums_set
    
    # Try multiple strategies
    strategies = [
        'ascending', 'descending', 'random', 'random_desc'
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        candidates = list(range(1, max_val + 1))
        
        if strategy == 'ascending':
            pass
        elif strategy == 'descending':
            candidates.reverse()
        elif strategy == 'random':
            random.shuffle(candidates)
        elif strategy == 'random_desc':
            random.shuffle(candidates)
            candidates.reverse()
        
        current_set = []
        sum_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sums = can_add_fast(x, current_set, sum_set)
            if ok:
                current_set.append(x)
                sum_set.update(new_sums)
        
        # Local improvement: try swapping elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current_set)):
                    if time.time() > deadline:
                        break
                    # Try removing current_set[j] and adding a different number
                    removed = current_set[j]
                    current_set[j] = None  # temporarily mark
                    
                    # Rebuild sum_set without removed
                    temp_set = [x for x in current_set if x is not None]
                    temp_sum_set = set()
                    # Recompute sums for temp_set
                    for a_idx in range(len(temp_set)):
                        a = temp_set[a_idx]
                        temp_sum_set.add(3*a)
                        for b_idx in range(a_idx+1, len(temp_set)):
                            b = temp_set[b_idx]
                            temp_sum_set.add(2*a + b)
                            temp_sum_set.add(a + 2*b)
                            for c_idx in range(b_idx+1, len(temp_set)):
                                c = temp_set[c_idx]
                                temp_sum_set.add(a + b + c)
                    
                    # Try to add a better number
                    for x in range(1, max_val + 1):
                        if x == removed or x in temp_set:
                            continue
                        ok, new_sums = can_add_fast(x, temp_set, temp_sum_set)
                        if ok:
                            # This swap is valid, but we need it to be an improvement
                            # Since we're just maintaining size, any valid swap works
                            # But we want to potentially enable future additions
                            current_set[j] = x
                            temp_set[j] = x
                            temp_sum_set.update(new_sums)
                            improved = True
                            break
                    
                    if not improved:
                        current_set[j] = removed
                        # Rebuild sum_set
                        sum_set = set()
                        for a_idx in range(len(current_set)):
                            a = current_set[a_idx]
                            sum_set.add(3*a)
                            for b_idx in range(a_idx+1, len(current_set)):
                                b = current_set[b_idx]
                                sum_set.add(2*a + b)
                                sum_set.add(a + 2*b)
                                for c_idx in range(b_idx+1, len(current_set)):
                                    c = current_set[c_idx]
                                    sum_set.add(a + b + c)
                    
                    if improved:
                        break
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 42:
                break
    
    return best_set
