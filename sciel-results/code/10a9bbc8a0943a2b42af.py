import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    N = 1000
    best_set = None
    best_size = 0
    
    # We'll use a greedy approach with randomization and local search
    # Key insight: the constraint is that every sum a+b (a<=b) has at most 2 representations
    # This is a B2[2] set (Sidon-like with multiplicity 2)
    
    # Precompute: for each pair (i,j) with i<=j, the sum is i+j
    # We need to track counts of each sum
    
    # Strategy: 
    # 1. Start with a greedy construction
    # 2. Use random restarts with different orderings
    # 3. Local search: try to swap elements to improve size
    
    def greedy_construction(order):
        """Greedy add elements in given order if constraint is satisfied."""
        chosen = []
        counts = {}  # sum -> count of representations
        for x in order:
            new_sums = []
            for y in chosen:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            # Check if adding x would violate the constraint
            valid = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        return chosen, counts
    
    def local_search(chosen, counts, max_iterations=5000):
        """Try to improve the set by swapping elements."""
        chosen = list(chosen)
        counts = dict(counts)
        n = len(chosen)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random unchosen element
            unchosen = [x for x in range(1, N+1) if x not in chosen]
            if not unchosen:
                break
            
            # Try random swaps: remove one, add one
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, n-1)
                x_remove = chosen[i]
                
                # Try to add a random unchosen element
                x_add = random.choice(unchosen)
                if x_add == x_remove:
                    continue
                
                # Temporarily remove x_remove
                # We need to recompute counts after removal
                # This is expensive, so let's be smart
                
                # Actually, let's try a different approach:
                # Try to add x_add first, see if it fits
                # If not, try removing some element that conflicts
                
                # Check if x_add can be added
                new_sums = [x_add + y for y in chosen] + [2 * x_add]
                trial_counts = dict(counts)
                valid = True
                for s in new_sums:
                    trial_counts[s] = trial_counts.get(s, 0) + 1
                    if trial_counts[s] > 2:
                        valid = False
                        break
                
                if valid:
                    # Add x_add
                    chosen.append(x_add)
                    for s in new_sums:
                        counts[s] = counts.get(s, 0) + 1
                    n += 1
                    improved = True
                    break
                else:
                    # Try to find an element to remove that would allow x_add
                    # This is more complex, skip for now
                    pass
            
            if not improved:
                # Try removing an element and adding two
                # Or just try random removals to see if we can add more
                if n > 0:
                    i = random.randint(0, n-1)
                    x_remove = chosen[i]
                    
                    # Remove x_remove
                    chosen.pop(i)
                    n -= 1
                    for y in chosen:
                        s = x_remove + y
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                    s = 2 * x_remove
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
                    
                    # Now try to add two random elements
                    unchosen = [x for x in range(1, N+1) if x not in chosen]
                    added = 0
                    for x_add in random.sample(unchosen, min(5, len(unchosen))):
                        new_sums = [x_add + y for y in chosen] + [2 * x_add]
                        trial_counts = dict(counts)
                        valid = True
                        for s in new_sums:
                            trial_counts[s] = trial_counts.get(s, 0) + 1
                            if trial_counts[s] > 2:
                                valid = False
                                break
                        if valid:
                            chosen.append(x_add)
                            for s in new_sums:
                                counts[s] = counts.get(s, 0) + 1
                            n += 1
                            added += 1
                        if added >= 1:
                            break
                    
                    if added == 0:
                        # Put x_remove back
                        chosen.append(x_remove)
                        n += 1
                        for y in chosen:
                            if y != x_remove:
                                s = x_remove + y
                                counts[s] = counts.get(s, 0) + 1
                        s = 2 * x_remove
                        counts[s] = counts.get(s, 0) + 1
        
        return chosen, counts
    
    # Main loop: try multiple random orderings
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        order = list(range(1, N+1))
        random.shuffle(order)
        
        # Greedy construction
        chosen, counts = greedy_construction(order)
        
        # Local search
        chosen, counts = local_search(chosen, counts)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
        
        # If we're doing well, we can be more aggressive
        if best_size > 500:
            # Try a few more local search iterations
            chosen, counts = local_search(chosen, counts, max_iterations=2000)
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = list(chosen)
    
    # Final: try to add any remaining elements
    if best_set is None:
        best_set = []
    
    # One more pass: try to add any element that fits
    counts = {}
    for a in best_set:
        for b in best_set:
            if a <= b:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
    
    for x in range(1, N+1):
        if x in best_set:
            continue
        new_sums = [x + y for y in best_set] + [2 * x]
        valid = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            best_set.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return best_set

# Seed the random generator for reproducibility
random.seed(42)
