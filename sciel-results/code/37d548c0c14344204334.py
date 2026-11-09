import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    
    N = 4096
    MAX_K = 80
    
    # Precompute some random orders to try different greedy starting points
    # The goal is to find a set of size K such that all triple sums are unique.
    
    # Helper to check if a set of numbers has unique triple sums
    # We will use a fast incremental approach for the main search.
    
    def get_conflicts(vals):
        """
        Returns a list of (i, j) pairs of indices in vals where adding vals[i] 
        to the set formed by vals without i would cause a conflict? 
        Actually, for Simulated Annealing, we need a measure of "invalidity".
        Let's define invalidity as the number of duplicate triple sums.
        """
        sums = np.zeros((len(vals), len(vals), len(vals)), dtype=np.int64)
        # This is O(K^3) memory and time, which for K=80 is 512k entries. 
        # 512k * 8 bytes = 4MB. This is fine.
        # But we can do better with a set.
        
        pass

    def count_conflicts(vals):
        """
        Counts the number of duplicate triple sums.
        A triple sum is defined by i <= j <= k.
        """
        sums = set()
        conflicts = 0
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i, n):
                vj = vals[j]
                for k in range(j, n):
                    vk = vals[k]
                    s = vi + vj + vk
                    if s in sums:
                        conflicts += 1
                    else:
                        sums.add(s)
        return conflicts

    # Simulated Annealing Strategy
    # State: a list of integers
    # Move: remove one integer, add another random integer (or swap)
    # Cost: number of conflicting triple sums (we want 0)
    
    best_val = 0
    best_set = []
    
    # Initial Greedy to get a good starting point
    # The seed program found ~30 elements quickly.
    # Let's run a fast greedy first to establish a baseline.
    
    def fast_greedy(order):
        selected = []
        existing_sums = set()
        for x in order:
            # Check if x can be added
            # New sums involving x:
            # 3x
            # 2x + y
            # x + y + z
            conflict = False
            new_sums_to_add = []
            
            k = len(selected)
            
            # 3x
            s3x = 3 * x
            if s3x in existing_sums:
                conflict = True
            else:
                new_sums_to_add.append(s3x)
            
            if not conflict:
                # 2x + y
                for y in selected:
                    s2xy = 2 * x + y
                    if s2xy in existing_sums:
                        conflict = True
                        break
                    new_sums_to_add.append(s2xy)
            
            if not conflict:
                # x + y + z
                for i in range(k):
                    y = selected[i]
                    for j in range(i, k):
                        z = selected[j]
                        sx_yz = x + y + z
                        if sx_yz in existing_sums:
                            conflict = True
                            break
                        new_sums_to_add.append(sx_yz)
                    if conflict:
                        break
            
            if not conflict:
                selected.append(x)
                existing_sums.update(new_sums_to_add)
                
        return selected

    # Run a few greedy attempts with random orders to find a good seed
    best_greedy_size = 0
    best_greedy_set = []
    
    for _ in range(20):
        if time.time() > DEADLINE - 10: break
        random_order = list(range(1, N+1))
        random.shuffle(random_order)
        # Bias towards smaller numbers? The seed program used 1..513.
        # Smaller numbers are denser, so conflicts are more likely.
        # Larger numbers are sparser.
        # Let's try a mix.
        
        # Actually, for B_3[1], it's known that powers of 3 work.
        # 3^0, 3^1, ... 3^12 = 531441 > 4096.
        # So {1, 3, 9, 27, 81, 243, 729, 2187} is a valid set of size 8.
        # We want to go beyond this.
        
        # Greedy with random shuffle
        g_set = fast_greedy(random_order)
        if len(g_set) > best_greedy_size:
            best_greedy_size = len(g_set)
            best_greedy_set = g_set
            
    # If we found a good greedy set, use it as start for SA.
    # If not, start with empty or small set.
    
    current_set = best_greedy_set[:]
    if not current_set:
        current_set = [1, 3, 9]
        
    current_conflicts = count_conflicts(current_set)
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute all possible moves? No, too many.
    # Move: pick a random index to remove, pick a random number to add.
    
    while time.time() < DEADLINE - 2.0 and T > min_T:
        # Generate a neighbor
        new_set = current_set[:]
        
        # Randomly remove an element
        if new_set:
            remove_idx = random.randint(0, len(new_set) - 1)
            removed_val = new_set.pop(remove_idx)
            
            # Randomly add an element
            # We should try to add elements that might reduce conflicts.
            # But random is okay for exploration.
            # Let's pick a random candidate from 1..N
            # Or pick from a pool of "likely" candidates.
            
            # To be efficient, we can just pick a random int.
            # But if we pick a duplicate, it's invalid immediately.
            # So we must ensure uniqueness of values in the set.
            
            # Let's try a few random candidates
            added = False
            for _ in range(10):
                candidate = random.randint(1, N)
                if candidate not in new_set:
                    new_set.append(candidate)
                    added = True
                    break
            
            if not added:
                # Restore
                new_set.insert(remove_idx, removed_val)
                continue
            
        else:
            # Empty set, add one
            new_set.append(random.randint(1, N))
            
        new_conflicts = count_conflicts(new_set)
        
        # Acceptance criterion
        delta = new_conflicts - current_conflicts
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_set = new_set
            current_conflicts = new_conflicts
            
            if current_conflicts == 0:
                # We found a valid set!
                if len(current_set) > best_val:
                    best_val = len(current_set)
                    best_set = current_set[:]
                
                # If we reached max size, we can stop or try to improve?
                # The task is to maximize cardinality.
                # If we have a valid set of size K, can we add more?
                # If K < MAX_K, we should try to extend.
                # But SA might get stuck.
                # If we have a valid set, we can try a greedy extension.
                
                # Let's try greedy extension on the valid set
                extended_set = list(current_set)
                existing_sums = set()
                # Rebuild sums for the valid set
                n_ext = len(extended_set)
                for i in range(n_ext):
                    for j in range(i, n_ext):
                        for k in range(j, n_ext):
                            existing_sums.add(extended_set[i] + extended_set[j] + extended_set[k])
                
                # Try to add more
                # We need to try candidates.
                # Which candidates?
                # We can iterate through 1..N and check.
                # But that's slow.
                # Let's just let SA continue, it might find a larger valid set.
                # Or we can force it to try to increase size.
                
                # If we are at a local minimum of conflicts=0, 
                # SA might wander.
                # If we find a valid set of size K, and K < MAX_K,
                # we should prioritize finding a valid set of size K+1.
                
                # Let's modify the objective function:
                # Primary: Minimize conflicts
                # Secondary: Maximize size
                # Cost = conflicts * 1000 - size
                # This way, reducing conflicts is more important, 
                # but among zero-conflict states, larger size is better.
                
                # However, count_conflicts is expensive.
                # Let's stick to minimizing conflicts first.
                # Once we have a valid set, we can run a separate phase to extend it.
                
                pass
            
        T *= cooling
        
    # Final check: if we have a valid set, try to extend it greedily
    if current_conflicts == 0:
        # Rebuild sums
        existing_sums = set()
        n_cur = len(current_set)
        for i in range(n_cur):
            for j in range(i, n_cur):
                for k in range(j, n_cur):
                    existing_sums.add(current_set[i] + current_set[j] + current_set[k])
        
        # Try to add more elements
        # Iterate through candidates in random order?
        # Or sorted order?
        # Sorted order (small to large) is usually good for packing.
        
        # Let's try to add elements from 1..N that are not in the set
        candidates = [x for x in range(1, N+1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > DEADLINE - 1.0:
                break
            if len(current_set) >= MAX_K:
                break
            
            # Check if x can be added
            conflict = False
            new_sums = []
            
            k = len(current_set)
            
            s3x = 3 * x
            if s3x in existing_sums:
                conflict = True
            else:
                new_sums.append(s3x)
            
            if not conflict:
                for y in current_set:
                    s2xy = 2 * x + y
                    if s2xy in existing_sums:
                        conflict = True
                        break
                    new_sums.append(s2xy)
            
            if not conflict:
                for i in range(k):
                    y = current_set[i]
                    for j in range(i, k):
                        z = current_set[j]
                        sx_yz = x + y + z
                        if sx_yz in existing_sums:
                            conflict = True
                            break
                        new_sums.append(sx_yz)
                    if conflict:
                        break
            
            if not conflict:
                current_set.append(x)
                existing_sums.update(new_sums)
                current_conflicts = 0 # Still valid
                
    # Update best
    if current_conflicts == 0:
        if len(current_set) > best_val:
            best_val = len(current_set)
            best_set = current_set[:]
            
    # If we still have time, do one more greedy pass on the best set found?
    # Or just return the best.
    
    # Ensure we return a valid set.
    # If best_set is empty, return a known valid small set.
    if not best_set:
        # Fallback to powers of 3
        best_set = [3**i for i in range(0, 13) if 3**i <= N]
        
    return best_set
