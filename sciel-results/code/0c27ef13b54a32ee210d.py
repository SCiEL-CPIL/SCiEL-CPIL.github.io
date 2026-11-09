# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start = time.time()
    DEADLINE = 115  # safety margin before 120s evaluator limit
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    PRESCRIBED_SET = set(PRESCRIBED)

    def compute_collisions(S):
        """Returns the number of duplicate sums. 0 means valid Sidon set."""
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        collisions = 0
        for count in sums.values():
            if count > 1:
                collisions += count - 1
        return collisions

    def is_valid(S):
        return compute_collisions(S) == 0

    def greedy_construct(rng, target_size=None):
        """
        Greedy construction. If target_size is None, adds as many as possible.
        Returns a valid set.
        """
        S = list(PRESCRIBED)
        S.sort()
        used = set(S)
        
        # Precompute sums for efficiency in checking? 
        # For greedy, simple check is fine if N is small (284).
        
        candidates = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(candidates)
        
        # Standard greedy: try to add each candidate in random order
        for x in candidates:
            if target_size and len(S) >= target_size:
                break
            # Check if x can be added
            valid = True
            for y in S:
                s = x + y
                # Check if s is already a sum of two elements in S
                # We need to know if s = a+b for some a,b in S (a!=b or a==b)
                # Efficient check:
                # If x is new, the new sums are x+y for y in S, and 2x.
                # We must ensure none of these are in the existing sum set.
                # Let's maintain a set of sums.
                pass
        
        # Re-implement with sum set for speed
        S = list(PRESCRIBED)
        S.sort()
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED_SET]
        rng.shuffle(candidates)
        
        for x in candidates:
            if target_size and len(S) >= target_size:
                break
            # Check if x can be added
            # New sums: x+y for y in S, and 2x
            ok = True
            for y in S:
                if x + y in sums:
                    ok = False
                    break
            if ok and 2 * x in sums:
                ok = False
            
            if ok:
                S.append(x)
                S.sort()
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
                
        return S

    def simulated_annealing(rng, target_size, initial_S=None, time_limit=10.0):
        """
        Attempts to find a valid Sidon set of exactly target_size.
        Returns the set if found, else None.
        """
        if initial_S and len(initial_S) == target_size:
            S = list(initial_S)
        else:
            # Initialize with prescribed elements and random others
            S = list(PRESCRIBED)
            available = [x for x in range(1, N + 1) if x not in PRESCRIBED_SET]
            rng.shuffle(available)
            S.extend(available[:target_size - len(PRESCRIBED)])
        
        S.sort()
        current_cost = compute_collisions(S)
        
        if current_cost == 0:
            return S

        best_S = list(S)
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        sa_start = time.time()
        
        while T > T_min and (time.time() - sa_start) < time_limit:
            # Generate neighbor
            S_new = list(S)
            # Pick a non-prescribed index to replace
            non_prescribed_indices = [i for i, v in enumerate(S_new) if v not in PRESCRIBED_SET]
            if not non_prescribed_indices:
                break
            idx = rng.choice(non_prescribed_indices)
            
            # Pick a new value
            # Avoid current value
            new_val = rng.randint(1, N)
            # Ensure distinctness
            while new_val in S_new:
                new_val = rng.randint(1, N)
            
            S_new[idx] = new_val
            S_new.sort()
            
            new_cost = compute_collisions(S_new)
            
            if new_cost < current_cost or (T > 0 and rng.random() < math.exp(-(new_cost - current_cost) / T)):
                S = S_new
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_S = list(S)
                    
                    if best_cost == 0:
                        return best_S

            T *= cooling
            
        return best_S if best_cost == 0 else None

    # Strategy:
    # 1. Find a good starting set using greedy.
    # 2. Try to maximize size using SA on increasing target sizes.
    
    rng = random.Random(42)
    
    # Phase 1: Greedy to get a baseline
    baseline_S = greedy_construct(rng)
    best_size = len(baseline_S)
    best_set = baseline_S
    
    # Phase 2: Try to improve size using SA
    # We know max is likely 20. Let's try to find a valid set of size 20, 19, etc.
    # Start from high and go down? Or start from current and go up?
    # Starting from current and going up is safer because we have a valid set of size best_size.
    
    current_size = best_size
    current_set = best_set
    
    # Try to extend
    while current_size < 20:
        next_size = current_size + 1
        if time.time() - start > DEADLINE - 5:
            break
        
        # Run SA for a few seconds to find a valid set of next_size
        # Use current_set as a seed (extend it randomly or just use it for initialization context)
        # Actually, SA needs a set of exactly next_size.
        # We can take current_set and add a random element, then let SA fix it.
        
        found = False
        attempts = 3
        for attempt in range(attempts):
            if time.time() - start > DEADLINE - 5:
                break
            
            # Initialize SA with a set of size next_size
            # Take current_set, add a random new element
            init_set = list(current_set)
            available = [x for x in range(1, N + 1) if x not in set(init_set)]
            if not available:
                break
            init_set.append(rng.choice(available))
            init_set.sort()
            
            # Run SA
            res = simulated_annealing(rng, next_size, initial_S=init_set, time_limit=15.0)
            if res:
                found = True
                current_size = next_size
                current_set = res
                break
        
        if not found:
            break
            
    # If we didn't reach 20, maybe we can improve the current best set via SA at the same size?
    # No, if we found a valid set of size k, that's the best we can do for that size.
    # We only care about maximizing size.
    
    return current_set

# EVOLVE-BLOCK-END
