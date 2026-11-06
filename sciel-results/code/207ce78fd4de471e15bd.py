import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precomputation
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # For each element x, we want to know which sets of 3 other elements {y, z, w}
    # would form a 4-AP with x.
    # If S contains {y, z, w}, we cannot add x.
    
    # Instead of storing triples, we can store masks.
    # But for efficiency in "finding a remover", we want to know the specific APs.
    
    # Let's precompute for each index i, a list of (mask, ap_indices)
    # where mask is the bitmask of the other 3 elements.
    # Actually, for the swap logic:
    # If we want to add 'a', we check if any forbidden triple for 'a' is fully in S.
    # If yes, we have a conflict. We want to remove one element from the conflict.
    # To maximize chance of success, we should remove an element that is "cheap" 
    # (part of few other potential APs) or "expensive" (part of many, so removing it helps more)?
    # Usually, removing an element that is in many other conflicts is better.
    
    # Let's precompute:
    # conflicts[i] = list of (mask, indices_tuple)
    # indices_tuple is the 3 indices that with i form a 4-AP.
    
    conflicts = [[] for _ in range(N)]
    # Also precompute degree: how many 4-APs each index is part of.
    # This helps in deciding which element to remove.
    degree = [0] * N
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            i, j, k, l = a-1, b-1, c-1, e-1
            
            # Update degrees
            degree[i] += 1
            degree[j] += 1
            degree[k] += 1
            degree[l] += 1
            
            # Store conflicts
            # For i, the other three are j, k, l
            mask_i = (1 << j) | (1 << k) | (1 << l)
            conflicts[i].append((mask_i, (j, k, l)))
            
            # For j, the other three are i, k, l
            mask_j = (1 << i) | (1 << k) | (1 << l)
            conflicts[j].append((mask_j, (i, k, l)))
            
            # For k, the other three are i, j, l
            mask_k = (1 << i) | (1 << j) | (1 << l)
            conflicts[k].append((mask_k, (i, j, l)))
            
            # For l, the other three are i, j, k
            mask_l = (1 << i) | (1 << j) | (1 << k)
            conflicts[l].append((mask_l, (i, j, k)))
            
    # 2. Helper Functions
    
    def can_add(bits, x):
        """Check if x can be added to set represented by bits."""
        for mask, _ in conflicts[x]:
            if (bits & mask) == mask:
                return False
        return True

    def get_conflicts(bits, x):
        """Get list of conflicting triples for x given current set bits."""
        confs = []
        for mask, indices in conflicts[x]:
            if (bits & mask) == mask:
                confs.append(indices)
        return confs

    def greedy_fill(bits):
        """Try to add any element that doesn't conflict."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order to break symmetry
            candidates = [i for i in range(N) if not (bits & (1 << i))]
            random.shuffle(candidates)
            for i in candidates:
                if time.time() > deadline:
                    break
                if not (bits & (1 << i)):
                    if can_add(bits, i):
                        bits |= (1 << i)
                        changed = True
        return bits

    def local_search(bits, max_iters=1000):
        """
        Swap-based local search.
        1. Pick a random candidate 'a' to add.
        2. If it conflicts, find a conflicting triple.
        3. Pick an element 'r' from that triple to remove.
        4. Perform swap (remove r, add a).
        5. If size increases (which it does if we just add? No, swap keeps size same).
        
        Wait, standard local search for MaxSAT/MaxCSP:
        We want to maximize cardinality.
        Strategy: 
        - Try to add a random element 'a'. 
        - If successful, size +1.
        - If failed, find conflict. Remove one element from conflict, add 'a'. Size same.
        - This is a plateau walk.
        
        Better Strategy:
        - Maintain a set S.
        - Identify "blocked" elements (cannot be added).
        - Try to unblock one by removing a "critical" element.
        
        Let's use a simpler robust method:
        1. Greedy fill to local optimum.
        2. Repeat:
           a. Pick random 'a' not in S.
           b. Get conflicts for 'a'.
           c. If no conflicts, add 'a' (size +1).
           d. If conflicts, pick one conflict triple T.
           e. Pick 'r' from T to remove. 
              Heuristic: Remove 'r' that has highest degree (most connected) 
              or lowest degree? 
              Removing high degree might open up more space.
              Let's try removing the one with highest degree.
           f. Remove 'r', add 'a'.
           g. If this leads to a state where we can add MORE elements later, good.
           h. Actually, just doing the swap keeps size same. 
           i. To increase size, we need to find a sequence of swaps that leads to a state 
              where an element can be added.
        
        Let's implement a "Kick and Recover" strategy:
        1. Start from local optimum.
        2. Perform a random kick (remove 1-3 random elements).
        3. Run greedy fill to new local optimum.
        4. If new size > old size, keep. Else, revert (or keep if we want diversity, but for max, revert).
        
        This is essentially Simulated Annealing with a greedy repair.
        """
        current_bits = bits
        current_size = bin(current_bits).count('1')
        
        # First, ensure we are at a local optimum via greedy
        current_bits = greedy_fill(current_bits)
        current_size = bin(current_bits).count('1')
        
        best_bits = current_bits
        best_size = current_size
        
        # Annealing parameters
        T = 1.0
        cooling = 0.99
        
        while time.time() < deadline:
            if T < 0.01:
                break
            
            # Pick a random element to remove (kick)
            # Sometimes remove 1, sometimes 2
            num_removals = random.choice([1, 1, 2])
            elements_in_set = [i for i in range(N) if (current_bits >> i) & 1]
            if not elements_in_set:
                break
            
            removed = random.sample(elements_in_set, min(num_removals, len(elements_in_set)))
            trial_bits = current_bits
            for r in removed:
                trial_bits &= ~(1 << r)
            
            # Greedy fill
            trial_bits = greedy_fill(trial_bits)
            trial_size = bin(trial_bits).count('1')
            
            # Acceptance criterion
            delta = trial_size - current_size
            if delta > 0:
                # Improvement
                current_bits = trial_bits
                current_size = trial_size
                if current_size > best_size:
                    best_bits = current_bits
                    best_size = current_size
            else:
                # Degradation or same
                # Accept with probability exp(delta / T)
                # Since delta is negative or 0, exp(neg) < 1
                if random.random() < (2.718 ** (delta / T)):
                    current_bits = trial_bits
                    current_size = trial_size
            
            T *= cooling
            
        return best_bits, best_size

    # 3. Initialization
    
    best_global_bits = 0
    best_global_size = 0
    
    # Strategy A: Ternary Digit Heuristic
    # Numbers with no 0 in ternary representation are 3-free.
    # Are they 4-free?
    # 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29...
    # Let's check 4-APs.
    # This set is known to be 3-AP free. It might contain 4-APs?
    # Actually, the set of numbers with no digit 0 in base 3 is 3-AP free.
    # Does it contain 4-APs?
    # 1, 4, 7, 10 -> 1(1), 4(11), 7(21), 10(101). No 0s. 
    # 1, 4, 7, 10 is a 4-AP with d=3.
    # So the ternary set is NOT 4-AP free.
    # We need to repair it.
    
    init_bits = 0
    for num in range(1, N + 1):
        x = num
        valid = True
        while x > 0:
            if x % 3 == 0:
                valid = False
                break
            x //= 3
        if valid:
            init_bits |= (1 << (num - 1))
            
    # Repair and optimize
    b, s = local_search(init_bits)
    if s > best_global_size:
        best_global_size = s
        best_global_bits = b

    # Strategy B: Random Restart
    # Run multiple random restarts with local search
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Pick a random subset of size ~ N/4
        # Or start empty and greedy?
        # Starting empty and greedy is fast but might get stuck in small sets.
        # Let's start with a random set of density 0.3
        random_bits = 0
        for i in range(N):
            if random.random() < 0.3:
                random_bits |= (1 << i)
        
        # Repair: Remove elements from any 4-APs
        # Simple repair: iterate through all 4-APs, if all present, remove one.
        # This is O(N^2) per pass.
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for d in range(1, (N - 1) // 3 + 1):
                for a in range(1, N - 3 * d + 1):
                    b_idx = a + d
                    c_idx = a + 2 * d
                    e_idx = a + 3 * d
                    
                    i, j, k, l = a-1, b_idx-1, c_idx-1, e_idx-1
                    
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    if (random_bits & mask) == mask:
                        # Remove one, say the last one
                        random_bits &= ~(1 << l)
                        changed = True
                        break
                if changed:
                    break
        
        # Now local search
        b, s = local_search(random_bits)
        if s > best_global_size:
            best_global_size = s
            best_global_bits = b

    # Strategy C: Specific Construction
    # Try a construction based on base 4 digits?
    # Numbers with no 0 or 1 in base 4?
    # This is getting complex. The random restarts with good local search should suffice.
    
    # Final check
    result = [i + 1 for i in range(N) if (best_global_bits >> i) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
