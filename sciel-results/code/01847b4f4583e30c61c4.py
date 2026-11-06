import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 150s limit
    
    N = 1000
    MAX_SUM = 2 * N
    
    def repair_and_optimize(S_init):
        """
        Takes an initial set, repairs it to be a valid B2[2] set, 
        then tries to add elements to maximize size.
        """
        S = set(S_init)
        in_set = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        # Initialize counts
        for x in S:
            in_set[x] = True
        
        # Compute initial counts
        # O(|S|^2)
        s_list = list(S)
        for i in range(len(s_list)):
            x = s_list[i]
            counts[2*x] += 1
            for j in range(i + 1, len(s_list)):
                y = s_list[j]
                counts[x+y] += 1
        
        # Repair Phase
        # While there are sums with count > 2
        # We limit iterations to prevent infinite loops or excessive time
        repair_iters = 0
        max_repair_iters = 5000
        
        while repair_iters < max_repair_iters:
            if time.time() > deadline:
                break
            
            # Check for conflicts
            max_count = np.max(counts)
            if max_count <= 2:
                break
            
            # Find a conflicting sum
            # We pick the sum with the highest count to resolve the most severe conflict
            conflict_sums = np.where(counts > 2)[0]
            if len(conflict_sums) == 0:
                break
            
            s = conflict_sums[np.argmax(counts[conflict_sums])]
            
            # Find pairs (a, b) in S such that a + b = s
            pairs = []
            # a <= b => a <= s/2
            for a in range(1, s // 2 + 1):
                b = s - a
                if b > N:
                    continue
                if in_set[a] and in_set[b]:
                    pairs.append((a, b))
            
            if not pairs:
                # Inconsistency in counts?
                break
            
            # Choose an element to remove
            # Heuristic: Remove an element that is part of the most conflicting sums?
            # Or just random from this pair?
            # Let's try to remove the element that is "less central" in the graph of conflicts.
            # Simple: Pick a random pair, then pick a random element from it.
            a, b = random.choice(pairs)
            # Remove a or b. Let's remove the one that appears in fewer other sums?
            # To keep it simple, remove a.
            S.remove(a)
            in_set[a] = False
            
            # Update counts
            counts[2*a] -= 1
            # Decrement counts for a + y for all y in S (which is now S \ {a})
            # We need to iterate over the current S
            for y in S:
                counts[a+y] -= 1
            
            repair_iters += 1
        
        # Optimization Phase
        # Try to add elements back
        # We do this in passes.
        # In each pass, we try to add any valid candidate.
        # We repeat until no more can be added.
        
        while True:
            if time.time() > deadline:
                break
            
            added_any = False
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Check validity of adding x
                # 1. Check 2x
                if counts[2*x] >= 2:
                    continue
                
                # 2. Check x + y for y in S
                s_indices = np.where(in_set)[0]
                if len(s_indices) > 0:
                    sums = x + s_indices
                    sum_counts = counts[sums]
                    if np.any(sum_counts >= 2):
                        continue
                
                # Valid, add x
                S.add(x)
                in_set[x] = True
                counts[2*x] += 1
                if len(s_indices) > 0:
                    new_sums = x + s_indices
                    counts[new_sums] += 1
                added_any = True
                # After adding, counts change. 
                # We continue checking other candidates with the updated counts.
                # This is efficient enough.
            
            if not added_any:
                break
        
        return S

    best_set = set()
    best_size = 0
    
    # Try different initial densities
    # Based on the upper bound ~89, and typical B2[2] densities being lower,
    # let's try p in [0.02, 0.08]
    
    num_attempts = 0
    while time.time() < deadline:
        num_attempts += 1
        
        p = random.uniform(0.02, 0.08)
        
        S_init = set()
        for i in range(1, N + 1):
            if random.random() < p:
                S_init.add(i)
        
        S_res = repair_and_optimize(S_init)
        
        if len(S_res) > best_size:
            best_size = len(S_res)
            best_set = S_res
            
        # Optional: If we hit a high size, we can try to refine further?
        # The repair_and_optimize already does local search.
        
    return list(best_set)
