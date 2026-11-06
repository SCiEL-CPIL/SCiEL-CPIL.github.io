"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    N = 1000
    
    # sum_counts[s] = number of pairs (a,b) in set with a<=b and a+b=s
    # Indexed 0..2N, we use indices 2..2N
    sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
    indicator = np.zeros(N + 1, dtype=np.uint8)
    chosen = []
    
    def add_to_counts(x):
        """Add x to the set, updating sum_counts. Returns True if valid (all <= 2)."""
        # For each existing y in chosen, add representation for sum x+y
        # Also add 2*x (the diagonal)
        for y in chosen:
            s = x + y
            sum_counts[s] += 1
            if sum_counts[s] > 2:
                return False
        sum_counts[2 * x] += 1
        if sum_counts[2 * x] > 2:
            return False
        return True
    
    def remove_from_counts(x):
        """Remove x from the set, updating sum_counts."""
        for y in chosen:
            if y != x:
                s = x + y
                sum_counts[s] -= 1
        sum_counts[2 * x] -= 1
    
    def is_valid():
        return np.all(sum_counts[2:2*N+1] <= 2)
    
    # Phase 1: Greedy baseline (ascending order)
    for x in range(1, N + 1):
        if time.time() > deadline - 60:
            break
        if add_to_counts(x):
            indicator[x] = 1
            chosen.append(x)
        else:
            # Revert
            for y in chosen:
                if y != x:
                    sum_counts[x + y] -= 1
            sum_counts[2 * x] -= 1
    
    best_set = list(chosen)
    best_size = len(chosen)
    best_counts = sum_counts.copy()
    best_indicator = indicator.copy()
    
    np.random.seed(42)
    
    # Phase 2: Probabilistic construction with LLL-inspired repair
    # Select each element with probability p, then repair over-represented sums
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        
        p = 0.45 + 0.1 * (attempt % 5) / 5.0  # Vary probability
        sc = np.zeros(2 * N + 1, dtype=np.int32)
        ind = np.zeros(N + 1, dtype=np.uint8)
        ch = []
        
        # Random selection
        for x in range(1, N + 1):
            if np.random.random() < p:
                # Try to add
                valid = True
                for y in ch:
                    s = x + y
                    sc[s] += 1
                    if sc[s] > 2:
                        valid = False
                        break
                if valid:
                    sc[2 * x] += 1
                    if sc[2 * x] > 2:
                        valid = False
                if valid:
                    ind[x] = 1
                    ch.append(x)
        
        # Repair phase: for each over-represented sum, remove a contributor
        # Iterate until stable or time runs out
        repair_rounds = 0
        while repair_rounds < 50 and time.time() < deadline - 15:
            repair_rounds += 1
            # Find over-represented sums
            over_sums = np.where(sc[2:2*N+1] > 2)[0] + 2
            if len(over_sums) == 0:
                break
            
            # For each over-represented sum, pick a random contributor to remove
            removed_this_round = False
            for s in over_sums:
                if time.time() > deadline - 15:
                    break
                # Find all pairs (a,b) with a<=b, a+b=s, both in set
                # a ranges from max(1, s-N) to min(N, s//2)
                a_lo = max(1, s - N)
                a_hi = min(N, s // 2)
                contributors = []
                for a in range(a_lo, a_hi + 1):
                    b = s - a
                    if b >= a and ind[a] and ind[b]:
                        contributors.append((a, b))
                
                if not contributors:
                    continue
                
                # Remove a random contributor
                a, b = contributors[np.random.randint(len(contributors))]
                # Remove the one that appears in more over-represented sums
                # (greedy heuristic: remove the more "problematic" element)
                if a == b:
                    # Remove a (diagonal)
                    for y in ch:
                        if y != a:
                            sc[a + y] -= 1
                    sc[2 * a] -= 1
                    ind[a] = 0
                    ch.remove(a)
                else:
                    # Decide which to remove: a or b
                    # Remove the one that contributes to more sums > 2
                    count_a = 0
                    count_b = 0
                    for y in ch:
                        if y != a and sc[a + y] > 2:
                            count_a += 1
                        if y != b and sc[b + y] > 2:
                            count_b += 1
                    if count_a >= count_b:
                        # Remove a
                        for y in ch:
                            if y != a:
                                sc[a + y] -= 1
                        sc[2 * a] -= 1
                        ind[a] = 0
                        ch.remove(a)
                    else:
                        # Remove b
                        for y in ch:
                            if y != b:
                                sc[b + y] -= 1
                        sc[2 * b] -= 1
                        ind[b] = 0
                        ch.remove(b)
                removed_this_round = True
            
            if not removed_this_round:
                break
        
        # Try to add back elements after repair
        for x in range(1, N + 1):
            if time.time() > deadline - 15:
                break
            if ind[x] == 0:
                valid = True
                for y in ch:
                    s = x + y
                    if sc[s] + 1 > 2:
                        valid = False
                        break
                if valid:
                    if sc[2 * x] + 1 > 2:
                        valid = False
                if valid:
                    for y in ch:
                        sc[x + y] += 1
                    sc[2 * x] += 1
                    ind[x] = 1
                    ch.append(x)
        
        if len(ch) > best_size:
            best_size = len(ch)
            best_set = list(ch)
            best_counts = sc.copy()
            best_indicator = ind.copy()
    
    # Phase 3: Local search on best set - try to add elements
    if time.time() < deadline - 5:
        sc = best_counts.copy()
        ind = best_indicator.copy()
        ch = list(best_set)
        
        # Try adding each missing element
        for x in range(1, N + 1):
            if time.time() > deadline - 3:
                break
            if ind[x] == 0:
                valid = True
                for y in ch:
                    s = x + y
                    if sc[s] + 1 > 2:
                        valid = False
                        break
                if valid and sc[2 * x] + 1 <= 2:
                    for y in ch:
                        sc[x + y] += 1
                    sc[2 * x] += 1
                    ind[x] = 1
                    ch.append(x)
                    best_size += 1
                    best_set = list(ch)
                    best_counts = sc
                    best_indicator = ind
    
    # Phase 4: Swap-based local search
    if time.time() < deadline - 2:
        sc = best_counts.copy()
        ind = best_indicator.copy()
        ch = list(best_set)
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            # Try: remove one random element, add two random others
            if len(ch) < 2:
                break
            
            idx = np.random.randint(len(ch))
            removed = ch[idx]
            
            # Temporarily remove
            temp_sc = sc.copy()
            temp_ind = ind.copy()
            temp_ch = ch[:]
            
            for y in temp_ch:
                if y != removed:
                    temp_sc[removed + y] -= 1
            temp_sc[2 * removed] -= 1
            temp_ind[removed] = 0
            temp_ch.pop(idx)
            
            # Try to add two elements
            added = []
            for _ in range(2):
                x = np.random.randint(1, N + 1)
                if temp_ind[x] == 0:
                    valid = True
                    for y in temp_ch:
                        s = x + y
                        if temp_sc[s] + 1 > 2:
                            valid = False
                            break
                    if valid and temp_sc[2 * x] + 1 <= 2:
                        for y in temp_ch:
                            temp_sc[x + y] += 1
                        temp_sc[2 * x] += 1
                        temp_ind[x] = 1
                        temp_ch.append(x)
                        added.append(x)
            
            if len(added) == 2:
                best_set = list(temp_ch)
                best_size = len(best_set)
                best_counts = temp_sc
                best_indicator = temp_ind
            # else: revert (keep original)
    
    return best_set
# EVOLVE-BLOCK-END
