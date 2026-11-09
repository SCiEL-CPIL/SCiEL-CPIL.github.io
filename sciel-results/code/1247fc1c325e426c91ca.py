import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def find_conflict_subset(elements):
        """
        Finds a subset of elements that sums to 200.
        Returns the subset (list) or None.
        Uses DP with backtracking.
        """
        target = 200
        reachable = [False] * (target + 1)
        parent = [None] * (target + 1)
        reachable[0] = True
        
        # The order of elements determines which specific subset is found
        for x in elements:
            if x > target:
                continue
            # Iterate backwards to ensure each element is used at most once
            for s in range(target, x - 1, -1):
                if not reachable[s] and reachable[s - x]:
                    reachable[s] = True
                    parent[s] = (s - x, x)
        
        if not reachable[target]:
            return None
        
        # Backtrack to find the subset
        subset = []
        s = target
        while s > 0:
            prev_s, x = parent[s]
            subset.append(x)
            s = prev_s
            
        return subset

    def can_add_mask(mask, x):
        """Check if adding x to a set with reachable mask would create sum 200."""
        if x > 200:
            return True
        if (mask >> (200 - x)) & 1:
            return False
        return True

    def add_element_mask(mask, x):
        """Add element x to the reachable mask."""
        if x > 200:
            return mask
        new_mask = mask | (mask << x)
        return new_mask & ((1 << 201) - 1)

    def compute_mask(elements):
        """Compute the full reachable mask for a set of elements."""
        mask = 1
        for x in elements:
            mask = add_element_mask(mask, x)
        return mask

    def solve_with_removal(strategy, seed=0):
        """
        Solves the problem by starting with all elements and removing conflicts.
        strategy: 'random', 'min', 'max', 'freq'
        """
        if seed is not None:
            random.seed(seed)
            
        S = list(range(1, 101))
        removed = []
        
        # Phase 1: Removal
        while time.time() < deadline - 5.0:
            # Find a conflict
            # For 'freq', we might want to shuffle S to find different conflicts
            # But S is the current set. The conflict is within S.
            # find_conflict_subset uses the order of S.
            # To get variety in 'freq', we shuffle a copy for the search?
            # No, the conflict must be in S.
            # The specific conflict found depends on the iteration order in find_conflict_subset.
            
            C = find_conflict_subset(S)
            if C is None:
                break
            
            # Select element to remove
            if strategy == 'random':
                x = random.choice(C)
            elif strategy == 'min':
                x = min(C)
            elif strategy == 'max':
                x = max(C)
            elif strategy == 'freq':
                # Sample multiple conflicts by shuffling the input order for the DP
                # Note: The set S is fixed, but the order in which we process them in DP
                # determines which specific subset is identified.
                conflicts = []
                for _ in range(5):
                    shuffled_S = S[:]
                    random.shuffle(shuffled_S)
                    C_sample = find_conflict_subset(shuffled_S)
                    if C_sample:
                        conflicts.append(C_sample)
                
                # Count frequencies
                freq = {}
                for c in conflicts:
                    for x in c:
                        freq[x] = freq.get(x, 0) + 1
                
                # Pick element from C with highest freq
                max_freq = max(freq.get(x, 0) for x in C)
                candidates = [x for x in C if freq.get(x, 0) == max_freq]
                x = random.choice(candidates)
            else:
                x = random.choice(C)
            
            S.remove(x)
            removed.append(x)
        
        # Phase 2: Polish (Add back)
        # Compute mask of current valid set S
        mask = compute_mask(S)
        
        # Sort removed elements. 
        # Heuristic: Try adding smaller elements first? Or random?
        # Smaller elements are more "dangerous" (more likely to cause conflict), 
        # but also more "valuable" for cardinality?
        # Actually, if we removed them, they *were* dangerous.
        # Let's try adding in random order, or sorted order.
        # Sorted order (smallest first) might allow packing more small numbers?
        # Let's try sorted.
        for x in sorted(removed):
            if time.time() > deadline - 2.0:
                break
            if can_add_mask(mask, x):
                S.append(x)
                mask = add_element_mask(mask, x)
        
        return S

    best_set = []
    best_size = 0
    
    # Run multiple strategies with different seeds
    strategies = ['freq', 'random', 'min', 'max']
    
    # First pass: Deterministic-ish
    for strat in strategies:
        if time.time() > deadline - 10.0:
            break
        res = solve_with_removal(strat, seed=42)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
    
    # Second pass: Randomized restarts for 'freq' and 'random'
    attempts = 0
    while time.time() < deadline - 5.0:
        strat = random.choice(['freq', 'random'])
        seed = random.randint(0, 10000)
        res = solve_with_removal(strat, seed=seed)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        attempts += 1
        if attempts >= 50: # Cap restarts to ensure time for polish if needed? 
             # Actually, solve_with_removal includes polish.
             # 50 restarts * ~0.1s = 5s. Safe.
             break

    if not best_set:
        best_set = [100]
        
    return best_set
