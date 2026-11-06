import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def count_conflicts_numpy(S_list):
        """Vectorized conflict counting using numpy."""
        n = len(S_list)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        
        arr = np.array(S_list, dtype=np.int64)
        S_set = set(S_list)
        
        # Build all pair sums (including i=j for a+a)
        # Use broadcasting: sums[i,j] = arr[i] + arr[j]
        sums = arr[:, None] + arr[None, :]  # shape (n, n)
        
        # For each sum, check if it's in S_set
        # We need to map sums back to indices in arr
        # Create a dict: value -> index (first occurrence)
        val_to_idx = {}
        for idx, v in enumerate(S_list):
            if v not in val_to_idx:
                val_to_idx[v] = idx
        
        # Count conflicts: for each (i,j) where arr[i]+arr[j] is in S_set,
        # increment conflicts for i, j, and the index of the sum
        conflicts = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            for j in range(i, n):
                s = S_list[i] + S_list[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    conflicts[i] += 1
                    if j != i:
                        conflicts[j] += 1
                    conflicts[k] += 1
        
        return conflicts

    def count_conflicts_fast(S_list):
        """Fast conflict counting with early termination per element."""
        n = len(S_list)
        if n == 0:
            return []
        
        S_set = set(S_list)
        val_to_idx = {}
        for idx, v in enumerate(S_list):
            if v not in val_to_idx:
                val_to_idx[v] = idx
        
        conflicts = [0] * n
        
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                s = a + b
                if s in val_to_idx:
                    k = val_to_idx[s]
                    conflicts[i] += 1
                    if j != i:
                        conflicts[j] += 1
                    conflicts[k] += 1
        
        return conflicts

    def greedy_optimize(V, time_limit, rng):
        S = set(V)
        removed = list(V)
        
        # 1. Greedy Removal with efficient conflict detection
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break
            
            S_list = list(S)
            conflicts = count_conflicts_fast(S_list)
            
            if not any(conflicts):
                break
                
            # Pick element with maximum conflicts
            max_deg = max(conflicts)
            candidates = [S_list[i] for i, c in enumerate(conflicts) if c == max_deg]
            x_to_remove = rng.choice(candidates)
            
            S.remove(x_to_remove)
            removed.remove(x_to_remove)

        # 2. Hill Climbing
        def can_add(x, S_set, S_list):
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True

        # Multiple passes of hill climbing
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            rng.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return list(S)

    def parity_seeded_optimize(V, time_limit, rng):
        """Start with a parity-based sum-free subset, then optimize."""
        # Odd numbers form a sum-free set (odd+odd=even, not odd)
        # Even numbers: even+even=even, so not sum-free in general
        # Strategy: take all odds, then try to add evens that don't conflict
        
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        
        S = set(odds)
        removed = list(evens)
        
        # Try to add evens that don't create conflicts
        def can_add(x, S_set, S_list):
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True
        
        # Greedily add evens
        S_list = list(S)
        S_set = S
        still_removed = []
        for x in evens:
            if can_add(x, S_set, S_list):
                S.add(x)
                S_list.append(x)
                S_set.add(x)
            else:
                still_removed.append(x)
        removed = still_removed
        
        # Now do hill climbing from this starting point
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            rng.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return list(S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Allocate time based on instance size
        if size <= 60:
            base_time = 10.0
        elif size <= 120:
            base_time = 15.0
        else:
            base_time = 25.0
        
        # Try multiple strategies
        strategies = [
            ("greedy", lambda t, r: greedy_optimize(V, t, r)),
            ("parity", lambda t, r: parity_seeded_optimize(V, t, r)),
        ]
        
        # Run each strategy with decreasing time budgets
        total_time = base_time
        time_per_strategy = total_time / len(strategies)
        
        for strat_name, strat_func in strategies:
            if time.time() > start_time + DEADLINE - 5:
                break
            
            remaining = min(time_per_strategy, DEADLINE - (time.time() - start_time))
            if remaining < 1.0:
                break
            
            rng = random.Random(seed + hash(strat_name) % 10000)
            res = strat_func(remaining, rng)
            
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
        
        # If time remains, do more restarts with the best strategy
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed - 5
        
        if remaining_total > 3.0 and best_size > 0:
            # Determine which strategy worked better and do more of that
            # For simplicity, alternate
            extra_restarts = min(3, int(remaining_total / 5.0))
            for r in range(extra_restarts):
                if time.time() > start_time + DEADLINE - 3:
                    break
                
                t_left = min(5.0, (DEADLINE - 3) - (time.time() - start_time))
                if t_left < 1.0:
                    break
                
                if r % 2 == 0:
                    strat_name, strat_func = strategies[0]
                else:
                    strat_name, strat_func = strategies[1]
                
                rng = random.Random(seed + r * 7919)
                res = strat_func(t_left, rng)
                
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = res
        
        results.append(best_set)

    return results
