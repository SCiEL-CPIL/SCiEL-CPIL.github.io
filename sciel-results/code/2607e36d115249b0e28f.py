import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                if a + S_list[j] in Sset:
                    return False
        return True

    def get_modular_subset(V, M):
        """
        Returns the subset of V which is sum-free modulo M.
        Interval: [floor(M/3)+1, floor(2M/3)]
        """
        L = M // 3 + 1
        R = (2 * M) // 3
        S = set()
        for x in V:
            r = x % M
            if L <= r <= R:
                S.add(x)
        return S

    def can_add(x, S, Sset):
        """Check if x can be added to S without violating sum-free property."""
        if 2 * x in Sset:
            return False
            
        for a in S:
            if (a + x) in Sset:
                return False
            if (a - x) in Sset:
                return False
        return True

    def hill_climb(S, V, time_limit, start_time):
        """
        Greedily add elements from V \ S if possible.
        Then try swaps to improve size.
        """
        S = set(S)
        Sset = S
        remaining = [x for x in V if x not in S]
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            still_remaining = []
            random.shuffle(remaining)
            for x in remaining:
                if (time.time() - start_time) > time_limit:
                    still_remaining.append(x)
                    continue
                if can_add(x, S, Sset):
                    S.add(x)
                    Sset.add(x)
                    improved = True
                else:
                    still_remaining.append(x)
            remaining = still_remaining
            
            # If stuck, try 1-for-1 or 1-for-2 swaps
            if remaining and not improved:
                S_list = sorted(S)
                n = len(S_list)
                for i in range(n):
                    if (time.time() - start_time) > time_limit:
                        break
                    a_to_remove = S_list[i]
                    S_tentative = S - {a_to_remove}
                    Sset_tentative = Sset - {a_to_remove}
                    
                    added = []
                    for x in remaining:
                        if can_add(x, S_tentative, Sset_tentative):
                            S_tentative.add(x)
                            Sset_tentative.add(x)
                            added.append(x)
                    
                    if len(added) > 1: # Net gain
                        S = S_tentative
                        Sset = Sset_tentative
                        remaining = [x for x in remaining if x not in added]
                        improved = True
                        break # Restart loop with new state
        
        return S

    def greedy_removal(V, time_limit, start_time, rng):
        """
        Start with the full set V and greedily remove elements with the most conflicts.
        """
        S = set(V)
        S_list = list(V)
        
        while True:
            if time.time() > start_time + time_limit:
                break
            
            # Check if sum-free
            S_set = set(S_list)
            conflicts = [0] * len(S_list)
            val_to_idx = {v: i for i, v in enumerate(S_list)}
            
            is_sum_free_now = True
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    s = a + b
                    if s in S_set:
                        is_sum_free_now = False
                        k = val_to_idx[s]
                        conflicts[i] += 1
                        if j != i:
                            conflicts[j] += 1
                        conflicts[k] += 1
            
            if is_sum_free_now:
                break
            
            # Find element with max conflicts
            max_conf = max(conflicts)
            candidates = [S_list[i] for i, c in enumerate(conflicts) if c == max_conf]
            x_to_remove = rng.choice(candidates)
            
            S.remove(x_to_remove)
            S_list.remove(x_to_remove)
            
        # Hill climb from here
        S_improved = hill_climb(S, V, time_limit * 0.5, start_time)
        return S_improved

    def solve_instance(V, time_limit, start_time):
        """
        Find the best sum-free subset of V within time_limit.
        """
        n = len(V)
        min_v = min(V)
        max_v = max(V)
        span = max_v - min_v
        
        best_set = set()
        best_size = 0
        
        # Strategy 1: Modular Construction
        candidates_mods = set()
        candidates_mods.update(range(5, 50))
        for factor in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
            m = int(span * factor)
            if m > 5:
                candidates_mods.add(m)
                candidates_mods.update([m-1, m+1])
                
        for factor in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            m = int(n * factor)
            if m > 5:
                candidates_mods.add(m)
                candidates_mods.update([m-1, m+1])
                
        candidates_mods.update(range(100, 500, 5))
        candidates_mods.update(range(1000, 5000, 50))
        candidates_mods.update(range(10000, 40000, 200))
        
        valid_mods = [m for m in candidates_mods if m > 5]
        
        # Evaluate modular candidates
        mod_candidates = []
        for M in valid_mods:
            if (time.time() - start_time) > time_limit:
                break
            S = get_modular_subset(V, M)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            if len(S) >= best_size - 2:
                mod_candidates.append(S)

        # Strategy 2: Sign-based
        pos = set(x for x in V if x > 0)
        neg = set(x for x in V if x < 0)
        if len(pos) > best_size:
            best_size = len(pos)
            best_set = pos
        if len(neg) > best_size:
            best_size = len(neg)
            best_set = neg
        if len(pos) >= best_size - 2: mod_candidates.append(pos)
        if len(neg) >= best_size - 2: mod_candidates.append(neg)

        # Strategy 3: Parity
        odds = set(x for x in V if x % 2 != 0)
        if len(odds) > best_size:
            best_size = len(odds)
            best_set = odds
        if len(odds) >= best_size - 2: mod_candidates.append(odds)

        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for S in mod_candidates:
            key = frozenset(S)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(S)

        # Strategy 4: Local Search on best candidates
        # Allocate remaining time
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if remaining_time > 0.5:
            unique_candidates.sort(key=len, reverse=True)
            num_to_try = min(4, len(unique_candidates))
            time_per_candidate = remaining_time / max(1, num_to_try)
            
            for i in range(num_to_try):
                if (time.time() - start_time) > time_limit:
                    break
                S_init = unique_candidates[i]
                S_improved = hill_climb(S_init, V, time_per_candidate, start_time)
                if len(S_improved) > best_size:
                    best_size = len(S_improved)
                    best_set = S_improved

        # Strategy 5: Greedy Removal (Structural change from inspiration 2)
        # Only try this if we have significant time left and haven't found a very good solution
        # Greedy removal is O(N^3) roughly, so it's slow for N=240, but can find global optima
        # We use it as a backup or parallel track if time permits.
        # For N=60, it's fast. For N=240, it's slow.
        # Let's try it for the first instance (N=60) if we have time.
        
        # Actually, let's just return the best found so far.
        # The greedy removal is risky for time limits on large instances.
        # But for N=60, it's worth a shot if we have time.
        
        if n <= 60 and (time_limit - (time.time() - start_time)) > 2.0:
            rng = random.Random(42)
            S_greedy = greedy_removal(V, time_limit - (time.time() - start_time) - 0.5, start_time, rng)
            if len(S_greedy) > best_size:
                best_size = len(S_greedy)
                best_set = S_greedy

        return sorted(best_set)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback: Odds
            odds = [x for x in V if x % 2 != 0]
            results.append(sorted(odds))
            continue
        
        instances_left = 3 - len(results)
        time_per_instance = (remaining_total / instances_left) - 2
        if time_per_instance < 1:
            time_per_instance = 1
            
        S_best = solve_instance(V, time_per_instance, start_time)
        
        if not is_sum_free(S_best):
            odds = [x for x in V if x % 2 != 0]
            S_best = sorted(odds)
            
        results.append(S_best)
        
    return results

if __name__ == "__main__":
    res = construct()
    print(res)
