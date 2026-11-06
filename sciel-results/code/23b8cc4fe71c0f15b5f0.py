import time
import random
import numpy as np
from functools import lru_cache

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
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def get_conflicts(S):
        """
        Returns a list of tuples (a, b, c) where a+b=c and a,b,c in S.
        Also returns a set of all elements involved in conflicts.
        """
        S_set = set(S)
        conflicts = []
        involved = set()
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                s = a + b
                if s in S_set:
                    conflicts.append((a, b, s))
                    involved.add(a)
                    involved.add(b)
                    involved.add(s)
        return conflicts, involved

    def count_conflicts_for_element(x, S_set, S_list):
        """
        Efficiently count how many conflicts x is involved in.
        x is not in S.
        """
        count = 0
        for a in S_list:
            if (x + a) in S_set:
                count += 1
            if (x - a) in S_set:
                count += 1
        return count

    def solve_instance(V, time_limit):
        """
        Solves for a large sum-free subset of V using modular seeding and simulated annealing.
        """
        V_set = set(V)
        
        # Strategy 1: Modular Seeding
        # Sets with all elements congruent to 1 mod 3 or 2 mod 3 are sum-free.
        # We pick the one that yields a larger subset.
        candidates = []
        
        # Mod 1
        mod1 = [x for x in V if x % 3 == 1]
        # Mod 2
        mod2 = [x for x in V if x % 3 == 2]
        
        # Also consider Mod 0? No, 0+0=0, 0+0=0 is a conflict if 0 is in set. 
        # But if set has multiple 0s mod 3, e.g. 3, 6, 3+3=6 is a conflict. 
        # So mod 0 sets are NOT generally sum-free. 
        # However, we can try to build a large sum-free set from mod 0 elements by removing conflicts, 
        # but usually mod 1 or 2 are much larger and conflict-free by default.
        
        # Let's also try a greedy construction on the whole set, but seeded with mod 1/2 is a strong baseline.
        # Actually, a mix might be better? No, mixing residues creates conflicts easily.
        # The best sum-free sets in integers are often large intervals or arithmetic progressions with step 3.
        
        best_seed = mod1 if len(mod1) > len(mod2) else mod2
        
        # Start with the best modular seed
        current_S = set(best_seed)
        
        # If the seed is small, maybe a greedy approach on the rest is better?
        # Let's also run a quick greedy removal on the full set to get another baseline.
        greedy_S = set(V)
        # Simple greedy removal: remove elements involved in most conflicts
        # This is O(N^3) worst case but N is small (<=240).
        # We'll do a few iterations of conflict removal.
        time_left_initial = time_limit
        # Use a fraction of time for greedy baseline if needed? 
        # Modular seed is O(N). Greedy is heavier.
        # Let's stick to Modular Seed as primary, then refine.
        
        # Simulated Annealing Refinement
        # State: current_S (set)
        # Objective: Maximize size of current_S while keeping it sum-free.
        # Moves:
        # 1. Add an element from V \ current_S if it doesn't create conflicts.
        # 2. Swap: Remove an element from current_S, Add an element from V \ current_S.
        #    This is tricky to verify quickly.
        
        # Better approach: Local Search with "Add" and "Remove" moves.
        # Since we want to maximize size, "Add" is the most important.
        # "Remove" is only useful if it allows adding 2+ elements later? 
        # Or if we get stuck.
        
        # Let's use a simple hill climbing with restarts and randomization.
        
        def can_add(x, S_set, S_list):
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True

        def local_search(S, time_limit_for_ls):
            S = set(S)
            S_list = list(S)
            S_set = S
            improved = True
            last_improvement_time = time.time()
            
            # Pre-compute list of candidates to add (V \ S)
            candidates = list(V_set - S)
            
            while improved and (time.time() - start_time) < start_time + time_limit_for_ls:
                improved = False
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                # Try to add candidates
                new_candidates = []
                for x in candidates:
                    if can_add(x, S_set, S_list):
                        S.add(x)
                        S_list.append(x)
                        S_set.add(x)
                        improved = True
                        last_improvement_time = time.time()
                    else:
                        new_candidates.append(x)
                
                if improved:
                    candidates = new_candidates
                else:
                    # No single addition possible. Try swaps?
                    # Swap: remove a, add b.
                    # This is complex. Let's try random removal and then re-add.
                    # Or just break and restart with a different seed?
                    # For now, break local search if no single add works.
                    break
                    
            return S

        # Run multiple restarts of local search with different initial seeds
        best_result = set()
        
        # Seed 1: Best modular
        s1 = local_search(best_seed, time_limit * 0.4)
        if len(s1) > len(best_result):
            best_result = s1
            
        # Seed 2: Other modular
        s2 = local_search(mod2 if best_seed == mod1 else mod1, time_limit * 0.4)
        if len(s2) > len(best_result):
            best_result = s2
            
        # Seed 3: Greedy Removal from full set
        # Implement a fast greedy removal
        greedy_S = set(V)
        # Remove elements involved in conflicts repeatedly
        # Use numpy for speed if possible? N is small, Python sets are fine.
        # Let's do a few rounds of "remove most conflicting element"
        for _ in range(5): # Max 5 rounds to keep it fast
            conflicts, involved = get_conflicts(list(greedy_S))
            if not conflicts:
                break
            # Count conflicts per element
            conf_count = {}
            for a, b, c in conflicts:
                conf_count[a] = conf_count.get(a, 0) + 1
                conf_count[b] = conf_count.get(b, 0) + 1
                conf_count[c] = conf_count.get(c, 0) + 1
            if not conf_count:
                break
            max_conf = max(conf_count.values())
            # Remove one element with max conflicts
            to_remove = [k for k, v in conf_count.items() if v == max_conf]
            rem = random.choice(to_remove)
            greedy_S.remove(rem)
            
        s3 = local_search(greedy_S, time_limit * 0.4)
        if len(s3) > len(best_result):
            best_result = s3

        # If we have time left, try random restarts with mixed seeds?
        # Or just return the best found.
        
        return list(best_result)

    results = []
    
    # Allocate time
    # 60: 10s, 120: 30s, 240: 70s? 
    # Total 110s.
    # Let's be conservative: 10, 20, 80.
    time_allocs = {60: 10.0, 120: 20.0, 240: 80.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed
        
        # Adjust time allocation dynamically
        if size == 60:
            t_limit = min(time_allocs[60], remaining_total * 0.1)
        elif size == 120:
            t_limit = min(time_allocs[120], remaining_total * 0.2)
        else:
            t_limit = min(time_allocs[240], remaining_total * 0.7)
            
        # Ensure we don't exceed global deadline
        t_limit = min(t_limit, DEADLINE - elapsed - 5.0) # 5s safety margin
        
        if t_limit <= 0:
            # Emergency: just return a valid sum-free set quickly
            # Modular seed is instant
            mod1 = [x for x in V if x % 3 == 1]
            mod2 = [x for x in V if x % 3 == 2]
            res = mod1 if len(mod1) > len(mod2) else mod2
            results.append(res)
            continue
            
        try:
            res = solve_instance(V, t_limit)
            # Verify
            if not is_sum_free(res):
                # Fallback to modular
                mod1 = [x for x in V if x % 3 == 1]
                mod2 = [x for x in V if x % 3 == 2]
                res = mod1 if len(mod1) > len(mod2) else mod2
            results.append(res)
        except Exception as e:
            # Fallback to modular
            mod1 = [x for x in V if x % 3 == 1]
            mod2 = [x for x in V if x % 3 == 2]
            res = mod1 if len(mod1) > len(mod2) else mod2
            results.append(res)

    return results
