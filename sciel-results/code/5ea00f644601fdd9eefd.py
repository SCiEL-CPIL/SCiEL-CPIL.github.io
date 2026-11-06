import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 145.0  # Safety margin

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

    def can_add(x, S, Sset):
        if 2 * x in Sset:
            return False
        for a in S:
            if (a + x) in Sset:
                return False
            if (a - x) in Sset:
                return False
        return True

    def try_add_all(S, V, Sset):
        """Greedily add any element from V that can be added to S. Returns new S, Sset, and list of added."""
        added = []
        remaining = [x for x in V if x not in Sset]
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for x in remaining:
                if can_add(x, S, Sset):
                    S.add(x)
                    Sset.add(x)
                    added.append(x)
                    changed = True
                else:
                    still_remaining.append(x)
            remaining = still_remaining
        return S, Sset, added

    def local_search(S_init, V, time_limit, start_time, rng):
        S = set(S_init)
        Sset = set(S)
        Vset = set(V)
        
        # Phase 1: Pure Addition
        S, Sset, _ = try_add_all(S, V, Sset)
        
        best_S = set(S)
        best_size = len(S)
        
        # Phase 2: 1-for-2 and 2-for-3 moves
        # We will iterate through possible removals and see if we can add more than removed.
        
        while (time.time() - start_time) < time_limit:
            improved = False
            
            # Try 1-for-k moves
            # For each u in S, remove u, then try to add as many as possible from V\S
            # If net gain > 0, accept.
            
            S_list = list(S)
            rng.shuffle(S_list)
            
            best_move = None
            best_gain = 0
            best_new_S = None
            best_new_Sset = None
            
            for u in S_list:
                if (time.time() - start_time) > time_limit:
                    break
                
                S_tentative = S - {u}
                Sset_tentative = Sset - {u}
                
                # Try to add elements from V \ S_tentative
                # We only care if we can add at least 2 elements to get net gain +1
                # Or 1 element to get net gain 0 (useless for size, but maybe for structure? No, we maximize size)
                # So we need to add >= 2 elements.
                
                # Optimization: Only check candidates that are "likely" to be addable.
                # But for N=240, checking all is fine.
                
                added_count = 0
                added_elements = []
                
                # To make this faster, we can just count how many can be added greedily?
                # No, greedy addition might block others.
                # However, for a 1-for-2 move, we just need to find *any* 2 elements that can be added.
                # If we find 2, we can add them.
                
                # Let's collect all addable elements to S_tentative
                addable = []
                for x in V:
                    if x in Sset_tentative:
                        continue
                    if can_add(x, S_tentative, Sset_tentative):
                        addable.append(x)
                
                if len(addable) >= 2:
                    # We can add at least 2.
                    # Let's try to add as many as possible to maximize the gain.
                    # But we must ensure they don't conflict with each other.
                    # We can just greedily add them.
                    
                    S_new = set(S_tentative)
                    Sset_new = set(Sset_tentative)
                    for x in addable:
                        if can_add(x, S_new, Sset_new):
                            S_new.add(x)
                            Sset_new.add(x)
                    
                    gain = len(S_new) - len(S)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = u
                        best_new_S = S_new
                        best_new_Sset = Sset_new
                        
            if best_gain > 0:
                S = best_new_S
                Sset = best_new_Sset
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                
                # After a move, we might be able to add more pure elements
                S, Sset, _ = try_add_all(S, V, Sset)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            
            # If no 1-for-k improvement, try 2-for-3?
            # 2-for-3 is much more expensive O(N^2 * N).
            # For N=240, N^2 = 57k. 57k * 240 = 13M ops. Might be too slow in Python per iteration.
            # We will only try 2-for-3 if we are stuck for a while or if N is small.
            
            if not improved and len(V) <= 120:
                # Try 2-for-3 moves for smaller instances
                S_list = list(S)
                n = len(S_list)
                found_improvement = False
                
                for i in range(n):
                    if (time.time() - start_time) > time_limit:
                        break
                    u1 = S_list[i]
                    for j in range(i+1, n):
                        if (time.time() - start_time) > time_limit:
                            break
                        u2 = S_list[j]
                        
                        S_tentative = S - {u1, u2}
                        Sset_tentative = Sset - {u1, u2}
                        
                        addable = []
                        for x in V:
                            if x in Sset_tentative:
                                continue
                            if can_add(x, S_tentative, Sset_tentative):
                                addable.append(x)
                        
                        if len(addable) >= 3:
                            S_new = set(S_tentative)
                            Sset_new = set(Sset_tentative)
                            for x in addable:
                                if can_add(x, S_new, Sset_new):
                                    S_new.add(x)
                                    Sset_new.add(x)
                            
                            gain = len(S_new) - len(S)
                            if gain > 0:
                                S = S_new
                                Sset = Sset_new
                                found_improvement = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                S, Sset, _ = try_add_all(S, V, Sset)
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                break # Restart loop with new state
                    if found_improvement:
                        break
                
                if found_improvement:
                    improved = True
            
            if not improved:
                # Stuck. Try random restart or perturbation?
                # Perturbation: Remove 2 random elements, try to add.
                if len(S) > 2:
                    # Pick 2 random elements to remove
                    S_list = list(S)
                    rng.shuffle(S_list)
                    u1, u2 = S_list[0], S_list[1]
                    S_tentative = S - {u1, u2}
                    Sset_tentative = Sset - {u1, u2}
                    
                    S_new, Sset_new, _ = try_add_all(S_tentative, V, Sset_tentative)
                    
                    if len(S_new) > len(S):
                        S = S_new
                        Sset = Sset_new
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                    else:
                        # Revert
                        pass
                else:
                    break # If set is tiny and stuck, stop? Or just break to save time.
        
        return best_S

    def solve_instance(V, time_limit, start_time):
        n = len(V)
        best_set = set()
        best_size = 0
        
        # 1. Modular Initializations
        candidates = []
        span = max(V) - min(V)
        
        mods = set()
        mods.update(range(5, 100))
        for factor in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
            m = int(span * factor)
            if m > 5:
                mods.add(m)
                mods.update([m-1, m+1])
        
        for M in mods:
            L = M // 3 + 1
            R = (2 * M) // 3
            S = set()
            for x in V:
                r = x % M
                if L <= r <= R:
                    S.add(x)
            if len(S) > 0:
                candidates.append(S)
        
        # 2. Sign/Parity Initializations
        pos = set(x for x in V if x > 0)
        neg = set(x for x in V if x < 0)
        odds = set(x for x in V if x % 2 != 0)
        candidates.append(pos)
        candidates.append(neg)
        candidates.append(odds)
        
        # 3. Random Greedy Initializations
        rng = random.Random(42)
        for _ in range(20):
            S = set()
            V_shuffled = V[:]
            rng.shuffle(V_shuffled)
            for x in V_shuffled:
                if can_add(x, S, S):
                    S.add(x)
            candidates.append(S)
        
        # Deduplicate
        unique_candidates = []
        seen = set()
        for S in candidates:
            key = frozenset(S)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(S)
        
        # Sort by size descending
        unique_candidates.sort(key=len, reverse=True)
        
        # Run local search on top candidates
        # Allocate time based on remaining time
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if remaining_time < 1:
            # Just return the best initial
            best_set = unique_candidates[0] if unique_candidates else set()
            return sorted(best_set)
        
        # We will run local search on the top 5 candidates
        # Distribute time
        num_to_try = min(5, len(unique_candidates))
        time_per_candidate = remaining_time / num_to_try
        
        for i in range(num_to_try):
            if (time.time() - start_time) > time_limit:
                break
            
            S_init = unique_candidates[i]
            S_res = local_search(S_init, V, time_per_candidate, start_time, rng)
            
            if len(S_res) > best_size:
                best_size = len(S_res)
                best_set = S_res
        
        return sorted(best_set)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback
            odds = [x for x in V if x % 2 != 0]
            results.append(sorted(odds))
            continue
        
        instances_left = 3 - len(results)
        time_per_instance = (remaining_total / instances_left) - 2
        if time_per_instance < 1:
            time_per_instance = 1
            
        S_best = solve_instance(V, time_per_instance, start)
        
        if not is_sum_free(S_best):
            odds = [x for x in V if x % 2 != 0]
            S_best = sorted(odds)
            
        results.append(S_best)
        
    return results

if __name__ == "__main__":
    res = construct()
    print(res)
