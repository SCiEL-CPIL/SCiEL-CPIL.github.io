import time
import random
import bisect
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin (limit is 150s)

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

    def ensure_sum_free(S, V, time_limit, start_time):
        """
        Ensure S is a sum-free subset of V.
        If S is already sum-free, return it.
        Otherwise, greedily remove elements from S until it is sum-free.
        """
        S = set(S)
        if is_sum_free(S):
            return S
        
        # Repeatedly find and remove conflicting elements
        while time.time() - start_time < time_limit:
            S_list = sorted(S)
            Sset = set(S_list)
            n = len(S_list)
            
            # Find all conflicts and count them
            conflict_count = {v: 0 for v in S_list}
            for i in range(n):
                a = S_list[i]
                for j in range(i, n):
                    b = S_list[j]
                    s = a + b
                    if s in Sset:
                        conflict_count[a] += 1
                        if b != a:
                            conflict_count[b] += 1
                        conflict_count[s] += 1
            
            if not any(conflict_count.values()):
                break
            
            # Remove the element with the most conflicts
            max_elem = max(conflict_count, key=conflict_count.get)
            S.discard(max_elem)
        
        # Final check - if still not sum-free, do a simple greedy rebuild
        if not is_sum_free(S):
            S_new = set()
            for x in sorted(S, key=abs):
                if is_sum_free(S_new | {x}):
                    S_new.add(x)
            S = S_new
        
        return S

    def conflicts_count(S):
        """Return dict of conflict counts per element (optimized)."""
        Sset = set(S)
        n = len(S)
        conflicts = {v: 0 for v in S}
        S_list = sorted(S)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                s = a + b
                if s in Sset:
                    conflicts[a] += 1
                    if b != a:
                        conflicts[b] += 1
                    conflicts[s] += 1
        return conflicts

    def can_add(x, Sset, S):
        """Check if x can be added to S without creating a sum-free violation."""
        for a in S:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def try_swap(S, Sset, removed, time_limit, start_time):
        """Try to swap: remove one element from S, add one or more from removed."""
        S_list = sorted(S)
        n = len(S)
        if n == 0 or not removed:
            return S, removed, False
        
        improved = False
        # Try removing each element and see if we can add more
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            a = S_list[i]
            # Tentatively remove a
            S_tentative = set(S_list[:i] + S_list[i+1:])
            Sset_tentative = set(S_tentative)
            
            # Try to add elements from removed
            added = []
            for x in removed:
                if time.time() - start_time > time_limit:
                    break
                if can_add(x, Sset_tentative, S_tentative):
                    S_tentative.add(x)
                    Sset_tentative.add(x)
                    added.append(x)
            
            # If we added more than we removed, accept the swap
            if len(added) > 1:  # removed 1, added >1
                S = S_tentative
                Sset = Sset_tentative
                removed = [x for x in removed if x not in added]
                improved = True
                # Re-sort for next iteration
                S_list = sorted(S)
                n = len(S)
        return S, removed, improved

    def try_double_swap(S, Sset, removed, time_limit, start_time):
        """Try to swap: remove two elements from S, add multiple from removed."""
        S_list = sorted(S)
        n = len(S)
        if n < 2 or not removed:
            return S, removed, False
        
        improved = False
        # Try removing pairs of elements
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, n):
                if time.time() - start_time > time_limit:
                    break
                a = S_list[i]
                b = S_list[j]
                # Tentatively remove a and b
                S_tentative = set(S_list[:i] + S_list[i+1:j] + S_list[j+1:])
                Sset_tentative = set(S_tentative)
                
                # Try to add elements from removed
                added = []
                for x in removed:
                    if time.time() - start_time > time_limit:
                        break
                    if can_add(x, Sset_tentative, S_tentative):
                        S_tentative.add(x)
                        Sset_tentative.add(x)
                        added.append(x)
                
                # If we added more than we removed, accept the swap
                if len(added) > 2:  # removed 2, added >2
                    S = S_tentative
                    Sset = Sset_tentative
                    removed = [x for x in removed if x not in added]
                    improved = True
                    # Re-sort for next iteration
                    S_list = sorted(S)
                    n = len(S)
                    break
        return S, removed, improved

    def greedy_remove(V, time_limit, start_time):
        """Remove elements with most conflicts until sum-free."""
        S = set(V)
        while time.time() - start_time < time_limit:
            if is_sum_free(S):
                break
            S_list = sorted(S)
            conflicts = conflicts_count(S_list)
            if not any(conflicts.values()):
                break
            # Remove the element with the most conflicts
            max_elem = max(conflicts, key=conflicts.get)
            S.discard(max_elem)
        return S

    def hill_climb(S, removed, time_limit, start_time):
        """Try to add back removed elements, with swap-based improvements."""
        S = set(S)
        Sset = set(S)
        removed = list(removed)
        improved = True
        iterations = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            iterations += 1
            still_removed = []
            random.shuffle(removed)
            for x in removed:
                if (time.time() - start_time) > time_limit:
                    still_removed.append(x)
                    continue
                if can_add(x, Sset, S):
                    S.add(x)
                    Sset.add(x)
                    improved = True
                else:
                    still_removed.append(x)
            removed = still_removed
            
            # Try swap-based improvements
            if removed and (time.time() - start_time) < time_limit:
                S, removed, swap_improved = try_swap(S, Sset, removed, time_limit, start_time)
                Sset = set(S)
                improved = improved or swap_improved
                
                # Try double swap if still stuck
                if removed and not improved and (time.time() - start_time) < time_limit:
                    S, removed, double_improved = try_double_swap(S, Sset, removed, time_limit, start_time)
                    Sset = set(S)
                    improved = improved or double_improved
        
        return S, removed

    def modular_seeded(V, time_limit, start_time):
        """
        DIRECT construction: use modular sum-free intervals.
        For a set of integers, numbers in (n/3, 2n/3] mod n are sum-free.
        We apply this to the range of V and use it as a seed, then optimize.
        """
        n = len(V)
        min_v = min(V)
        max_v = max(V)
        
        # Try multiple modular constructions with different moduli
        best_S = set()
        best_removed = list(V)
        
        # Construction 1: modular interval (n/3, 2n/3]
        range_size = max_v - min_v + 1
        candidates_mods = list(set([n, 2*n, 3*n, 4*n, 5*n, range_size, range_size // 2, range_size // 3, range_size // 4]))
        
        for M in candidates_mods:
            if M <= 0:
                continue
            lower = M // 3 + 1
            upper = (2 * M) // 3
            S = set()
            for x in V:
                r = (x - min_v) % M
                if lower <= r <= upper:
                    S.add(x)
            if len(S) > len(best_S):
                best_S = set(S)
                best_removed = [x for x in V if x not in S]
        
        # Construction 2: sign-based (all positive or all negative)
        pos = [x for x in V if x > 0]
        neg = [x for x in V if x < 0]
        if len(pos) > len(best_S):
            best_S = set(pos)
            best_removed = neg
        if len(neg) > len(best_S):
            best_S = set(neg)
            best_removed = pos
        
        # Construction 3: parity (odd numbers are sum-free)
        odds = [x for x in V if x % 2 != 0]
        if len(odds) > len(best_S):
            best_S = set(odds)
            best_removed = [x for x in V if x % 2 == 0]
        
        # Construction 4: top/middle/bottom thirds by value
        S_sorted = sorted(V)
        third = len(S_sorted) // 3
        top_third = set(S_sorted[2*third:])
        mid_third = set(S_sorted[third:2*third])
        if len(top_third) > len(best_S):
            best_S = set(top_third)
            best_removed = [x for x in V if x not in best_S]
        if len(mid_third) > len(best_S):
            best_S = set(mid_third)
            best_removed = [x for x in V if x not in best_S]
        
        # Now optimize from the best seed
        # First ensure sum-free
        if not is_sum_free(best_S):
            best_S = greedy_remove(list(best_S), time_limit * 0.3, start_time)
            best_removed = [x for x in V if x not in best_S]
        
        # Hill climb to add more
        if (time.time() - start_time) < time_limit:
            best_S, best_removed = hill_climb(best_S, best_removed, time_limit * 0.7, start_time)
        
        return best_S

    def greedy_all(V, time_limit, start_time):
        """Start from all, greedily remove, then hill climb."""
        S = greedy_remove(V, time_limit * 0.4, start_time)
        removed = [x for x in V if x not in S]
        if (time.time() - start_time) < time_limit:
            S, removed = hill_climb(S, removed, time_limit * 0.6, start_time)
        return S

    def local_search_polish(S, removed, time_limit, start_time):
        """Final high-precision polish: try all single and double swaps systematically."""
        S = set(S)
        Sset = set(S)
        removed = list(removed)
        if not removed:
            return S, removed
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            S_list = sorted(S)
            n = len(S_list)
            
            # Single swaps: remove 1, try to add multiple
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                a = S_list[i]
                S_tent = set(S_list[:i] + S_list[i+1:])
                Sset_tent = set(S_tent)
                added = []
                for x in removed:
                    if can_add(x, Sset_tent, S_tent):
                        S_tent.add(x)
                        Sset_tent.add(x)
                        added.append(x)
                if len(added) > 1:
                    S = S_tent
                    Sset = Sset_tent
                    removed = [x for x in removed if x not in added]
                    improved = True
                    S_list = sorted(S)
                    n = len(S_list)
            
            # Double swaps: remove 2, try to add 3+
            if not improved and n >= 2 and time.time() - start_time < time_limit:
                for i in range(n):
                    if time.time() - start_time > time_limit:
                        break
                    for j in range(i+1, n):
                        if time.time() - start_time > time_limit:
                            break
                        S_tent = set(S_list[:i] + S_list[i+1:j] + S_list[j+1:])
                        Sset_tent = set(S_tent)
                        added = []
                        for x in removed:
                            if can_add(x, Sset_tent, S_tent):
                                S_tent.add(x)
                                Sset_tent.add(x)
                                added.append(x)
                        if len(added) > 2:
                            S = S_tent
                            Sset = Sset_tent
                            removed = [x for x in removed if x not in added]
                            improved = True
                            S_list = sorted(S)
                            n = len(S_list)
                            break
        return S, removed

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback: Greedy construction from scratch to ensure validity
            S_fallback = set()
            for x in V:
                if can_add(x, set(S_fallback), S_fallback):
                    S_fallback.add(x)
            results.append(sorted(S_fallback))
            continue
        
        # Allocate time per instance: give more time to larger instances
        instances_left = 3 - len(results)
        # Weight: larger instances get more time
        weights = {60: 1.0, 120: 1.5, 240: 2.5}
        w = weights.get(size, 1.0)
        total_weight = sum(weights.get(s, 1.0) for s in [60, 120, 240][len(results):])
        time_per_instance = (remaining_total - 3) * (w / total_weight)
        
        best_size = 0
        best_set = set()
        
        # Strategy 1: modular seeded
        t1 = time_per_instance * 0.45
        S1 = modular_seeded(V, t1, start)
        # Safety check
        if is_sum_free(S1):
            if len(S1) > best_size:
                best_size = len(S1)
                best_set = set(S1)
        else:
            # Repair if necessary
            S1_fixed = ensure_sum_free(S1, V, 1.0, time.time())
            if len(S1_fixed) > best_size:
                best_size = len(S1_fixed)
                best_set = set(S1_fixed)
        
        # Strategy 2: greedy remove
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t2 = min(time_per_instance * 0.35, DEADLINE - elapsed - 3)
            S2 = greedy_all(V, t2, start)
            if is_sum_free(S2):
                if len(S2) > best_size:
                    best_size = len(S2)
                    best_set = set(S2)
            else:
                S2_fixed = ensure_sum_free(S2, V, 1.0, time.time())
                if len(S2_fixed) > best_size:
                    best_size = len(S2_fixed)
                    best_set = set(S2_fixed)
        
        # Strategy 3: random restarts with different seeds
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t3 = min(time_per_instance * 0.25, DEADLINE - elapsed - 3)
            num_trials = min(5, max(2, int(t3 / 3)))
            for trial in range(num_trials):
                if DEADLINE - (time.time() - start) < 3:
                    break
                rng = random.Random(seed + trial * 1000 + 42)
                # Random subset seed of varying sizes
                sample_size = max(5, size // (3 + trial % 3))
                sample = set(rng.sample(V, sample_size))
                if not is_sum_free(sample):
                    sample = greedy_remove(list(sample), min(2.0, t3 / (num_trials * 2.0)), start)
                removed = [x for x in V if x not in sample]
                sample, removed = hill_climb(sample, removed, t3 / (num_trials + 1.0), start)
                
                if is_sum_free(sample):
                    if len(sample) > best_size:
                        best_size = len(sample)
                        best_set = set(sample)
                else:
                    sample_fixed = ensure_sum_free(sample, V, 1.0, time.time())
                    if len(sample_fixed) > best_size:
                        best_size = len(sample_fixed)
                        best_set = set(sample_fixed)
        
        # Strategy 4: Final polish - local search on the best set
        elapsed = time.time() - start
        if DEADLINE - elapsed > 2 and best_set:
            t4 = min(3.0, DEADLINE - elapsed - 1)
            removed_final = [x for x in V if x not in best_set]
            if removed_final:
                best_set, removed_final = local_search_polish(set(best_set), removed_final, t4, start)
                if is_sum_free(best_set):
                    if len(best_set) > best_size:
                        best_size = len(best_set)
                else:
                    best_set_fixed = ensure_sum_free(best_set, V, 1.0, time.time())
                    if len(best_set_fixed) > best_size:
                        best_size = len(best_set_fixed)
                        best_set = best_set_fixed
        
        # Final safety check before appending
        if not is_sum_free(best_set):
            # Last resort: greedy add from scratch
            S_final = set()
            for x in V:
                if can_add(x, set(S_final), S_final):
                    S_final.add(x)
            best_set = S_final
            
        results.append(sorted(best_set))
    
    return results
