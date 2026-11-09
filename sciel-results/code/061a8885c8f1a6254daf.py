"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Find a maximum-size Sidon set (B_2 set) in [1, 284] that must contain
    the prescribed elements {1, 2, 5, 11}.
    
    A Sidon set has all pairwise sums a+b (a<=b) distinct.
    This is equivalent to a Golomb ruler where differences are distinct,
    but here we require sums to be distinct (which is the same condition
    for the set of all pairwise sums including doubles).
    
    Strategy:
    1. Start with the prescribed elements {1, 2, 5, 11}.
    2. Use a combination of greedy construction, local search (simulated annealing),
       and randomized restarts to find a large valid Sidon set.
    3. The known maximum without constraints is 20 (Golomb ruler length 283).
       With the prescribed elements, we aim to get close to 20.
    """
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    def valid_sidon(s):
        """Check if set s is a valid Sidon set (all pairwise sums distinct)."""
        s = sorted(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def sum_set(s):
        """Return the set of all pairwise sums for set s."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def compute_sum_counts(s):
        """Return dict mapping each pairwise sum to its count."""
        counts = defaultdict(int)
        for i in range(len(s)):
            for j in range(i, len(s)):
                counts[s[i] + s[j]] += 1
        return counts
    
    def is_valid_count(counts):
        """Check if all counts are 1 (valid Sidon)."""
        return all(c == 1 for c in counts.values())
    
    def try_add_element(s, sums, x):
        """
        Try to add element x to set s.
        Returns (success, new_sums, conflict_count)
        """
        if x in s:
            return False, sums, 0
        conflict = 0
        new_sums = set()
        for y in s:
            sm = x + y
            if sm in sums:
                conflict += 1
            else:
                new_sums.add(sm)
        sm2 = 2 * x
        if sm2 in sums:
            conflict += 1
        else:
            new_sums.add(sm2)
        if conflict == 0:
            return True, sums | new_sums, 0
        return False, sums, conflict
    
    def remove_element_sums(s, sums, x):
        """Remove element x from s and update sums. Returns new_sums."""
        new_sums = set()
        for i, a in enumerate(s):
            if a == x:
                continue
            for j, b in enumerate(s):
                if i <= j and a != x and b != x:
                    new_sums.add(a + b)
        # Simpler: recompute
        s2 = [v for v in s if v != x]
        return sum_set(s2)
    
    def greedy_construct(prescribed, rng):
        """
        Greedy construction starting from prescribed elements.
        Try to add elements in order (or randomized order) that maintain Sidon property.
        Uses O(N) conflict checking per candidate.
        """
        a = list(prescribed)
        a_set = set(a)
        sums = sum_set(a)
        
        # Try adding elements from 12 to 284
        candidates = list(range(12, 285))
        rng.shuffle(candidates)
        
        for x in candidates:
            if x in a_set:
                continue
            # O(N) conflict check
            conflict = False
            for y in a:
                if (x + y) in sums:
                    conflict = True
                    break
            if not conflict and (2 * x) in sums:
                conflict = True
            if not conflict:
                a.append(x)
                a_set.add(x)
                for y in a:
                    sums.add(x + y)
                sums.add(2 * x)
        
        return sorted(a)
    
    def local_search(s, rng, time_limit=2.0):
        """
        Local search via simulated annealing to try to grow the Sidon set.
        Uses O(N) incremental sum updates for speed.
        Operations:
        - Add a new element if possible
        - Swap an element for another (O(N) conflict check)
        - Targeted swap: find element causing most conflicts, replace it
        """
        a = list(s)
        a_set = set(a)
        sums = sum_set(a)
        best = len(a)
        best_set = list(a)
        
        start = time.time()
        temp = 10.0
        cooling = 0.9995
        
        while time.time() - start < time_limit and temp > 0.01:
            # Random operation
            op = rng.random()
            
            if op < 0.35:
                # Try to add a random element
                available = [x for x in range(1, 285) if x not in a_set]
                if not available:
                    temp *= cooling
                    continue
                x = rng.choice(available)
                # O(N) conflict check
                conflict = False
                for y in a:
                    if (x + y) in sums:
                        conflict = True
                        break
                if not conflict and (2 * x) in sums:
                    conflict = True
                if not conflict:
                    a.append(x)
                    a_set.add(x)
                    for y in a:
                        sums.add(x + y)
                    sums.add(2 * x)
                    if len(a) > best:
                        best = len(a)
                        best_set = list(a)
                temp *= cooling
                
            elif op < 0.70:
                # Try swap: remove one element, add another
                if len(a) < 2:
                    temp *= cooling
                    continue
                idx = rng.randrange(len(a))
                removed = a[idx]
                
                # Compute sums without removed element (O(N^2) but only for this operation)
                a_temp = [v for i, v in enumerate(a) if i != idx]
                sums_temp = sum_set(a_temp)
                
                available = [x for x in range(1, 285) if x not in a_set or x == removed]
                if available:
                    x = rng.choice(available)
                    if x == removed:
                        # No change
                        pass
                    else:
                        # O(N) conflict check
                        conflict = False
                        for y in a_temp:
                            if (x + y) in sums_temp:
                                conflict = True
                                break
                        if not conflict and (2 * x) in sums_temp:
                            conflict = True
                        if not conflict:
                            a[idx] = x
                            a_set.discard(removed)
                            a_set.add(x)
                            sums = sums_temp
                            for y in a:
                                sums.add(x + y)
                            sums.add(2 * x)
                            if len(a) > best:
                                best = len(a)
                                best_set = list(a)
                        else:
                            # Revert (no change since we didn't modify a)
                            pass
                temp *= cooling
                
            else:
                # Targeted swap: find element with most conflicts, try to replace it
                if len(a) < 3:
                    temp *= cooling
                    continue
                # Count conflicts for each element
                conflict_counts = {}
                for i, elem in enumerate(a):
                    a_without = [v for j, v in enumerate(a) if j != i]
                    sums_without = sum_set(a_without)
                    # Check which elements can't be added due to conflicts with elem
                    # Actually, let's just try removing elem and see if we can add more
                    pass
                
                # Simpler: try removing a random non-prescribed element and adding two
                idx = rng.randrange(len(a))
                removed = a[idx]
                a_temp = [v for i, v in enumerate(a) if i != idx]
                a_temp_set = set(a_temp)
                sums_temp = sum_set(a_temp)
                
                added = 0
                for _ in range(2):
                    available = [x for x in range(1, 285) if x not in a_temp_set]
                    if not available:
                        break
                    x = rng.choice(available)
                    conflict = False
                    for y in a_temp:
                        if (x + y) in sums_temp:
                            conflict = True
                            break
                    if not conflict and (2 * x) in sums_temp:
                        conflict = True
                    if not conflict:
                        a_temp.append(x)
                        a_temp_set.add(x)
                        for y in a_temp:
                            sums_temp.add(x + y)
                        sums_temp.add(2 * x)
                        added += 1
                
                if added >= 1 and len(a_temp) > best:
                    a = a_temp
                    a_set = set(a)
                    sums = sums_temp
                    best = len(a)
                    best_set = list(a)
                elif added == 0:
                    # Revert: keep original
                    pass
                temp *= cooling
        
        return best_set
    
    def hill_climb_grow(s, rng):
        """
        Aggressively try to grow the set by finding any valid addition.
        """
        a = list(s)
        sums = sum_set(a)
        
        # Try all possible additions
        for x in range(1, 285):
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        
        return sorted(a)
    
    # Main search
    prescribed = [1, 2, 5, 11]
    best_result = list(prescribed)
    best_size = len(prescribed)
    
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        result = greedy_construct(prescribed, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 2: Local search from best and other good solutions
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        if trial == 0:
            start_set = best_result
        else:
            start_set = greedy_construct(prescribed, rng)
        
        result = local_search(start_set, rng, time_limit=3.0)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 3: Hill climbing from best
    if time.time() < deadline - 15:
        result = hill_climb_grow(best_result, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 4: Additional local search with perturbation
    if time.time() < deadline - 10:
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            start_set = list(best_result)
            # Add some random perturbation
            if len(start_set) > 5:
                for _ in range(2):
                    idx = rng.randrange(len(start_set))
                    if start_set[idx] not in prescribed:
                        old = start_set[idx]
                        candidates = [x for x in range(1, 285) if x not in start_set]
                        if candidates:
                            start_set[idx] = rng.choice(candidates)
                            # Check validity
                            if not valid_sidon(start_set):
                                start_set[idx] = old
            
            result = local_search(start_set, rng, time_limit=2.0)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    # Phase 5: More local search with perturbation
    if time.time() < deadline - 15:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            start_set = list(best_result)
            # Add some random perturbation
            if len(start_set) > 5:
                for _ in range(3):
                    idx = rng.randrange(len(start_set))
                    if start_set[idx] not in prescribed:
                        old = start_set[idx]
                        candidates = [x for x in range(1, 285) if x not in start_set]
                        if candidates:
                            start_set[idx] = rng.choice(candidates)
                            # Check validity
                            if not valid_sidon(start_set):
                                start_set[idx] = old
            
            result = local_search(start_set, rng, time_limit=3.0)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    # Phase 6: Final hill climb
    if time.time() < deadline - 5:
        result = hill_climb_grow(best_result, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Verify final result
    assert valid_sidon(best_result), "Final result is not a valid Sidon set"
    assert set(prescribed).issubset(set(best_result)), "Missing prescribed elements"
    assert all(1 <= x <= 284 for x in best_result), "Elements out of range"
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
