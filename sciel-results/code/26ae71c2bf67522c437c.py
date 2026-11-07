import time
import random

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    n_primes = len(primes)
    
    # Precompute exponent vectors for all candidates
    def factorize(n):
        v = [0] * n_primes
        m = n
        for i, p in enumerate(primes):
            while m % p == 0:
                v[i] += 1
                m //= p
        return tuple(v)
    
    vecs = {n: factorize(n) for n in candidates}
    
    # Direct construction using primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def build_sums(current_set):
        """Build the set of all subset-sum exponent vectors for current_set."""
        sums = {tuple([0]*n_primes)}
        for x in current_set:
            v = vecs[x]
            new_sums = set()
            for s in sums:
                new_sums.add(tuple(s[i] + v[i] for i in range(n_primes)))
            sums.update(new_sums)
        return sums
    
    def is_valid_addition(current_sums, candidate):
        """Check if adding candidate keeps all subset sums distinct."""
        v = vecs[candidate]
        for s in current_sums:
            sv = tuple(s[i] + v[i] for i in range(n_primes))
            if sv in current_sums:
                return False
        return True
    
    def try_add_to_set(current_set, current_sums, candidate):
        """Try to add candidate. Returns (success, new_set, new_sums) or (False, None, None)."""
        if is_valid_addition(current_sums, candidate):
            v = vecs[candidate]
            new_set = current_set + [candidate]
            new_sums = set()
            for s in current_sums:
                new_sums.add(tuple(s[i] + v[i] for i in range(n_primes)))
            new_sums_combined = current_sums | new_sums
            return (True, new_set, new_sums_combined)
        return (False, None, None)
    
    def greedy_from_order(ordered_candidates):
        """Greedy construction from a given ordering of candidates."""
        current_set = []
        current_sums = {tuple([0]*n_primes)}
        for x in ordered_candidates:
            success, new_set, new_sums = try_add_to_set(current_set, current_sums, x)
            if success:
                current_set = new_set
                current_sums = new_sums
        return current_set
    
    def local_search_add(current_set):
        """Try to add more elements to current_set via local search."""
        current_sums = build_sums(current_set)
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in candidates:
                if x in current_set:
                    continue
                success, new_set, new_sums = try_add_to_set(current_set, current_sums, x)
                if success:
                    current_set = new_set
                    current_sums = new_sums
                    added = True
                    break
        return current_set
    
    def local_search_swap(current_set):
        """Try to improve current_set by swapping one element for another."""
        current_sums = build_sums(current_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, elem in enumerate(current_set):
                remaining = current_set[:i] + current_set[i+1:]
                remaining_sums = build_sums(remaining)
                for x in candidates:
                    if x in remaining:
                        continue
                    success, new_set, new_sums = try_add_to_set(remaining, remaining_sums, x)
                    if success:
                        current_set = new_set
                        current_sums = new_sums
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    def local_search_double_swap(current_set):
        """Try to improve by removing two elements and adding two others."""
        current_sums = build_sums(current_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(current_set)
            for i in range(n):
                for j in range(i+1, n):
                    remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                    remaining_sums = build_sums(remaining)
                    # Try adding pairs of candidates
                    for x in candidates:
                        if x in remaining:
                            continue
                        success1, new_set1, new_sums1 = try_add_to_set(remaining, remaining_sums, x)
                        if success1:
                            for y in candidates:
                                if y in new_set1 or y == x:
                                    continue
                                success2, new_set2, new_sums2 = try_add_to_set(new_set1, new_sums1, y)
                                if success2 and len(new_set2) > len(current_set):
                                    current_set = new_set2
                                    current_sums = new_sums2
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
        return current_set
    
    random.seed(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    for restart in range(500):
        if time.time() > deadline - 2:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order(shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 18:
                break
    
    # Phase 2: Local search - try to add elements to best_set
    if time.time() < deadline - 3:
        current_set = local_search_add(best_set[:])
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 3: More aggressive search with different heuristics
    if time.time() < deadline - 5:
        sort_keys = [
            lambda x: x,
            lambda x: -x,
            lambda x: sum(vecs[x]),
            lambda x: -sum(vecs[x]),
            lambda x: max(vecs[x], default=0),
            lambda x: -max(vecs[x], default=0),
            lambda x: sum(v*v for v in vecs[x]),
            lambda x: -sum(v*v for v in vecs[x]),
        ]
        for sort_key in sort_keys:
            if time.time() > deadline - 2:
                break
            sorted_candidates = sorted(candidates, key=sort_key)
            current_set = greedy_from_order(sorted_candidates)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Phase 4: Swap-based local search on best set
    if time.time() < deadline - 4:
        current_set = local_search_swap(best_set[:])
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 5: Double swap local search
    if time.time() < deadline - 3:
        current_set = local_search_double_swap(best_set[:])
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 6: Final random restarts with local search polish
    while time.time() < deadline - 1:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        current_set = greedy_from_order(shuffled)
        if time.time() < deadline - 1:
            current_set = local_search_add(current_set)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
