import time
import random
import math

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # LCM of 1..22 for converting reciprocal sums to integers
    # 232792560
    lcm_all = 232792560
    
    # Precompute reciprocal values as integers (1/i * LCM)
    recip_int = {i: lcm_all // i for i in range(1, 23)}
    
    def check_dissociated(elems):
        """
        Check if the set of elements (1-based values 1..22)
        has all distinct reciprocal subset sums.
        Uses integer arithmetic for speed.
        Returns True if dissociated, False otherwise.
        """
        k = len(elems)
        if k == 0:
            return True

        seen = {0}
        current_sums = [0]

        for e in elems:
            v = recip_int[e]
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)

        return True

    def greedy_build(ordering):
        """
        Greedily build a dissociated set following the given ordering.
        Returns the set found and the set of sums.
        """
        S = []
        cur_sums = {0}
        for e in ordering:
            if time.time() > deadline:
                break
            v = recip_int[e]
            # Check collision: for any existing sum s, is s + v already in cur_sums?
            collision = False
            for s in cur_sums:
                if (s + v) in cur_sums:
                    collision = True
                    break
            if not collision:
                S.append(e)
                # Update sums
                new_sums = set()
                for s in cur_sums:
                    new_sums.add(s + v)
                cur_sums = cur_sums | new_sums
        return S, cur_sums

    def expand_set(S, cur_sums):
        """
        Try to add remaining elements to S.
        Returns (new_S, new_cur_sums)
        """
        remaining = [e for e in range(1, 23) if e not in S]
        # Sort remaining by value descending (smaller reciprocal first) to reduce collision risk?
        # Or ascending? Let's try descending (larger numbers first) as they have smaller reciprocals.
        remaining.sort(reverse=True)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                v = recip_int[e]
                collision = False
                for s in cur_sums:
                    if (s + v) in cur_sums:
                        collision = True
                        break
                if not collision:
                    S.append(e)
                    new_sums = set()
                    for s in cur_sums:
                        new_sums.add(s + v)
                    cur_sums = cur_sums | new_sums
                    remaining.remove(e)
                    changed = True
        return S, cur_sums

    def local_search(S):
        """
        Perform swap-based local search to improve the set size.
        """
        S = list(S)
        cur_sums = {0}
        for e in S:
            v = recip_int[e]
            new_sums = set()
            for s in cur_sums:
                new_sums.add(s + v)
            cur_sums = cur_sums | new_sums
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            remaining = [e for e in range(1, 23) if e not in S]
            
            # Try all swaps
            for i in range(len(S)):
                if time.time() > deadline - 5:
                    break
                e_remove = S[i]
                for e_add in remaining:
                    if time.time() > deadline - 5:
                        break
                    # Candidate set: replace e_remove with e_add
                    candidate = S[:i] + S[i+1:] + [e_add]
                    
                    # Check if candidate is dissociated
                    if check_dissociated(candidate):
                        # Try to expand this candidate
                        c_sums = {0}
                        for e in candidate:
                            v = recip_int[e]
                            new_sums = set()
                            for s in c_sums:
                                new_sums.add(s + v)
                            c_sums = c_sums | new_sums
                        
                        expanded, exp_sums = expand_set(candidate, c_sums)
                        
                        if len(expanded) > len(S):
                            S = expanded
                            cur_sums = exp_sums
                            improved = True
                            break
                        elif len(expanded) == len(S):
                            # Even if size is same, we might accept it if it's "better" in some way?
                            # For now, only accept if size increases to keep search focused.
                            # However, sometimes a swap of same size unlocks further additions.
                            # Let's accept same-size swaps if they allow expansion in the next iteration?
                            # To avoid cycles, we should be careful.
                            # Let's just accept if size increases.
                            pass
                if improved:
                    break
        return S

    best_set = [1]
    best_size = 1

    # Strategy 1: Greedy builds with various orderings
    orderings = [
        list(range(22, 0, -1)),       # Descending
        list(range(1, 23)),           # Ascending
        list(range(2, 23, 2)) + list(range(1, 23, 2)), # Evens then Odds
        list(range(1, 23, 2)) + list(range(2, 23, 2)), # Odds then Evens
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        S, sums = greedy_build(ordering)
        if S:
            S, sums = expand_set(S, sums)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S

    # Random orderings
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        S, sums = greedy_build(order)
        if S:
            S, sums = expand_set(S, sums)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S

    # Strategy 2: Greedy Remove
    # Start with all 22, remove elements causing collisions
    S_remove = list(range(1, 23))
    # We need a fast way to find a collision and remove one element
    # This is complex to implement efficiently in a tight loop without precomputation.
    # Instead, we can just run local search on the best set found so far.
    
    # Strategy 3: Local Search on best_set
    if time.time() < deadline - 10:
        improved_S = local_search(best_set)
        if len(improved_S) > best_size:
            best_set = improved_S
            best_size = len(best_set)

    # Additional Local Search passes with different randomizations if time permits
    if time.time() < deadline - 5:
        # Try starting from a slightly different set?
        # Or just run local search again on the current best to see if it finds a different local optimum?
        # Since local search is deterministic given the set, it will return the same result.
        # We need to randomize the order of swaps or use simulated annealing.
        
        # Let's implement a simple randomized local search
        S_rs = list(best_set)
        # Shuffle the set to break symmetry in swap order? No, set is unordered.
        # Instead, randomize the order of 'remaining' elements in the swap loop.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            remaining = [e for e in range(1, 23) if e not in S_rs]
            random.shuffle(remaining)
            
            for i in range(len(S_rs)):
                if time.time() > deadline - 2:
                    break
                e_remove = S_rs[i]
                for e_add in remaining:
                    if time.time() > deadline - 2:
                        break
                    candidate = S_rs[:i] + S_rs[i+1:] + [e_add]
                    if check_dissociated(candidate):
                        c_sums = {0}
                        for e in candidate:
                            v = recip_int[e]
                            new_sums = set()
                            for s in c_sums:
                                new_sums.add(s + v)
                            c_sums = c_sums | new_sums
                        
                        expanded, exp_sums = expand_set(candidate, c_sums)
                        if len(expanded) > len(S_rs):
                            S_rs = expanded
                            improved = True
                            break
                if improved:
                    break
        
        if len(S_rs) > best_size:
            best_set = S_rs
            best_size = len(best_set)

    # Final verification
    if best_set and check_dissociated(best_set):
        return sorted(best_set)
    else:
        return [1]

# Run construct if called directly for testing
if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Result: {result}, Size: {len(result)}, Time: {end-start:.2f}s")
