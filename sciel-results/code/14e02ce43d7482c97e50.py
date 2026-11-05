"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0  # safety margin before 30s limit
    start = time.time()
    
    # We'll search for a minimal-size maximal Sidon set.
    # Strategy: 
    # 1. Build a Sidon set greedily in various orders
    # 2. Try to remove elements while maintaining maximality
    # 3. Use randomized restarts to explore different constructions
    
    best_size = N + 1
    best_set = None
    
    # Bitset-based maximality check: much faster than individual element checks
    # sums_mask: bitset where bit i is set if sum i is in the sum set
    # For element x to be addable, we need:
    #   - 2x not in sums
    #   - x+a not in sums for all a in S
    # We can precompute for each x whether it's addable using bit operations
    
    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums
    
    def is_maximal_fast(S, sums=None):
        """Fast maximality check using precomputed sum set."""
        if sums is None:
            sums = get_sums(S)
        S_set = set(S)
        sums_set = set(sums)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added: 2x not in sums, and x+a not in sums for all a in S
            if 2 * x in sums_set:
                continue
            can_add = True
            for a in S:
                if x + a in sums_set:
                    can_add = False
                    break
            if can_add:
                return False
        return True
    
    def try_remove_and_check_maximal(S, idx, sums_cache=None):
        """Try removing S[idx]. If the result is still maximal, return the new set."""
        new_S = S[:idx] + S[idx+1:]
        if not new_S:
            return None
        if is_maximal_fast(new_S):
            return new_S
        return None
    
    # Phase 1: Greedy constructions with different orderings
    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            # Check if x can be added
            ok = True
            if 2 * x in sums:
                ok = False
            else:
                for a in S:
                    if x + a in sums:
                        ok = False
                        break
            if ok:
                # Also check that all new sums are distinct from each other and from existing
                new_sums = [2*x] + [x + a for a in S]
                if len(set(new_sums)) != len(new_sums):
                    ok = False
                else:
                    for ns in new_sums:
                        if ns in sums:
                            ok = False
                            break
            if ok:
                S.append(x)
                sums.add(2*x)
                for a in S[:-1]:  # S[:-1] is S without the newly added x
                    sums.add(x + a)
        return S
    
    # Try ascending greedy (baseline)
    S0 = greedy_construct(range(1, N + 1))
    S0.sort()
    # Prune: try to remove elements while maintaining maximality
    improved = True
    while improved:
        improved = False
        for i in range(len(S0) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_S = try_remove_and_check_maximal(S0, i)
            if new_S is not None:
                S0 = new_S
                improved = True
                if len(S0) < best_size:
                    best_size = len(S0)
                    best_set = list(S0)
        if time.time() > deadline:
            break
    
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)
    
    # Phase 2: Randomized greedy constructions
    rng = random.Random(42)
    
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = greedy_construct(order)
        S.sort()
        
        # Prune
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    break
                new_S = try_remove_and_check_maximal(S, i)
                if new_S is not None:
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = list(S)
            if time.time() > deadline:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Occasionally try a local search from current best
        if trial % 10 == 0 and best_set is not None:
            S_local = list(best_set)
            # Try swapping: remove one, add another
            for remove_idx in range(len(S_local)):
                if time.time() > deadline:
                    break
                S_removed = S_local[:remove_idx] + S_local[remove_idx+1:]
                S_removed_set = set(S_removed)
                # Try to find an element to add back that keeps it Sidon
                # But we want to keep it maximal, so we need to add something
                # Actually, for local search, let's just try random additions after removal
                # This is complex; skip for now.
    
    # Phase 3: Try to further optimize by swapping elements
    if best_set is not None and time.time() < deadline - 5:
        S = list(best_set)
        improved = True
        max_iterations = 1000
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            # Try removing each element and see if we can add a different one
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                S_without = S[:i] + S[i+1:]
                if not S_without:
                    continue
                sums_without = get_sums(S_without)
                S_without_set = set(S_without)
                # Try to find an element x not in S_without that can be added
                # and results in a maximal set
                for x in range(1, N + 1):
                    if x in S_without_set:
                        continue
                    # Check if x can be added to S_without
                    ok = True
                    if 2 * x in sums_without:
                        ok = False
                    else:
                        for a in S_without:
                            if x + a in sums_without:
                                ok = False
                                break
                    if not ok:
                        continue
                    new_S = sorted(S_without + [x])
                    # Check if new_S is maximal
                    if is_maximal_fast(new_S):
                        S = new_S
                        improved = True
                        if len(S) < best_size:
                            best_size = len(S)
                            best_set = list(S)
                        break
    
    if best_set is None:
        # Fallback: just return the ascending greedy
        S0 = greedy_construct(range(1, N + 1))
        S0.sort()
        improved = True
        while improved:
            improved = False
            for i in range(len(S0) - 1, -1, -1):
                new_S = try_remove_and_check_maximal(S0, i)
                if new_S is not None:
                    S0 = new_S
                    improved = True
        best_set = S0
    
    # Final verification
    # Ensure it's a valid Sidon set
    S = best_set
    S_sorted = sorted(S)
    # Check Sidon
    seen_sums = set()
    for i in range(len(S_sorted)):
        for j in range(i, len(S_sorted)):
            s = S_sorted[i] + S_sorted[j]
            if s in seen_sums:
                # Should not happen, but if it does, fall back
                S0 = greedy_construct(range(1, N + 1))
                S0.sort()
                return S0
            seen_sums.add(s)
    # Check maximality
    if not is_maximal_fast(S):
        # Try to repair: add elements that can be added
        changed = True
        while changed:
            changed = False
            sums = get_sums(S)
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                ok = True
                if 2 * x in sums:
                    ok = False
                else:
                    for a in S:
                        if x + a in sums:
                            ok = False
                            break
                if ok:
                    S.append(x)
                    S_set.add(x)
                    sums.add(2*x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)
        best_set = S
    
    return best_set

# EVOLVE-BLOCK-END
