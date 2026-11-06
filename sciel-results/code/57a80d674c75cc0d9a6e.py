import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    # Generate all subset sums of reciprocals for a given set
    def get_reachable_sums(S):
        reachable = {Fraction(0)}
        for x in S:
            target = Fraction(1, x)
            new_reachable = set(reachable)
            for s in reachable:
                new_reachable.add(s + target)
            reachable = new_reachable
        return reachable
    
    # Check if a set is valid (no element's reciprocal is a sum of others' reciprocals)
    def is_valid(S):
        if not S:
            return False
        S_set = set(S)
        for k in S_set:
            target = Fraction(1, k)
            others = [x for x in S_set if x != k]
            # Check if target can be formed from subset of others
            # Use meet-in-the-middle for efficiency
            if len(others) <= 20:
                reachable = {Fraction(0)}
                for x in others:
                    t = Fraction(1, x)
                    new_reachable = set(reachable)
                    for s in reachable:
                        new_reachable.add(s + t)
                    reachable = new_reachable
                if target in reachable:
                    return False
            else:
                # For larger sets, use a more efficient check
                # We can use the fact that we only need to check if target is reachable
                # Build reachable sums incrementally
                reachable = {Fraction(0)}
                for x in others:
                    t = Fraction(1, x)
                    new_reachable = set(reachable)
                    for s in reachable:
                        ns = s + t
                        if ns <= target:  # Only keep sums <= target for efficiency
                            new_reachable.add(ns)
                        elif ns == target:
                            return False
                    reachable = new_reachable
                if target in reachable:
                    return False
        return True
    
    # Find a conflicting subset: given S and target k, find subset of S\{k} summing to 1/k
    def find_conflict(S, k):
        others = [x for x in S if x != k]
        target = Fraction(1, k)
        # Simple backtracking
        n = len(others)
        if n > 25:
            return None  # Too large for simple search
        
        # Meet in the middle
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        def subset_sums(arr):
            sums = {Fraction(0): []}
            for i, x in enumerate(arr):
                t = Fraction(1, x)
                new_sums = {}
                for s, idxs in sums.items():
                    ns = s + t
                    if ns <= target:
                        new_sums[ns] = idxs + [i]
                for s, idxs in sums.items():
                    if s not in new_sums:
                        new_sums[s] = idxs
                sums = new_sums
            return sums
        
        left_sums = subset_sums(left)
        right_sums = subset_sums(right)
        
        for ls, lidx in left_sums.items():
            if ls > target:
                continue
            needed = target - ls
            if needed in right_sums:
                ridx = right_sums[needed]
                return [left[i] for i in lidx] + [right[i] for i in ridx]
        return None
    
    # Greedy construction (decreasing order)
    def greedy_construct():
        S = []
        reachable = {Fraction(0)}
        for k in range(24, 0, -1):
            target = Fraction(1, k)
            if target in reachable:
                continue
            S.append(k)
            new_reachable = set(reachable)
            for s in reachable:
                new_reachable.add(s + target)
            reachable = new_reachable
        return S
    
    # Local search / repair
    def local_search(S, time_budget):
        S = list(S)
        best_S = S[:]
        best_len = len(S)
        
        # Get current reachable sums for S
        reachable = get_reachable_sums(S)
        
        # Rejected elements
        all_elements = set(range(1, 25))
        rejected = all_elements - set(S)
        
        while time.time() < start_time + time_budget:
            improved = False
            
            # Try to add each rejected element
            for r in sorted(rejected, key=lambda x: -x):
                if time.time() > start_time + time_budget:
                    break
                target = Fraction(1, r)
                if target not in reachable:
                    # Can add directly
                    S.append(r)
                    rejected.discard(r)
                    # Update reachable
                    new_reachable = set(reachable)
                    for s in reachable:
                        new_reachable.add(s + target)
                    reachable = new_reachable
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best_S = S[:]
                    continue
                
                # Conflict exists - find it and try to resolve
                conflict = find_conflict(S, r)
                if conflict is None:
                    continue
                
                # Try removing different subsets of the conflict
                # First try removing single elements
                for c in conflict:
                    new_S = [x for x in S if x != c]
                    # Check if new_S U {r} is valid
                    new_S_with_r = new_S + [r]
                    if is_valid(new_S_with_r):
                        S = new_S_with_r
                        rejected.discard(r)
                        reachable = get_reachable_sums(S)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best_S = S[:]
                        break
                else:
                    # Try removing pairs from conflict (only if conflict is small)
                    if len(conflict) <= 6:
                        from itertools import combinations
                        found = False
                        for pair in combinations(conflict, 2):
                            new_S = [x for x in S if x not in pair]
                            new_S_with_r = new_S + [r]
                            if is_valid(new_S_with_r):
                                S = new_S_with_r
                                rejected.discard(r)
                                reachable = get_reachable_sums(S)
                                improved = True
                                if len(S) > best_len:
                                    best_len = len(S)
                                    best_S = S[:]
                                found = True
                                break
                        if found:
                            continue
                    # Try removing the entire conflict
                    new_S = [x for x in S if x not in conflict]
                    new_S_with_r = new_S + [r]
                    if is_valid(new_S_with_r):
                        S = new_S_with_r
                        rejected.discard(r)
                        reachable = get_reachable_sums(S)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best_S = S[:]
            
            # If no improvement, try random swaps
            if not improved:
                # Try swapping a rejected element with a kept element
                swapped = False
                for r in list(rejected):
                    if time.time() > start_time + time_budget:
                        break
                    for k in S:
                        # Try swapping k for r
                        new_S = [x for x in S if x != k] + [r]
                        if is_valid(new_S):
                            S = new_S
                            rejected.discard(r)
                            reachable = get_reachable_sums(S)
                            swapped = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best_S = S[:]
                            break
                    if swapped:
                        break
                if not swapped:
                    # No more improvements possible
                    break
        
        return best_S
    
    # Main algorithm
    # Start with greedy
    S = greedy_construct()
    
    # Run local search with time budget
    remaining = deadline - time.time()
    if remaining > 1:
        S = local_search(S, remaining - 2)
    
    # Ensure we return a valid nonempty set
    if not S:
        return [24]
    
    # Final validation
    if not is_valid(S):
        # Fallback to greedy
        S = greedy_construct()
    
    return S

# EVOLVE-BLOCK-START
# The above code is the complete program. The EVOLVE-BLOCK is the entire construct function and helpers.
# EVOLVE-BLOCK-END
