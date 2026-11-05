import time
import math
import random
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Universe
    universe = list(range(1, 23))
    n_universe = len(universe)
    
    # Precompute global LCM and weights
    global_lcm = 1
    for x in universe:
        global_lcm = lcm(global_lcm, x)
        
    # w[x] is the integer weight corresponding to 1/x scaled by global_lcm
    weights = {x: global_lcm // x for x in universe}

    def is_dissociated_incremental(S):
        """
        Check if S is dissociated by building the set of subset sums incrementally.
        Returns (is_dissociated, subset_sums_set) if dissociated, else (False, None).
        """
        if not S:
            return True, {0}
            
        sums = {0}
        for x in S:
            w = weights[x]
            new_sums = set()
            for s in sums:
                new_s = s + w
                if new_s in sums:
                    return False, None
                new_sums.add(new_s)
            sums.update(new_sums)
            
        return True, sums

    def try_add(S, x, current_sums):
        """
        Check if S union {x} is dissociated, given the subset sums of S.
        Returns (is_dissociated, new_sums_set) if dissociated, else (False, None).
        """
        w = weights[x]
        # Check for collision: does any s in current_sums equal s' + w for some s' in current_sums?
        # i.e., does w belong to {s1 - s2 | s1, s2 in current_sums}?
        # Or simply: do current_sums and {s + w for s in current_sums} intersect?
        for s in current_sums:
            if s + w in current_sums:
                return False, None
        
        # No collision, generate new sums
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
            
        return True, new_sums

    def greedy_build(order, current_sums=None):
        """Build a dissociated set by greedily adding elements in the given order."""
        if current_sums is None:
            current_sums = {0}
        current_set = []
        for x in order:
            ok, new_sums = try_add(current_set, x, current_sums)
            if ok:
                current_set.append(x)
                current_sums = new_sums
        return current_set, current_sums

    def expand_set(S, current_sums):
        """Try to expand a dissociated set by adding any remaining element."""
        remaining = [x for x in universe if x not in S]
        added = True
        while added:
            added = False
            for x in remaining[:]:
                ok, new_sums = try_add(S, x, current_sums)
                if ok:
                    S.append(x)
                    current_sums = new_sums
                    remaining.remove(x)
                    added = True
                    if time.time() - start_time > timeout:
                        break
            if time.time() - start_time > timeout:
                break
        return S, current_sums

    best_set = []
    best_size = 0
    best_sums = {0}

    # Strategy 1: Greedy with descending order (largest elements first)
    candidates_desc = sorted(universe, reverse=True)
    greedy_set, greedy_sums = greedy_build(candidates_desc)
    greedy_set, greedy_sums = expand_set(greedy_set, greedy_sums)
    if len(greedy_set) > best_size:
        best_set = list(greedy_set)
        best_size = len(greedy_set)
        best_sums = greedy_sums

    # Strategy 2: Greedy with ascending order
    candidates_asc = sorted(universe)
    greedy_set_asc, greedy_sums_asc = greedy_build(candidates_asc)
    greedy_set_asc, greedy_sums_asc = expand_set(greedy_set_asc, greedy_sums_asc)
    if len(greedy_set_asc) > best_size:
        best_set = list(greedy_set_asc)
        best_size = len(greedy_set_asc)
        best_sums = greedy_sums_asc

    # Strategy 3: Backtracking search for maximum cardinality
    def backtrack_search(target_size, current_set, current_sums, start_idx, candidates):
        """Recursively try to build a set of target_size."""
        if time.time() - start_time > timeout:
            return None, None
        if len(current_set) == target_size:
            return list(current_set), current_sums
        
        remaining_needed = target_size - len(current_set)
        available = candidates[start_idx:]
        if len(available) < remaining_needed:
            return None, None
        
        for i, x in enumerate(available):
            ok, new_sums = try_add(current_set, x, current_sums)
            if ok:
                current_set.append(x)
                result_set, result_sums = backtrack_search(target_size, current_set, new_sums, start_idx + i + 1, candidates)
                if result_set is not None:
                    return result_set, result_sums
                current_set.pop()
        
        return None, None

    # Try to find sets of increasing size
    for target_size in range(best_size + 1, 16):
        if time.time() - start_time > timeout:
            break
        
        ordered_candidates = sorted(universe, reverse=True)
        result_set, result_sums = backtrack_search(target_size, [], {0}, 0, ordered_candidates)
        if result_set is not None:
            best_set = result_set
            best_size = target_size
            best_sums = result_sums
            continue

    # Strategy 4: Randomized restarts with various heuristics
    random.seed(42)
    restarts = 300 # Increased restarts due to speedup
    for r in range(restarts):
        if time.time() - start_time > timeout:
            break
        
        if r % 4 == 0:
            order = sorted(universe, reverse=True)
        elif r % 4 == 1:
            order = sorted(universe)
        elif r % 4 == 2:
            order = universe[:]
            random.shuffle(order)
        else:
            order = sorted(universe, key=lambda x: abs(x - 11.5) + random.random() * 5)
        
        current_set, current_sums = greedy_build(order)
        current_set, current_sums = expand_set(current_set, current_sums)
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            best_sums = current_sums

    # Strategy 5: Local search - try swapping elements to improve
    if time.time() - start_time < timeout - 10:
        for _ in range(50): # Increased iterations
            if time.time() - start_time > timeout:
                break
            
            current = list(best_set)
            current_sums = best_sums
            improved = True
            while improved:
                improved = False
                for i in range(len(current)):
                    for x in universe:
                        if x in current:
                            continue
                        # Try swapping current[i] with x
                        # First, remove current[i] from the set of sums? 
                        # It's easier to rebuild sums for the candidate set if it's small, 
                        # or just check if the new set is dissociated from scratch using the fast incremental method.
                        # Since n is small (< 20), rebuilding is fast.
                        
                        test_set = [current[j] for j in range(len(current)) if j != i]
                        test_set.append(x)
                        
                        # Check if test_set is dissociated
                        ok, test_sums = is_dissociated_incremental(test_set)
                        if ok:
                            # Try to expand
                            expanded_set, expanded_sums = expand_set(test_set, test_sums)
                            if len(expanded_set) > len(current):
                                current = expanded_set
                                current_sums = expanded_sums
                                improved = True
                                break
                    if improved:
                        break
            
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
                best_sums = current_sums

    return best_set
