"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s limit
    
    # Precompute reciprocal fractions
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    def _can_form_target(target, candidates):
        """Check if target can be formed as sum of reciprocals of a non-empty subset of candidates."""
        if not candidates:
            return False
        # Sort by value descending (larger reciprocals first) for better pruning
        cands = sorted(candidates, key=lambda x: -recips[x])
        n = len(cands)
        total = sum(recips[x] for x in cands)
        if total < target:
            return False
        
        # Precompute suffix sums for pruning
        suffix = [Fraction(0)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + recips[cands[i]]
        
        found = [False]
        
        def dfs(idx, acc):
            if found[0]:
                return
            if acc == target:
                found[0] = True
                return
            if idx >= n:
                return
            if acc > target:
                return
            # Prune: even taking all remaining can't reach target
            if acc + suffix[idx] < target:
                return
            # Include
            dfs(idx + 1, acc + recips[cands[idx]])
            # Skip
            dfs(idx + 1, acc)
        
        dfs(0, Fraction(0, 1))
        return found[0]
    
    def _is_valid(selected):
        """Check if selected set is reciprocal-subsum-free."""
        sel_set = set(selected)
        for a in selected:
            target = recips[a]
            others = [b for b in selected if b != a]
            if _can_form_target(target, others):
                return False
        return True
    
    def _greedy_build(order, deadline):
        """Greedily add elements in given order, checking validity."""
        selected = []
        selected_set = set()
        for elem in order:
            if time.time() > deadline:
                break
            if elem in selected_set:
                continue
            candidate = selected + [elem]
            if _is_valid(candidate):
                selected = candidate
                selected_set.add(elem)
        return selected
    
    def _local_search_add(current, deadline):
        """Try to add more elements to current set."""
        current = list(current)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for elem in range(1, 25):
                if time.time() > deadline:
                    break
                if elem in current_set:
                    continue
                candidate = current + [elem]
                if _is_valid(candidate):
                    current = candidate
                    current_set.add(elem)
                    improved = True
                    break
        return current
    
    def _local_search_swap(current, deadline):
        """Try 1-swap improvements: remove one, add one or two."""
        best = list(current)
        best_set = set(best)
        
        for remove_elem in list(best):
            if time.time() > deadline:
                break
            temp = [x for x in best if x != remove_elem]
            temp_set = set(temp)
            
            # Try adding one element
            for add_elem in range(1, 25):
                if time.time() > deadline:
                    break
                if add_elem in temp_set:
                    continue
                candidate = temp + [add_elem]
                if _is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    best_set = set(best)
            
            # Try adding two elements
            for add_elem1 in range(1, 25):
                if time.time() > deadline:
                    break
                if add_elem1 in temp_set:
                    continue
                for add_elem2 in range(add_elem1 + 1, 25):
                    if time.time() > deadline:
                        break
                    if add_elem2 in temp_set:
                        continue
                    candidate = temp + [add_elem1, add_elem2]
                    if _is_valid(candidate) and len(candidate) > len(best):
                        best = candidate
                        best_set = set(best)
        
        return best
    
    def _random_greedy(deadline):
        """Greedy with random ordering."""
        order = list(range(1, 25))
        random.shuffle(order)
        return _greedy_build(order, deadline)
    
    # Strategy 1: Multiple static orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        [x for x in range(24, 0, -1) if x >= 5],
        [x for x in range(24, 0, -1) if x >= 3],
        [x for x in range(24, 0, -1) if x >= 2],
        [x for x in range(1, 25) if x >= 5],
        [x for x in range(1, 25) if x >= 3],
    ]
    
    best_set = []
    for order in orders:
        if time.time() > deadline - 10.0:
            break
        result = _greedy_build(order, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # Strategy 2: Randomized greedy (multiple trials)
    if time.time() < deadline - 15.0:
        for _ in range(20):
            if time.time() > deadline - 10.0:
                break
            result = _random_greedy(deadline)
            if len(result) > len(best_set):
                best_set = result
    
    # Local search: add more elements
    if best_set and time.time() < deadline - 10.0:
        best_set = _local_search_add(best_set, deadline)
    
    # Local search: 1-swap improvements
    if best_set and time.time() < deadline - 5.0:
        improved = _local_search_swap(best_set, deadline)
        if len(improved) > len(best_set):
            best_set = improved
            # After swap, try adding again
            if time.time() < deadline - 3.0:
                best_set = _local_search_add(best_set, deadline)
    
    # Final fallback: ensure non-empty
    if not best_set:
        best_set = [24]
    
    return list(best_set)
# EVOLVE-BLOCK-END
