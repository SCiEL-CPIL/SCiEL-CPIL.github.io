"""Rewritten program: branch-and-bound with cardinality-indexed subset-sum DP."""
import time
import random

# EVOLVE-BLOCK-START

def _add_element(sumsets, x, m):
    """
    Try to add element x to a set of size m with current sumsets.
    sumsets[c] = set of subset sums of cardinality c, for c=0..m.
    Returns new sumsets (list of sets for c=0..m+1) if valid, else None.
    """
    # Precompute: for each cardinality c, we need to check new sums (from c-1 + x)
    # against all old sumsets[d] for d ≠ c.
    
    # Optimization: build a set of all sums in sumsets[d] for d in 0..m
    # Then for new sum at cardinality c, check if it's in (all_sums - sumsets[c])
    # But that's not quite right because we need to exclude sumsets[c] specifically.
    
    # Simpler: just check each new sum against each d ≠ c
    # With m small (≤60) and sums bounded by 256*60=15360, this should be OK
    
    new_sumsets = [set(sumsets[c]) for c in range(m + 1)]
    new_sumsets.append(set())
    
    for c in range(1, m + 2):
        for s in sumsets[c - 1]:
            new_s = s + x
            # Check collision with old sumsets[d] for d ≠ c
            for d in range(m + 1):
                if d == c:
                    continue
                if new_s in sumsets[d]:
                    return None
            new_sumsets[c].add(new_s)
    
    return new_sumsets

def _greedy(order, deadline):
    """Greedy construction: add elements in given order if valid."""
    current_set = []
    sumsets = [{0}]
    for x in order:
        if time.time() > deadline:
            break
        m = len(current_set)
        new_sumsets = _add_element(sumsets, x, m)
        if new_sumsets is not None:
            current_set.append(x)
            sumsets = new_sumsets
    return current_set

def _branch_and_bound(elements, target_k, deadline):
    """
    Branch and bound to find a valid set of size target_k from elements.
    elements should be sorted in preferred order (descending for heuristic).
    """
    def dfs(start_idx, current_set, sumsets, depth):
        if time.time() > deadline:
            return None
        if depth == target_k:
            return list(current_set)
        remaining = target_k - depth
        if len(elements) - start_idx < remaining:
            return None
        
        for i in range(start_idx, len(elements)):
            if time.time() > deadline:
                return None
            x = elements[i]
            m = depth
            new_sumsets = _add_element(sumsets, x, m)
            if new_sumsets is not None:
                current_set.append(x)
                result = dfs(i + 1, current_set, new_sumsets, depth + 1)
                if result is not None:
                    return result
                current_set.pop()
        return None
    
    return dfs(0, [], [{0}], 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Greedy with multiple orderings
    best_set = []
    
    # Descending
    s = _greedy(list(range(256, 0, -1)), deadline)
    if len(s) > len(best_set):
        best_set = s
    
    # Ascending  
    s = _greedy(list(range(1, 257)), deadline)
    if len(s) > len(best_set):
        best_set = s
    
    # Random shuffles
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        s = _greedy(order, deadline)
        if len(s) > len(best_set):
            best_set = s
    
    # B&B to try to improve
    current_best = len(best_set)
    
    # Try increasing target sizes
    for target_k in range(current_best + 1, min(current_best + 5, 61)):
        if time.time() > deadline:
            break
        elements = list(range(256, 0, -1))
        result = _branch_and_bound(elements, target_k, deadline)
        if result is not None:
            best_set = result
            current_best = target_k
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END

