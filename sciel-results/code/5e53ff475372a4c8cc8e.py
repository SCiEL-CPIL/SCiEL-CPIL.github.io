"""Maximum independent set on the square-sum conflict graph via branch-and-bound with bitset pruning."""

# EVOLVE-BLOCK-START
import math
import time

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _build_conflict_bitsets():
    """Return list of ints where bit j set in masks[i] means i and j conflict (i+j is a perfect square)."""
    N = 256
    masks = [0] * N
    for i in range(N):
        val = i + 1
        m = 0
        # diagonal: 2*val square?
        if _is_square(2 * val):
            m |= (1 << i)
        for j in range(i + 1, N):
            if _is_square(val + (j + 1)):
                m |= (1 << j)
        masks[i] = m
    return masks

def _greedy_independent_set(cands, masks, strategy='min_deg'):
    """
    Greedy: repeatedly pick a vertex, remove it and its neighbors.
    strategy: 'min_deg', 'max_deg', 'random', 'index'
    Returns the count and the set of indices.
    """
    count = 0
    remaining = cands
    selected = []
    import random
    
    while remaining:
        best_v = -1
        best_score = None
        
        temp = remaining
        while temp:
            low = temp & (-temp)
            v = low.bit_length() - 1
            deg = (masks[v] & remaining).bit_count()
            
            if strategy == 'min_deg':
                score = deg
                if best_score is None or score < best_score:
                    best_score = score
                    best_v = v
            elif strategy == 'max_deg':
                score = deg
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            elif strategy == 'random':
                # Pick randomly among all? No, just break early or pick first?
                # For speed, let's just pick the first one if random, or use a random seed?
                # Actually, for a deterministic improvement, let's stick to min/max.
                # We can add 'random' by shuffling the bits? That's expensive.
                # Let's just use 'min_deg' and 'max_deg' and 'index'.
                if best_score is None:
                    best_score = 0
                    best_v = v
                break # Just pick the first one for 'random' placeholder, or implement properly?
            elif strategy == 'index':
                # Pick smallest index
                if best_score is None:
                    best_score = v
                    best_v = v
                # Since we iterate low to high, the first one is the smallest index
                break
            
            temp -= low
            
        if best_v == -1:
            break
            
        selected.append(best_v)
        count += 1
        # Remove best_v and its neighbors from remaining
        remaining &= ~(masks[best_v] | (1 << best_v))
        
    return count, selected

def _find_initial_solution(masks, N):
    """Run multiple greedy strategies to find a good initial independent set."""
    full = (1 << N) - 1
    best_count = 0
    best_selected = []
    
    strategies = ['min_deg', 'max_deg', 'index']
    
    for strat in strategies:
        cnt, sel = _greedy_independent_set(full, masks, strat)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel
            
    # Also try a simple random restart a few times? 
    # Given time limits, deterministic strategies are safer.
    # We can also try sorting by degree first and then greedy?
    
    return best_count, best_selected

def _solve_max_independent_set(masks, N, time_budget, init_count, init_selected):
    """Branch and bound for maximum independent set. Returns a list of selected vertex indices."""
    start = time.time()
    full = (1 << N) - 1

    # Precompute degree ordering for initial branching hint?
    # We'll use dynamic degree in the search.
    
    best_size = init_count
    best_set = set(init_selected)
    # Convert best_set to bitset for fast comparison? Or just keep as set/list.
    # For the final result, we need the list.
    
    # To speed up the "is better" check, we can keep best as a bitset.
    best_bitset = 0
    for v in init_selected:
        best_bitset |= (1 << v)

    def search(C, current_bitset, current_size):
        nonlocal best_size, best_bitset
        if time.time() - start > time_budget:
            return
            
        # Upper bound: current_size + number of candidates
        # If this is <= best_size, we can prune.
        if current_size + C.bit_count() <= best_size:
            return
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_bitset = current_bitset
            return

        # Pick a vertex to branch on: choose one with high degree in C for better pruning
        # High degree in C means it conflicts with many other candidates.
        # Including it removes many candidates (good for depth, maybe bad for width?).
        # Excluding it removes only itself.
        # Standard heuristic: pick max degree vertex.
        
        temp = C
        v = -1
        maxdeg = -1
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            deg = (masks[idx] & C).bit_count()
            if deg > maxdeg:
                maxdeg = deg
                v = idx
            temp -= low
            
        if v == -1:
            return

        # Branch 1: Include v
        # New candidates: C without v and without neighbors of v
        newC = C & ~(masks[v] | (1 << v))
        newCurrentBitset = current_bitset | (1 << v)
        search(newC, newCurrentBitset, current_size + 1)

        # Branch 2: Exclude v
        # New candidates: C without v
        C2 = C & ~(1 << v)
        search(C2, current_bitset, current_size)

    # Start search
    search(full, 0, 0)

    # Convert best bitset to list of values (1-indexed)
    result = []
    b = best_bitset
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        result.append(idx + 1)
        b -= low
    return result

def construct():
    import time
    start = time.time()
    N = 256
    masks = _build_conflict_bitsets()
    
    # Find a good initial solution
    init_count, init_selected = _find_initial_solution(masks, N)
    
    # Use branch and bound with time budget
    time_budget = 55.0
    result = _solve_max_independent_set(masks, N, time_budget, init_count, init_selected)
    
    # Fallback if something went wrong
    if not result:
        return list(range(1, 257, 3))
    
    # Validate: ensure it's a valid independent set
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if _is_square(a + b):
                    # invalid; fall back
                    return list(range(1, 257, 3))
    return sorted(res_set)
# EVOLVE-BLOCK-END
