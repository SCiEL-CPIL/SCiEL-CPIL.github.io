"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START

def _can_add_element(x, selected, sums):
    """Check if element x can be added to maintain Sidon property."""
    for a in selected:
        s = x + a
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_element(x, selected, sums):
    """Add element x to selected set and update sums."""
    selected.append(x)
    for a in selected:
        sums.add(x + a)
    sums.add(2 * x)

def _remove_element(idx, selected, sums):
    """Remove element at idx from selected and rebuild sums."""
    removed = selected.pop(idx)
    # Rebuild sums from scratch (simpler and correct)
    new_sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            new_sums.add(selected[i] + selected[j])
    return new_sums

def _is_maximal_fast(selected, sums, n=N):
    """Check if the Sidon set is maximal using precomputed sums."""
    selected_set = set(selected)
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        if _can_add_element(x, selected, sums):
            return False
    return True

def _greedy_maximal_construction(order, n=N):
    """Greedy construction: add elements in given order to build a maximal Sidon set."""
    selected = []
    sums = set()
    
    for x in order:
        if x < 1 or x > n:
            continue
        if _can_add_element(x, selected, sums):
            _add_element(x, selected, sums)
    
    return selected, sums

def _local_search_minimize(selected, sums, n=N, max_iters=10000, deadline=None):
    """
    Local search to minimize size of a maximal Sidon set.
    Strategy: try removing elements, if not maximal, try swapping to restore maximality.
    """
    best = list(selected)
    best_sums = set(sums)
    best_len = len(selected)
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        current = list(selected)
        current_sums = set(sums)
        current_set = set(current)
        
        # Try removing a random element
        if len(current) > 1:
            idx = rng.randrange(len(current))
            candidate = current[:idx] + current[idx+1:]
            candidate_sums = set()
            for i in range(len(candidate)):
                for j in range(i, len(candidate)):
                    candidate_sums.add(candidate[i] + candidate[j])
            
            # Check if still Sidon (should be) and if maximal
            if _is_maximal_fast(candidate, candidate_sums, n):
                # Found a smaller maximal set!
                selected = candidate
                sums = candidate_sums
                best = candidate[:]
                best_sums = candidate_sums
                best_len = len(candidate)
                continue
        
        # If removal didn't work, try a swap: remove one, add another
        if len(current) > 0:
            idx = rng.randrange(len(current))
            removed_elem = current[idx]
            
            # Try to find a replacement that makes it maximal
            temp = current[:idx] + current[idx+1:]
            temp_sums = set()
            for i in range(len(temp)):
                for j in range(i, len(temp)):
                    temp_sums.add(temp[i] + temp[j])
            
            # Find elements that can be added to temp
            candidates = []
            for x in range(1, n + 1):
                if x not in set(temp) and x != removed_elem:
                    if _can_add_element(x, temp, temp_sums):
                        candidates.append(x)
            
            if candidates:
                # Try each candidate to see if resulting set is maximal
                for x in candidates:
                    candidate = temp + [x]
                    candidate_set = set(candidate)
                    candidate_sums = set(temp_sums)
                    for a in candidate:
                        candidate_sums.add(x + a)
                    candidate_sums.add(2 * x)
                    
                    if _is_maximal_fast(candidate, candidate_sums, n):
                        selected = candidate
                        sums = candidate_sums
                        best = candidate[:]
                        best_sums = candidate_sums
                        best_len = len(candidate)
                        break
    
    return best, best_sums

def _multi_start_search(deadline):
    """Multi-start local search from various initial maximal Sidon sets."""
    best = None
    best_len = float('inf')
    
    random.seed(42)
    
    # Generate diverse initial maximal sets
    initial_configs = []
    
    # Random orderings
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        initial_configs.append(order)
    
    # Structured orderings (arithmetic progressions)
    for step in [3, 5, 7, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 5:
            break
        for base in range(1, min(step, 5) + 1):
            order = [x for x in range(base, N + 1, step)]
            # Fill in remaining elements randomly
            remaining = [x for x in range(1, N + 1) if x not in set(order)]
            random.shuffle(remaining)
            order.extend(remaining)
            initial_configs.append(order)
    
    # Process each initial config
    for order in initial_configs:
        if time.time() > deadline - 2:
            break
        
        # Build maximal Sidon set
        maximal_set, maximal_sums = _greedy_maximal_construction(order, N)
        
        if len(maximal_set) < best_len:
            best_len = len(maximal_set)
            best = maximal_set[:]
        
        # Run local search to minimize
        if time.time() < deadline - 1:
            searched, _ = _local_search_minimize(
                maximal_set, maximal_sums, N,
                max_iters=500,
                deadline=deadline - 1
            )
            if len(searched) < best_len:
                best_len = len(searched)
                best = searched[:]
    
    return best

def _final_polish(best, deadline):
    """Final polish: aggressive removal attempts on the best solution."""
    if best is None:
        return best
    
    current = list(best)
    current_sums = set()
    for i in range(len(current)):
        for j in range(i, len(current)):
            current_sums.add(current[i] + current[j])
    
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            candidate_sums = set()
            for a in range(len(candidate)):
                for b in range(a, len(candidate)):
                    candidate_sums.add(candidate[a] + candidate[b])
            
            if _is_maximal_fast(candidate, candidate_sums, N):
                current = candidate
                current_sums = candidate_sums
                improved = True
                break
    
    return current

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256] using multi-start local search."""
    start_time = time.time()
    deadline = start_time + 25.0  # 25 second safety margin (well under 30s limit)
    
    best = _multi_start_search(deadline)
    
    # Final polish
    if best is not None and time.time() < deadline - 0.5:
        best = _final_polish(best, deadline)
    
    # Fallback if something went wrong
    if best is None or len(best) == 0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        best, _ = _greedy_maximal_construction(order, N)
    
    return best

# EVOLVE-BLOCK-END
