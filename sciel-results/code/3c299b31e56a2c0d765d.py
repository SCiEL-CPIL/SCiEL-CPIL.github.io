"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _sidon_sums(selected):
    """Compute all pair sums of a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Check if a set is a valid Sidon set (all pair sums distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, n=N):
    """Check if the Sidon set is maximal (cannot add any missing element)."""
    selected_set = set(selected)
    sums = _sidon_sums(selected)
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Check if adding x would create a duplicate sum
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            s = 2 * x
            if s in sums:
                can_add = False
        if can_add:
            return False  # Not maximal
    return True

def _greedy_maximal_sidon(order, n=N):
    """Greedy construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        if x < 1 or x > n:
            continue
        # Check if x can be added
        conflict = False
        for a in selected:
            s = x + a
            if s in sums:
                conflict = True
                break
        if not conflict:
            s = 2 * x
            if s in sums:
                conflict = True
        if not conflict:
            selected.append(x)
            # Add all new sums
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _try_removal(selected, n=N):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _is_maximal(candidate, n):
                selected = candidate
                improved = True
                break
    return selected

def _local_search_with_swaps(selected, n=N, max_iters=100):
    """Local search: try removals and swaps to reduce size while maintaining maximality."""
    selected = list(selected)
    best = selected[:]
    best_len = len(selected)
    
    for _ in range(max_iters):
        improved = False
        
        # Try removing each element
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            if _is_maximal(candidate, n):
                best = candidate
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another, then try to remove more
        selected_set = set(best)
        for i in range(len(best)):
            if time.time() > 55:
                return best
            removed = best[i]
            base = best[:i] + best[i+1:]
            base_set = set(base)
            
            for x in range(1, n + 1):
                if x in base_set:
                    continue
                # Check if x can be added to base
                can_add = True
                base_sums = _sidon_sums(base)
                for a in base:
                    if x + a in base_sums:
                        can_add = False
                        break
                if can_add and 2*x in base_sums:
                    can_add = False
                
                if not can_add:
                    continue
                
                candidate = base + [x]
                # Now try to remove something from candidate
                for j in range(len(candidate)):
                    cand2 = candidate[:j] + candidate[j+1:]
                    if len(cand2) < best_len and _is_maximal(cand2, n):
                        best = cand2
                        best_len = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _repair_minimize(selected, n=N):
    """Aggressive minimization: repeatedly try to find a smaller maximal set by repair."""
    best = list(selected)
    best_len = len(best)
    improved = True
    
    while improved and time.time() < 55:
        improved = False
        
        # Try removing each element
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            if _is_maximal(candidate, n):
                best = candidate
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-out 1-in swaps
        best_set = set(best)
        for i in range(len(best)):
            removed = best[i]
            base = best[:i] + best[i+1:]
            base_set = set(base)
            base_sums = _sidon_sums(base)
            
            for x in range(1, n + 1):
                if x in base_set:
                    continue
                
                # Check if x can be added to base
                can_add = True
                for a in base:
                    if x + a in base_sums:
                        can_add = False
                        break
                if can_add and 2*x in base_sums:
                    can_add = False
                
                if not can_add:
                    continue
                
                candidate = base + [x]
                
                # Try to remove one more element from candidate
                for j in range(len(candidate)):
                    cand2 = candidate[:j] + candidate[j+1:]
                    if len(cand2) < best_len and _is_maximal(cand2, n):
                        best = cand2
                        best_len = len(best)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return best

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin
    
    best = None
    best_len = float('inf')
    
    # Strategy 1: Multiple random orderings with greedy construction
    random.seed(42)
    
    # Try several different orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if attempt < 50:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif attempt < 100:
            # Descending order with some randomization
            order = list(range(N, 0, -1))
            random.shuffle(order[:N//2])
        elif attempt < 150:
            # Grouped by residue classes
            order = []
            for r in range(N):
                for x in range(r + 1, N + 1, N):
                    order.append(x)
            random.shuffle(order)
        else:
            # Biased towards larger numbers first
            order = list(range(1, N + 1))
            weights = [1.0 / (x + 1) for x in order]
            # Sample without replacement using weights
            remaining = list(range(1, N + 1))
            order = []
            for _ in range(N):
                total_w = sum(weights[i-1] for i in remaining)
                r = random.random() * total_w
                cum = 0
                for i in remaining:
                    cum += weights[i-1]
                    if r <= cum:
                        order.append(i)
                        remaining.remove(i)
                        break
    
        selected = _greedy_maximal_sidon(order, N)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best = selected[:]
    
    # Try to improve via removal
    if time.time() < deadline - 10 and best is not None:
        improved = _try_removal(best, N)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved
    
    # Try local search with swaps on the best found
    if time.time() < deadline - 5 and best is not None:
        improved = _local_search_with_swaps(best, N, max_iters=50)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved
    
    # Final aggressive repair-based minimization
    if time.time() < deadline - 2 and best is not None:
        improved = _repair_minimize(best, N)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved
    
    return best if best is not None else [1, 2]

# EVOLVE-BLOCK-END
