"""Improved Sidon set construction for [1,284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_valid_sidon(s):
    """Check if sorted list s is a valid Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            v = s[i] + s[j]
            if v in sums:
                return False
            sums.add(v)
    return True

def _get_sums(s):
    """Get all pairwise sums (i <= j) of sorted list s."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return sums

def _can_add(s, sums, x):
    """Check if x can be added to Sidon set s with existing sums."""
    for y in s:
        if x + y in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_to_sums(sums, s, x):
    """Add x to sums set, assuming compatibility."""
    for y in s:
        sums.add(x + y)
    sums.add(2 * x)

def _greedy_construct(start_set, order):
    """Greedy construction using a given order of candidates."""
    s = list(start_set)
    sums = _get_sums(s)
    for x in order:
        if x in s:
            continue
        if _can_add(s, sums, x):
            s.append(x)
            _add_to_sums(sums, s, x)
    s.sort()
    return s

def _local_search(s, deadline, rng):
    """Local search: try to add elements, and try swaps to enable additions."""
    s = sorted(s)
    sums = _get_sums(s)
    best = list(s)
    
    # Phase 1: Try to add any missing element
    changed = True
    while changed:
        changed = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                changed = True
                if time.time() > deadline:
                    return best
    
    # Phase 2: Try single swaps (remove one, try to add two)
    for _ in range(2000):
        if time.time() > deadline:
            return best
        if len(s) < len(best):
            s = list(best)
            sums = _get_sums(s)
        
        # Try removing one element and adding two others
        if len(s) < 2:
            break
        i = rng.randrange(len(s))
        removed = s[i]
        s_temp = s[:i] + s[i+1:]
        sums_temp = _get_sums(s_temp)
        
        # Try to add two elements
        added = 0
        new_elements = []
        for x in range(1, 285):
            if x in s_temp or x == removed:
                continue
            if _can_add(s_temp, sums_temp, x):
                s_temp.append(x)
                _add_to_sums(sums_temp, s_temp, x)
                added += 1
                new_elements.append(x)
                if added == 2:
                    break
        
        if added >= 1 and len(s_temp) > len(best):
            s = s_temp
            sums = sums_temp
            s.sort()
            best = list(s)
    
    # Phase 3: Try to add more elements after swaps
    changed = True
    while changed:
        changed = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                changed = True
                if time.time() > deadline:
                    return best
    
    return best

def _random_order(start_set, rng):
    """Generate a random order of candidates excluding start_set."""
    candidates = [x for x in range(1, 285) if x not in start_set]
    rng.shuffle(candidates)
    return candidates

def _csp_backtrack(start_set, deadline, rng, max_depth=16):
    """
    CSP-based backtracking search with forward checking and MRV heuristic.
    Objective: maximize the number of selected elements.
    
    Variables: candidate elements from 1..284 (excluding start_set)
    Constraints: all pairwise sums must be distinct
    Heuristic: MRV - select variable with fewest compatible values remaining
    """
    start_set = sorted(start_set)
    current_sums = _get_sums(start_set)
    current_set = set(start_set)
    best_found = list(start_set)
    best_size = len(start_set)
    
    # Get all candidates
    all_candidates = [x for x in range(1, 285) if x not in current_set]
    
    def _get_compatible_candidates(sums, s_set, candidates):
        """Get candidates that can be added without violating constraints."""
        compatible = []
        for x in candidates:
            if x in s_set:
                continue
            # Check if x can be added
            valid = True
            for y in s_set:
                if x + y in sums:
                    valid = False
                    break
            if valid and 2 * x in sums:
                valid = False
            if valid:
                compatible.append(x)
        return compatible
    
    def _backtrack(s_set, sums, depth, candidates):
        """Recursive backtracking with forward checking."""
        nonlocal best_found, best_size
        
        if time.time() > deadline:
            return
        
        if len(s_set) > best_size:
            best_size = len(s_set)
            best_found = sorted(s_set)
        
        if depth >= max_depth:
            return
        
        # Forward checking: get compatible candidates
        compatible = _get_compatible_candidates(sums, s_set, candidates)
        
        if not compatible:
            return
        
        # MRV heuristic: pick the candidate that constrains the most other candidates
        # Approximate by picking the one with fewest "conflicts" with remaining candidates
        # For efficiency, just pick randomly from compatible with some bias
        
        # Use a scoring function: prefer candidates that leave more options
        # Simple heuristic: pick from middle of sorted compatible list
        compatible.sort()
        
        # Try a few candidates (not all, to keep search tractable)
        num_to_try = min(5, len(compatible))
        indices = rng.sample(range(len(compatible)), num_to_try) if len(compatible) > num_to_try else range(len(compatible))
        
        for idx in indices:
            if time.time() > deadline:
                return
            
            x = compatible[idx]
            
            # Add x
            new_sums = sums.copy()
            for y in s_set:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            new_s_set = s_set | {x}
            
            # Recursive call
            _backtrack(new_s_set, new_sums, depth + 1, candidates)
            
            # Backtrack: remove x (handled by using copies)
    
    # Run backtracking from initial state
    _backtrack(set(start_set), current_sums, 0, all_candidates)
    
    return best_found

def _csp_iterative_deepening(start_set, deadline, rng, max_iterations=5):
    """
    Iterative deepening CSP search.
    Start with greedy solution, then try to improve via backtracking.
    """
    start_set = sorted(start_set)
    best = list(start_set)
    best_size = len(start_set)
    
    # First get a good initial solution via greedy
    greedy_sol = _greedy_construct(start_set, list(range(12, 285)))
    if len(greedy_sol) > best_size:
        best = list(greedy_sol)
        best_size = len(greedy_sol)
    
    # Now try to improve using CSP backtracking
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Use the current best as starting point for backtracking
        # But backtracking from a large set is expensive, so we use a smaller max_depth
        max_depth = max(1, 18 - len(best))  # Decrease depth as set grows
        
        improved = _csp_backtrack(best, deadline, rng, max_depth=max_depth)
        
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    start_set = [1, 2, 5, 11]
    best = list(start_set)
    best_size = len(start_set)
    
    rng = random.Random(42)
    
    # Phase 1: Deterministic greedy
    order = list(range(12, 285))
    s = _greedy_construct(start_set, order)
    if len(s) > best_size:
        best = list(s)
        best_size = len(s)
    
    # Phase 2: CSP-based search with iterative deepening
    # This is the new approach: use backtracking with forward checking
    if time.time() < deadline - 30:  # Reserve time for local search
        csp_result = _csp_iterative_deepening(best, deadline - 30, rng, max_iterations=3)
        if len(csp_result) > best_size:
            best = list(csp_result)
            best_size = len(csp_result)
    
    # Phase 3: Randomized greedy restarts with local search
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:  # Reserve time for final polish
            break
        
        rng_restarts = random.Random(r * 12345 + 67890)
        
        # Different strategies for ordering
        strategy = r % 4
        
        if strategy == 0:
            order = _random_order(start_set, rng_restarts)
        elif strategy == 1:
            order = _random_order(start_set, rng_restarts)
            order.sort(key=lambda x: (rng_restarts.random(), x))
        elif strategy == 2:
            order = _random_order(start_set, rng_restarts)
            order.sort(key=lambda x: (rng_restarts.random(), -x))
        else:
            small = [x for x in range(12, 150) if x not in start_set]
            large = [x for x in range(150, 285) if x not in start_set]
            rng_restarts.shuffle(small)
            rng_restarts.shuffle(large)
            order = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small):
                    order.append(small[si])
                    si += 1
                if li < len(large):
                    order.append(large[li])
                    li += 1
        
        s = _greedy_construct(start_set, order)
        s = _local_search(s, deadline - 10, rng_restarts)
        
        if len(s) > best_size:
            best = list(s)
            best_size = len(s)
    
    # Phase 4: Final local search on the best found
    if time.time() < deadline - 5:
        best = _local_search(best, deadline, rng)
    
    # Phase 5: Final CSP polish on the best result
    if time.time() < deadline - 3:
        # Try a short CSP search to see if we can squeeze out one more element
        final_csp = _csp_backtrack(best, deadline - 1, rng, max_depth=4)
        if len(final_csp) > best_size:
            best = list(final_csp)
            best_size = len(final_csp)
    
    # Verify the result is valid
    if not _is_valid_sidon(best):
        # Fallback to a known valid set
        best = [1, 2, 5, 11]
        sums = _get_sums(best)
        for x in range(12, 285):
            if _can_add(best, sums, x):
                best.append(x)
                _add_to_sums(sums, best, x)
        best.sort()
    
    return best
# EVOLVE-BLOCK-END
