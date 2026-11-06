"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import defaultdict

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, n=N):
    """Check if selected is inclusion-maximal: no missing element can be added."""
    selected_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Check if adding x preserves Sidon property
        ok = True
        for a in selected:
            s = a + x
            if s in sums:
                ok = False
                break
        if ok:
            if 2 * x in sums:
                ok = False
        if ok:
            return False  # x can be added, so not maximal
    return True

def _greedy_minimal_sidon(order=None):
    """
    Greedy construction: process elements in given order, add if it doesn't break
    Sidon property. The key insight for minimality: we want a small set that is
    maximal. So we should be selective about what we add.
    
    Strategy: To get a small maximal set, we want to add elements that 'block'
    many other elements from being added later. An element x blocks y if
    x+y or 2y or x+y conflicts with existing sums.
    
    Actually, the standard approach for minimal maximal Sidon sets:
    - We need the set to be Sidon AND maximal.
    - Maximal means every element not in the set conflicts with it.
    - To minimize size, we want each chosen element to conflict with as many
      non-chosen elements as possible.
    
    Greedy approach: Start empty. At each step, pick the element that, when added,
    causes the most other elements to become non-addable (conflict), while maintaining
    Sidon property.
    """
    if order is None:
        order = list(range(1, N + 1))
    
    selected = []
    sums = set()
    
    # Track which elements are still 'available' (can be added without conflict)
    # An element x is available if:
    #   - 2*x not in sums
    #   - for all a in selected, a+x not in sums
    
    def is_available(x):
        if 2 * x in sums:
            return False
        for a in selected:
            if a + x in sums:
                return False
        return True
    
    # Greedy: repeatedly pick the element that maximizes the number of newly blocked elements
    available = set(range(1, N + 1))
    
    while available:
        best_elem = None
        best_blocked_count = -1
        best_new_sums = None
        
        for x in available:
            # Simulate adding x
            new_sums = set(sums)
            new_selected = selected + [x]
            
            # Check Sidon validity
            valid = True
            for a in selected:
                s = a + x
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            if 2 * x in new_sums:
                continue
            new_sums.add(2 * x)
            
            # Count how many other available elements become unavailable
            blocked = 0
            for y in available:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                if 2 * y in new_sums:
                    blocked += 1
                    continue
                y_blocked = False
                for a in new_selected:
                    if a + y in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    blocked += 1
            
            if blocked > best_blocked_count:
                best_blocked_count = blocked
                best_elem = x
                best_new_sums = new_sums
        
        if best_elem is None:
            break
        
        selected.append(best_elem)
        sums = best_new_sums
        available.discard(best_elem)
        
        # Remove newly unavailable elements from available
        new_available = set()
        for y in available:
            if is_available(y):
                new_available.add(y)
        available = new_available
    
    return selected

def _greedy_with_lookahead(order=None, lookahead=2):
    """Greedy with limited lookahead for better blocking."""
    if order is None:
        order = list(range(1, N + 1))
    
    selected = []
    sums = set()
    
    def is_available(x):
        if 2 * x in sums:
            return False
        for a in selected:
            if a + x in sums:
                return False
        return True
    
    available = set(range(1, N + 1))
    
    while available and len(selected) < 50:  # safety bound
        best_elem = None
        best_score = -1
        best_new_sums = None
        best_selected = None
        
        # Sample a subset if too many candidates
        candidates = list(available)
        if len(candidates) > 30:
            candidates = random.sample(candidates, 30)
        
        for x in candidates:
            # Simulate adding x
            new_sums = set(sums)
            new_selected = selected + [x]
            
            valid = True
            for a in selected:
                s = a + x
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            if 2 * x in new_sums:
                continue
            new_sums.add(2 * x)
            
            # Score: number of newly blocked elements
            blocked = 0
            for y in available:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    blocked += 1
                    continue
                y_blocked = False
                for a in new_selected:
                    if a + y in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    blocked += 1
            
            # Add a small bonus for smaller x (ties broken by position)
            score = blocked * 100 - x
            
            if score > best_score:
                best_score = score
                best_elem = x
                best_new_sums = new_sums
                best_selected = new_selected
        
        if best_elem is None:
            break
        
        selected = best_selected
        sums = best_new_sums
        available.discard(best_elem)
        
        # Update available
        new_available = set()
        for y in available:
            if is_available(y):
                new_available.add(y)
        available = new_available
    
    return selected

def _try_local_search(initial, max_iters=10000):
    """
    Local search to reduce size: try removing elements and re-adding to maintain maximality.
    Actually, we want to minimize size while maintaining Sidon + maximal.
    
    Strategy: Try to find a smaller maximal Sidon set by:
    1. Start with a valid maximal Sidon set
    2. Try to swap out one element for a better one that blocks more
    3. Or try to find a smaller subset that is still maximal (unlikely if truly minimal)
    """
    best = list(initial)
    best_size = len(best)
    
    # Try removing elements and seeing if we can still achieve maximality with fewer
    # This is hard, so instead try perturbations
    
    for _ in range(max_iters):
        # Pick a random element to remove
        if len(best) <= 1:
            break
        idx = random.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx+1:]
        
        if not _is_valid_sidon(candidate):
            continue
        
        # Check if candidate can be made maximal by adding elements
        # Actually we want to keep it the same size or smaller
        # Let's try: is candidate maximal? If not, can we add one element to make it maximal?
        if _is_maximal(candidate):
            best = candidate
            best_size = len(best)
        else:
            # Try adding one element to make it maximal
            cand_set = set(candidate)
            sums = set()
            for i in range(len(candidate)):
                for j in range(i, len(candidate)):
                    sums.add(candidate[i] + candidate[j])
            
            for x in range(1, N + 1):
                if x in cand_set:
                    continue
                # Check if x can be added
                ok = True
                for a in candidate:
                    if a + x in sums:
                        ok = False
                        break
                if ok and 2 * x in sums:
                    ok = False
                if ok:
                    new_set = candidate + [x]
                    if _is_maximal(new_set):
                        # Same size, but maybe different structure - skip for now
                        pass
                    break
    
    return best

def construct():
    start_time = time.time()
    
    best_result = None
    best_size = float('inf')
    
    # First, get a baseline from simple ascending greedy
    baseline = _greedy_minimal_sidon()
    if _is_valid_sidon(baseline) and _is_maximal(baseline):
        best_result = baseline
        best_size = len(baseline)
    
    # Try multiple random orderings with greedy
    random.seed(42)
    
    for iteration in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        result = _greedy_with_lookahead(order)
        
        if len(result) < best_size:
            if _is_valid_sidon(result) and _is_maximal(result):
                best_result = result
                best_size = len(result)
        
        # Try local search on the best result
        if best_result is not None and best_size < 20:
            improved = _try_local_search(best_result, max_iters=500)
            if len(improved) < best_size:
                if _is_valid_sidon(improved) and _is_maximal(improved):
                    best_result = improved
                    best_size = len(improved)
    
    # Also try some structured orderings
    for perm_type in ['ascending', 'descending', 'even_first', 'odd_first']:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        if perm_type == 'ascending':
            order = list(range(1, N + 1))
        elif perm_type == 'descending':
            order = list(range(N, 0, -1))
        elif perm_type == 'even_first':
            order = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        else:
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        
        result = _greedy_with_lookahead(order)
        if len(result) < best_size:
            if _is_valid_sidon(result) and _is_maximal(result):
                best_result = result
                best_size = len(result)
    
    if best_result is None:
        # Fallback: ascending greedy
        best_result = _greedy_minimal_sidon()
    
    return best_result

# EVOLVE-BLOCK-END
