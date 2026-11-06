import time
import random
from itertools import combinations

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = sorted(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    if m >= 3:
        for i in range(m - 2):
            a = S_list[i]
            for j in range(i + 1, m - 1):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (a + b + c) % x == 0:
                        return False

    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    for y in S_list:
        target = (-x) % y
        others = [v for v in S_list if v != y]
        
        seen_residues = set()
        for u in others:
            ru = u % y
            needed_rv = (target - ru) % y
            
            if needed_rv in seen_residues:
                return False
            
            seen_residues.add(ru)
            
    return True

def _is_valid(S):
    """Full validation."""
    S_list = sorted(S)
    m = len(S_list)
    for idx_a, a in enumerate(S_list):
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        found = False
        for i in range(len(others) - 2):
            for j in range(i + 1, len(others) - 1):
                for k in range(j + 1, len(others)):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        found = True
                        break
                if found: break
            if found: break
        if found:
            return False
    return True

def _precompute_forbidden(n):
    """
    Precompute for each candidate divisor d, the set of triples (a,b,c) with a<b<c,
    all in [1,n], a!=d, b!=d, c!=d, such that d | (a+b+c).
    Return: dict mapping d -> set of frozenset({a,b,c})
    """
    forbidden = {}
    for d in range(1, n + 1):
        triples = set()
        for a in range(1, n + 1):
            if a == d:
                continue
            for b in range(a + 1, n + 1):
                if b == d:
                    continue
                for c in range(b + 1, n + 1):
                    if c == d:
                        continue
                    if (a + b + c) % d == 0:
                        triples.add((a, b, c))
        forbidden[d] = triples
    return forbidden

def _backtrack_solve(n, target_size, deadline, start_order=None):
    """
    Backtracking solver with forward checking.
    Returns a valid set of size target_size if found within deadline, else None.
    """
    if start_order is None:
        start_order = list(range(n, 0, -1))
    
    forbidden = _precompute_forbidden(n)
    
    # State: selected set, and for each candidate not yet decided, track conflicts
    # We'll use a recursive backtracking with MRV (minimum remaining values) heuristic
    
    def backtrack(selected, candidates, depth):
        if time.time() > deadline:
            return None
        if len(selected) >= target_size:
            return set(selected)
        
        if not candidates:
            return None
        
        # MRV: pick candidate with fewest compatible options
        # For each candidate x, check if it can be added to selected
        # and count how many other candidates remain compatible
        
        best_x = None
        best_count = float('inf')
        best_new_candidates = None
        
        for x in candidates:
            if not _can_add(selected, x):
                continue
            
            # Count how many other candidates are still compatible with selected + {x}
            new_selected = selected | {x}
            count = 0
            new_cands = []
            for y in candidates:
                if y == x:
                    continue
                if _can_add(new_selected, y):
                    count += 1
                    new_cands.append(y)
            
            if count < best_count:
                best_count = count
                best_x = x
                best_new_candidates = new_cands
            
            if best_count <= 1:
                break
        
        if best_x is None:
            return None
        
        # Try best_x first, then others in order
        result = backtrack(selected | {best_x}, best_new_candidates, depth + 1)
        if result is not None:
            return result
        
        # Try other candidates
        for x in candidates:
            if x == best_x:
                continue
            if not _can_add(selected, x):
                continue
            
            new_selected = selected | {x}
            new_cands = [y for y in candidates if y != x and _can_add(new_selected, y)]
            
            result = backtrack(new_selected, new_cands, depth + 1)
            if result is not None:
                return result
        
        return None
    
    return backtrack(set(), list(start_order), 0)

def _greedy_with_order(order, deadline):
    """Greedy selection following a specific order."""
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
    return current_set

def _local_search_improve(S, deadline):
    """Try to improve set S by adding elements or doing swaps."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any unselected element
        for x in range(1, 193):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        
        # If no direct add, try 1-out 2-in swap
        if not improved:
            best_swap_gain = 0
            best_swap = None
            
            # Try removing one element and adding two
            for a in list(S):
                if time.time() > deadline:
                    break
                trial_set = S - {a}
                
                # Find all elements that can be added to trial_set
                addable = []
                for x in range(1, 193):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        addable.append(x)
                
                # Try pairs from addable
                for x, y in combinations(addable, 2):
                    trial2 = trial_set | {x, y}
                    if _can_add(trial_set | {x}, y):
                        # Valid swap: remove a, add x and y
                        if best_swap is None:
                            best_swap = (a, x, y)
                            best_swap_gain = 1
                        break
                if best_swap is not None:
                    break
            
            if best_swap is not None:
                a, x, y = best_swap
                S.remove(a)
                S.add(x)
                S.add(y)
                improved = True
    
    return S

def _try_backtrack_improve(S, deadline):
    """
    Try to improve set S using backtracking to find a larger valid superset.
    Returns improved set or original S if no improvement found.
    """
    if time.time() > deadline:
        return S
    
    current_size = len(S)
    
    # Try to find a valid set of size current_size + 1 using backtracking
    # Start from the current set and try to add one more element
    remaining = [x for x in range(1, 193) if x not in S]
    
    # Sort remaining by how "easy" they are to add (fewer conflicts)
    def conflict_count(x):
        count = 0
        for y in S:
            if y == x:
                continue
            # Check if x and y conflict with any pair in S
            for z in S:
                if z == x or z == y:
                    continue
                # Check if x divides y+z+something... this is expensive
                # Just use _can_add as proxy
                pass
        return 0
    
    # Simpler: just try adding each remaining element via backtracking
    for x in remaining:
        if time.time() > deadline:
            break
        if _can_add(S, x):
            new_S = S | {x}
            # Try to extend further with backtracking
            further = _backtrack_solve(192, current_size + 2, deadline, 
                                       [y for y in range(192, 0, -1) if y not in new_S])
            if further is not None and len(further) > len(S):
                return further
            return new_S
    
    # Try removing one element and adding two via backtracking
    for a in list(S):
        if time.time() > deadline:
            break
        trial_set = S - {a}
        remaining_after = [x for x in range(1, 193) if x not in trial_set]
        
        # Try to find two elements to add
        for x in remaining_after:
            if time.time() > deadline:
                break
            if not _can_add(trial_set, x):
                continue
            trial2 = trial_set | {x}
            for y in remaining_after:
                if y == x:
                    continue
                if _can_add(trial2, y):
                    new_S = trial2 | {y}
                    if len(new_S) > len(S):
                        return new_S
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy
    order = list(range(n, 0, -1))
    current_set = _greedy_with_order(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
    
    # Strategy 2: Randomized Greedy with multiple restarts
    rng = random.Random(42)
    
    # Try different bias strategies
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        if attempt % 4 == 0:
            # Pure random
            order = list(range(1, n + 1))
            rng.shuffle(order)
        elif attempt % 4 == 1:
            # Bias towards larger numbers
            order = list(range(n, 0, -1))
            rng.shuffle(order)
        elif attempt % 4 == 2:
            # Bias towards numbers with fewer divisors (primes are safer)
            def divisor_count(x):
                count = 0
                for d in range(1, int(x**0.5) + 1):
                    if x % d == 0:
                        count += 1
                        if d != x // d:
                            count += 1
                return count
            
            order = list(range(1, n + 1))
            order.sort(key=lambda x: (divisor_count(x), -x))
            rng.shuffle(order)
        else:
            # Mix: interleave large and small
            large = list(range(n, n//2, -1))
            small = list(range(n//2, 0, -1))
            rng.shuffle(large)
            rng.shuffle(small)
            order = []
            for i in range(max(len(large), len(small))):
                if i < len(large):
                    order.append(large[i])
                if i < len(small):
                    order.append(small[i])
        
        current_set = _greedy_with_order(order, deadline)
        
        # Local search improvement
        current_set = _local_search_improve(current_set, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
    
    # Strategy 3: Backtracking improvement on best_set
    if time.time() < deadline:
        best_set = _try_backtrack_improve(best_set, deadline)
    
    # Final local search on best_set
    best_set = _local_search_improve(best_set, deadline)
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback
    return [192]
