"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflict_sets():
    """Build conflict sets: for each number, the set of other numbers that conflict with it."""
    # a + b is a perfect square
    # Max sum is 256+256=512, so squares up to 512: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    squares = [s*s for s in range(1, 23) if s*s <= 512]
    
    conflicts = [set() for _ in range(257)]
    for a in range(1, 257):
        for b in range(a, 257):
            if a + b in squares:
                if a == b:
                    # diagonal: a+a must not be square
                    conflicts[a].add(a)  # mark self-conflict
                else:
                    conflicts[a].add(b)
                    conflicts[b].add(a)
    return conflicts

def _greedy_max_independent(conflicts):
    """Greedy: pick elements with fewest conflicts among remaining."""
    remaining = set(range(1, 257))
    result = []
    while remaining:
        # Pick element with minimum degree in the induced subgraph
        best = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(conflicts[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best = v
        result.append(best)
        remaining.discard(best)
        for u in list(remaining):
            if best in conflicts[u]:
                remaining.discard(u)
    return result

def _local_search(conflicts, initial_set, max_iters=50000, deadline=None):
    """Local search: try adding elements, removing conflicting ones."""
    current = set(initial_set)
    best = set(initial_set)
    
    def is_valid(s):
        for a in s:
            if a in conflicts[a] and a in s:
                # diagonal conflict
                if a + a in [x*x for x in range(1,23) if x*x <= 512]:
                    return False
        for a in s:
            for b in s:
                if a < b:
                    if a + b in [x*x for x in range(1,23) if x*x <= 512]:
                        return False
        return True
    
    # Precompute squares for fast lookup
    squares = set(s*s for s in range(1, 23) if s*s <= 512)
    
    def is_valid_fast(s):
        for a in s:
            if 2*a in squares:
                return False
        for a in s:
            for b in s:
                if a < b and a + b in squares:
                    return False
        return True
    
    # Verify initial set
    if not is_valid_fast(current):
        # Fix by removing problematic elements
        for a in list(current):
            if 2*a in squares:
                current.discard(a)
        changed = True
        while changed:
            changed = False
            for a in list(current):
                for b in list(current):
                    if a < b and a + b in squares:
                        current.discard(b)
                        changed = True
                        break
    
    best = set(current)
    best_size = len(current)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try adding a random element not in current
        candidates = [x for x in range(1, 257) if x not in current]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        for add_candidate in candidates:
            # Check if we can add this element
            if 2*add_candidate in squares:
                continue
            
            # Find all elements that conflict with add_candidate
            conflicting = [x for x in current if x + add_candidate in squares]
            
            # Try removing the conflicting elements and adding add_candidate
            new_set = set(current)
            new_set.difference_update(conflicting)
            new_set.add(add_candidate)
            
            # Verify validity
            if is_valid_fast(new_set):
                if len(new_set) > best_size:
                    current = new_set
                    best = set(new_set)
                    best_size = len(current)
                    improved = True
                    break
        
        if not improved:
            # Try single removals and re-additions
            removed = random.choice(list(current))
            current.discard(removed)
            
            # Try to add something better
            candidates2 = [x for x in range(1, 257) if x not in current]
            random.shuffle(candidates2)
            for add_candidate in candidates2:
                if 2*add_candidate in squares:
                    continue
                conflicting = [x for x in current if x + add_candidate in squares]
                if len(conflicting) == 0:
                    new_set = set(current)
                    new_set.add(add_candidate)
                    if is_valid_fast(new_set) and len(new_set) > best_size:
                        current = new_set
                        best = set(new_set)
                        best_size = len(current)
                        break
            else:
                # Put removed back
                current.add(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    conflicts = _build_conflict_sets()
    squares = set(s*s for s in range(1, 23) if s*s <= 512)
    
    # Start with greedy solution
    greedy_result = _greedy_max_independent(conflicts)
    best_set = set(greedy_result)
    best_size = len(best_set)
    
    # Multiple local search restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Perturb the current best
        current = set(best_set)
        
        # Random perturbation: remove some random elements
        num_remove = random.randint(1, min(5, len(current)))
        to_remove = random.sample(list(current), num_remove)
        for x in to_remove:
            current.discard(x)
        
        # Try to add random elements
        candidates = [x for x in range(1, 257) if x not in current]
        random.shuffle(candidates)
        for add_candidate in candidates[:20]:
            if 2*add_candidate in squares:
                continue
            conflicting = [x for x in current if x + add_candidate in squares]
            if len(conflicting) == 0:
                current.add(add_candidate)
        
        # Local search from this perturbed state
        result = _local_search(conflicts, list(current), max_iters=10000, deadline=deadline)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Final polish: try all possible single additions
    for x in range(1, 257):
        if x in best_set:
            continue
        if 2*x in squares:
            continue
        conflicting = [y for y in best_set if x + y in squares]
        if len(conflicting) == 0:
            best_set.add(x)
            best_size = len(best_set)
    
    # Return as sorted list
    return sorted(best_set)

# EVOLVE-BLOCK-END
