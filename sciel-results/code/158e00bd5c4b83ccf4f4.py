import math
import time
import random

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    n = 256
    elements = list(range(1, n + 1))
    
    # Precompute perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Build conflict: for each a, which b >= a conflict with a (a+b is square)
    # We need a+b not in squares for all a<=b in set
    # Also 2a not in squares (diagonal)
    
    # First, filter out elements where 2*a is a perfect square (diagonal constraint)
    # 2a = k^2 means a = k^2/2, so k must be even, k=2m, a=2m^2
    # a = 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242
    diagonal_bad = set()
    for k in range(1, int(math.isqrt(512)) + 1):
        s = k * k
        if s % 2 == 0:
            a = s // 2
            if 1 <= a <= n:
                diagonal_bad.add(a)
    
    # Valid candidates (pass diagonal test)
    candidates = [x for x in elements if x not in diagonal_bad]
    
    # Build adjacency: for each candidate, which other candidates conflict
    # Conflict if a+b is a perfect square
    adj = {x: set() for x in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i:]:
            if a + b in squares:
                adj[a].add(b)
                adj[b].add(a)
    
    def is_valid_subset(subset_set):
        for a in subset_set:
            if 2 * a in squares:
                return False
            for b in subset_set:
                if b >= a and (a + b) in squares:
                    return False
        return True
    
    # Greedy: process candidates in order of increasing degree (fewest conflicts first)
    def greedy_by_degree():
        remaining = set(candidates)
        result = set()
        deg = {x: len(adj[x] & remaining) for x in remaining}
        
        while remaining:
            # Pick element with minimum degree among remaining
            best = min(remaining, key=lambda x: (deg[x], x))
            result.add(best)
            # Remove all conflicts of best from remaining
            conflicts = adj[best] & remaining
            remaining -= conflicts
            for c in conflicts:
                deg[c] -= 1
            # Also update degrees for remaining (they lost 'best' as neighbor if connected)
            # Actually we removed conflicts, so for remaining elements, their degree to remaining might change
            # Let's just recompute for simplicity if set is small
            del deg[best]
            for c in remaining:
                # degree within remaining
                pass  # skip for speed, approximate
        
        return result
    
    # Better greedy with full degree tracking
    def greedy_full():
        remaining = set(candidates)
        result = set()
        
        while remaining:
            # For each remaining element, count conflicts within remaining
            best = None
            best_deg = float('inf')
            for x in remaining:
                d = len(adj[x] & remaining)
                if d < best_deg or (d == best_deg and x < best):
                    best_deg = d
                    best = x
            
            result.add(best)
            remaining.discard(best)
            # Remove all neighbors of best from remaining
            for nb in adj[best]:
                remaining.discard(nb)
        
        return result
    
    # Try greedy and get initial solution
    best_set = greedy_full()
    best_size = len(best_set)
    
    # Local search: try to add removed elements, or swap
    # Represent as set for fast lookup
    current = set(best_set)
    
    def try_add(x):
        if x in current:
            return False
        if 2 * x in squares:
            return False
        for y in current:
            if (x + y) in squares:
                return False
        return True
    
    # Iterated local search
    random.seed(42)
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Try to add any element that fits
        added = False
        for x in candidates:
            if x not in current and try_add(x):
                current.add(x)
                added = True
        
        # Try swaps: remove one, add another
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
        
        # Random perturbation: remove a random element, try to add others
        if random.random() < 0.3:
            if current:
                to_remove = random.choice(list(current))
                current.discard(to_remove)
                # Try to add back various elements
                for x in candidates:
                    if x not in current and try_add(x):
                        current.add(x)
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
    
    # Final greedy pass on best_set
    final_set = set(best_set)
    # Try to add more
    for x in candidates:
        if x not in final_set and try_add(x):
            final_set.add(x)
    
    if len(final_set) == 0:
        return [1]
    
    return sorted(final_set)
