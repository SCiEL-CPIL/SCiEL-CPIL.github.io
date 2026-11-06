"""Direct structural approach for equal-LCM-triple-free subset of [1,128]."""
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure: for each element, which pairs with it form bad triples
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    # Precompute for each x: list of (y,z) pairs where (x,y,z) is bad
    # Also precompute a bitmask-like structure for fast can_add checks
    # For each x, store the set of (y,z) pairs sorted for quick lookup
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True
    
    def greedy_fill(S, order=None):
        """Fill S by adding elements in given order (or natural order)."""
        if order is None:
            order = elements
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    def fast_local_search(S, max_rounds=20):
        """Fast local search: try additions and simple 1-out-1-in swaps."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            # Try additions
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
            # Try simple swaps: remove x, try to add y
            for x in list(S):
                S.remove(x)
                best_elem = None
                for y in elements:
                    if y != x and y not in S and can_add(y, S):
                        best_elem = y
                        break
                if best_elem is not None:
                    S.add(best_elem)
                    improved = True
                else:
                    S.add(x)
        return S
    
    def branch_and_bound_improved(S, candidates, best_size, best_set, depth=0):
        """Improved branch and bound with better pruning."""
        if time.time() > deadline - 10:
            return best_size, best_set
        
        if not candidates:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return best_size, best_set
        
        # Upper bound: current size + number of candidates
        if len(S) + len(candidates) <= best_size:
            return best_size, best_set
        
        # Pick the most constrained candidate
        best_x = None
        best_conflict_count = -1
        for x in candidates:
            count = 0
            for (y, z) in conflict_pairs[x]:
                if y in candidates and z in candidates:
                    count += 1
            if count > best_conflict_count:
                best_conflict_count = count
                best_x = x
        
        if best_x is None:
            best_x = candidates[0]
        
        # Branch 1: Include best_x (try first if it can be added)
        if can_add(best_x, S):
            S_new = S | {best_x}
            candidates_new = set()
            for x in candidates:
                if x != best_x and can_add(x, S_new):
                    candidates_new.add(x)
            best_size, best_set = branch_and_bound_improved(S_new, candidates_new, best_size, best_set, depth + 1)
        
        # Branch 2: Exclude best_x
        candidates_new = set(candidates)
        candidates_new.discard(best_x)
        best_size, best_set = branch_and_bound_improved(S, candidates_new, best_size, best_set, depth + 1)
        
        return best_size, best_set
    
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    triple_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    orderings = [
        sorted(elements, key=lambda x: (triple_count[x], x)),
        sorted(elements, key=lambda x: (triple_count[x], -x)),
        sorted(elements, key=lambda x: x),
        sorted(elements, key=lambda x: -x),
    ]
    
    for order in orderings:
        if time.time() > deadline - 40:
            break
        S = set()
        greedy_fill(S, order)
        S = fast_local_search(S, max_rounds=10)
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 2: Branch and bound from a good starting point
    if time.time() < deadline - 35:
        S_start = set(best)
        candidates = set()
        for x in elements:
            if x not in S_start and can_add(x, S_start):
                candidates.add(x)
        
        if candidates:
            best_size, best = branch_and_bound_improved(S_start, candidates, best_size, best)
    
    # Strategy 3: Randomized greedy with fast local search
    random.seed(42)
    num_starts = 50
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        greedy_fill(S, order)
        S = fast_local_search(S, max_rounds=5)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Final fill
    if time.time() < deadline - 3:
        greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)
