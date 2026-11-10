# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitmask-based branch-and-bound for maximum independent set, combined with
    local search (1-add, 1-swap) and multiple random restarts.
    """
    m = 205
    deadline = time.time() + 620  # safety margin before 630s limit
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: conflict[i] is a bitmask where bit j is set if i and j conflict
    # Two elements conflict if their difference (in either direction) is in forbidden
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
    
    # Precompute neighbor masks for faster operations
    # available_mask: for a given set S (bitmask), the candidates that can be added
    # are those not in S and not conflicting with any element in S
    
    def get_available(S_mask, candidates_mask):
        """Return bitmask of candidates that can be added to set S."""
        # Candidates not already in S
        avail = candidates_mask & ~S_mask
        # Remove candidates that conflict with any element in S
        for i in range(m):
            if (S_mask >> i) & 1:
                avail &= ~conflict[i]
        return avail
    
    def greedy_construct(order):
        """Greedy construction given an ordering of candidates."""
        S_mask = 0
        for candidate in order:
            if (S_mask >> candidate) & 1:
                continue
            # Check if candidate conflicts with any in S
            ok = True
            for i in range(m):
                if (S_mask >> i) & 1:
                    if (conflict[i] >> candidate) & 1:
                        ok = False
                        break
            if ok:
                S_mask |= (1 << candidate)
        return S_mask
    
    def local_search(S_mask, max_iter=10000):
        """
        Local search: try 1-add and 1-swap moves to improve the set size.
        Returns improved S_mask.
        """
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try 1-add: find a candidate that can be added
            avail = get_available(S_mask, (1 << m) - 1)
            if avail != 0:
                # Pick the first available candidate
                # Find lowest set bit
                bit = avail & (-avail)
                idx = bit.bit_length() - 1
                S_mask |= (1 << idx)
                improved = True
                continue
            
            # Try 1-swap: remove one element, try to add two new ones
            # For each element in S, try removing it and see if we can add more
            S_list = [i for i in range(m) if (S_mask >> i) & 1]
            for i in S_list:
                S_temp = S_mask & ~(1 << i)
                avail_temp = get_available(S_temp, (1 << m) - 1)
                if avail_temp == 0:
                    continue
                # Try adding one element
                # Find the best element to add (one that allows further additions)
                # Simple: just add the first available
                bit = avail_temp & (-avail_temp)
                idx = bit.bit_length() - 1
                S_new = S_temp | (1 << idx)
                # Check if we can add another
                avail2 = get_available(S_new, (1 << m) - 1)
                if avail2 != 0:
                    bit2 = avail2 & (-avail2)
                    idx2 = bit2.bit_length() - 1
                    S_new |= (1 << idx2)
                    # Count elements
                    if S_new.bit_count() > S_mask.bit_count():
                        S_mask = S_new
                        improved = True
                        break
                elif S_new.bit_count() > S_mask.bit_count():
                    S_mask = S_new
                    improved = True
                    break
        
        return S_mask
    
    # Branch and bound for maximum independent set
    def branch_and_bound(candidates_mask, current_S_mask, best_mask, best_size):
        """
        Recursive branch and bound.
        candidates_mask: bitmask of remaining candidates to consider
        current_S_mask: current independent set
        best_mask: best found so far
        best_size: size of best found
        Returns (best_mask, best_size)
        """
        if time.time() >= deadline:
            return best_mask, best_size
        
        # Upper bound: current size + number of candidates that could potentially be added
        # Simple bound: current_size + popcount(candidates_mask)
        # But we can do better: greedy coloring bound
        # For simplicity, use a greedy bound: count how many non-conflicting candidates we can greedily pick
        if candidates_mask == 0:
            if current_S_mask.bit_count() > best_size:
                best_mask = current_S_mask
                best_size = current_S_mask.bit_count()
            return best_mask, best_size
        
        # Greedy upper bound
        greedy_count = 0
        temp_mask = candidates_mask
        while temp_mask != 0:
            bit = temp_mask & (-temp_mask)
            idx = bit.bit_length() - 1
            greedy_count += 1
            # Remove this and all its neighbors from consideration
            temp_mask &= ~conflict[idx]
            temp_mask &= ~bit
        
        if current_S_mask.bit_count() + greedy_count <= best_size:
            return best_mask, best_size
        
        # Pick a vertex to branch on
        # Choose vertex with highest degree in the induced subgraph
        best_vertex = -1
        best_degree = -1
        temp = candidates_mask
        while temp != 0:
            bit = temp & (-temp)
            idx = bit.bit_length() - 1
            # Degree in induced subgraph
            deg = (conflict[idx] & candidates_mask).bit_count()
            if deg > best_degree:
                best_degree = deg
                best_vertex = idx
            temp &= ~bit
        
        if best_vertex == -1:
            return best_mask, best_size
        
        # Branch 1: include best_vertex
        if (conflict[best_vertex] & current_S_mask) == 0:
            # best_vertex doesn't conflict with current S
            new_S = current_S_mask | (1 << best_vertex)
            new_candidates = candidates_mask & ~(1 << best_vertex) & ~conflict[best_vertex]
            best_mask, best_size = branch_and_bound(
                new_candidates, new_S, best_mask, best_size
            )
        
        # Branch 2: exclude best_vertex
        new_candidates = candidates_mask & ~(1 << best_vertex)
        best_mask, best_size = branch_and_bound(
            new_candidates, current_S_mask, best_mask, best_size
        )
        
        return best_mask, best_size
    
    # Main search loop with multiple strategies
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    random.seed(42)
    
    # Natural order
    S = greedy_construct(range(m))
    S = local_search(S)
    if S.bit_count() > best_size:
        best_mask = S
        best_size = S.bit_count()
    
    # Reverse order
    S = greedy_construct(range(m-1, -1, -1))
    S = local_search(S)
    if S.bit_count() > best_size:
        best_mask = S
        best_size = S.bit_count()
    
    # Random orderings
    for trial in range(50):
        if time.time() >= deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        S = greedy_construct(order)
        S = local_search(S)
        if S.bit_count() > best_size:
            best_mask = S
            best_size = S.bit_count()
    
    # Strategy 2: Branch and bound on a reduced problem
    # Start from the best found, try to improve via B&B on the remaining candidates
    # But B&B on 205 vertices is infeasible. Instead, use a heuristic B&B on a subset.
    
    # Strategy 3: More aggressive local search with 2-swap
    def local_search_2swap(S_mask, max_iter=5000):
        """Try 2-swap moves: remove two, add three (net +1)."""
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            
            S_list = [i for i in range(m) if (S_mask >> i) & 1]
            
            # Try 1-add first
            avail = get_available(S_mask, (1 << m) - 1)
            if avail != 0:
                bit = avail & (-avail)
                idx = bit.bit_length() - 1
                S_mask |= (1 << idx)
                improved = True
                continue
            
            # Try 1-swap (remove 1, add 1) - only useful if it enables further adds
            # Try 2-swap: remove 2, add 2 (net same, but different structure) or remove 2, add 3
            for i, j in combinations(S_list, 2):
                if time.time() >= deadline:
                    return S_mask
                S_temp = S_mask & ~(1 << i) & ~(1 << j)
                avail_temp = get_available(S_temp, (1 << m) - 1)
                if avail_temp == 0:
                    continue
                # Try to add 2 or 3 elements
                added = 0
                S_new = S_temp
                remaining = avail_temp
                while remaining != 0 and added < 3:
                    bit = remaining & (-remaining)
                    idx = bit.bit_length() - 1
                    S_new |= (1 << idx)
                    added += 1
                    remaining = get_available(S_new, (1 << m) - 1)
                
                if S_new.bit_count() > S_mask.bit_count():
                    S_mask = S_new
                    improved = True
                    break
        
        return S_mask
    
    # Apply 2-swap local search to best
    S = local_search_2swap(best_mask)
    if S.bit_count() > best_size:
        best_mask = S
        best_size = S.bit_count()
    
    # More random restarts with local search
    for trial in range(200):
        if time.time() >= deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        S = greedy_construct(order)
        S = local_search(S, max_iter=5000)
        S = local_search_2swap(S, max_iter=2000)
        if S.bit_count() > best_size:
            best_mask = S
            best_size = S.bit_count()
    
    # Final polish: try to add any remaining elements
    S = best_mask
    for _ in range(10):
        avail = get_available(S, (1 << m) - 1)
        if avail == 0:
            break
        bit = avail & (-avail)
        idx = bit.bit_length() - 1
        S |= (1 << idx)
    if S.bit_count() > best_size:
        best_mask = S
        best_size = S.bit_count()
    
    # Convert to array
    result = np.array([i for i in range(m) if (best_mask >> i) & 1], dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
