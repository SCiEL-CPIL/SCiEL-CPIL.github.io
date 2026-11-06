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
    
    # Precompute degree for each element (number of bad triples it participates in)
    degree = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
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
        """Fast local search: try additions, 1-out-1-in swaps, and 2-out-2-in swaps."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            
            # 1. Try additions
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
            
            # 2. Try 1-out-1-in swaps
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
            
            # 3. Try 2-out-2-in swaps (limited to avoid explosion)
            # Only try if we have time and S is large enough
            if time.time() < deadline - 10 and len(S) > 50:
                S_list = list(S)
                # Sample some pairs to swap to keep it fast
                # We try to remove x1, x2 and add y1, y2
                # To be efficient, we only try swaps where we remove low-degree elements first?
                # Or just random samples?
                # Let's try a few random pairs
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    i, j = random.sample(range(len(S_list)), 2)
                    x1, x2 = S_list[i], S_list[j]
                    S.remove(x1)
                    S.remove(x2)
                    
                    # Try to add two elements
                    found_y1 = None
                    found_y2 = None
                    for y1 in elements:
                        if y1 not in S and can_add(y1, S):
                            S.add(y1)
                            found_y1 = y1
                            for y2 in elements:
                                if y2 not in S and y2 != y1 and can_add(y2, S):
                                    S.add(y2)
                                    found_y2 = y2
                                    break
                            break
                            
                    if found_y1 is not None and found_y2 is not None:
                        improved = True
                    else:
                        # Rollback
                        if found_y1: S.remove(found_y1)
                        if found_y2: S.remove(found_y2)
                        S.add(x1)
                        S.add(x2)
                        # Update S_list if necessary, but it's just a snapshot
                        # Actually, S_list is stale after modifications, but we break out of inner loops
                        # and use list(S) again in next iteration if improved
                        # If not improved, we restored S.
                        # We should break if we found a swap? No, keep trying.
                        # But S_list is stale. Let's just rely on the next round's list(S) if we continue.
                        # For simplicity, we just accept that S_list is a snapshot from start of round.
                        pass
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
    # Sort by degree (conflict count) descending: high conflict elements are harder to place, 
    # but placing them first might block many others? 
    # Or sort by degree ascending: place "safe" elements first to fill up, then try hard ones?
    # Usually for independent set, placing high degree first is better for pruning, 
    # but for greedy construction, placing low degree first often yields larger sets 
    # because they conflict with fewer things.
    
    orderings = [
        sorted(elements, key=lambda x: (degree[x], x)), # Low degree first
        sorted(elements, key=lambda x: (-degree[x], x)), # High degree first
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
