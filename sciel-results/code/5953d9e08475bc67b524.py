"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max:
                continue
            if c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _is_valid_addition(S, x, triples, elem_to_triples):
    """Check if x can be added to S without creating a forbidden triple."""
    for tidx in elem_to_triples[x]:
        a, b, c = triples[tidx]
        # Check if the other two elements are in S
        others = [y for y in (a, b, c) if y != x]
        if others[0] in S and others[1] in S:
            return False
    return True

def _find_valid_swap(S, x_not_in_S, u_in_S, triples, elem_to_triples):
    """Check if swapping u (in S) with x (not in S) results in a valid set."""
    # New set is S' = S \ {u} U {x}
    # We need to check if x can be added to S \ {u}
    # This is equivalent to checking if x conflicts with any pair in S \ {u}
    for tidx in elem_to_triples[x]:
        a, b, c = triples[tidx]
        others = [y for y in (a, b, c) if y != x]
        # If both others are in S, and neither is u, then conflict.
        # If one is u, it's removed, so no conflict from this triple.
        if (others[0] in S and others[1] in S) and (u not in others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    n_max = 180
    all_elements = list(range(1, n_max + 1))
    
    # Build the hypergraph
    triples = _build_triples(n_max)
    
    # Pre-compute incidence
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Initial Greedy Construction
    def greedy_construct():
        S = set(all_elements)
        # Count conflicts
        # triple_incount: how many elements of triple i are in S
        triple_incount = [0] * len(triples)
        for idx, (a, b, c) in enumerate(triples):
            cnt = (a in S) + (b in S) + (c in S)
            triple_incount[idx] = cnt
        
        # elem_conflict_count: number of triples where all 3 elements are in S
        elem_conflict_count = {i: 0 for i in range(1, n_max + 1)}
        for idx, cnt in enumerate(triple_incount):
            if cnt == 3:
                a, b, c = triples[idx]
                elem_conflict_count[a] += 1
                elem_conflict_count[b] += 1
                elem_conflict_count[c] += 1
        
        total_conflicts = sum(1 for cnt in triple_incount if cnt == 3)
        
        while total_conflicts > 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                break
            
            # Find element with max conflict count
            max_conflict = 0
            best_elem = None
            # Randomize tie-breaking
            candidates = [x for x in S if elem_conflict_count[x] == max_conflict]
            if not candidates:
                # Find actual max
                for x in S:
                    if elem_conflict_count[x] > max_conflict:
                        max_conflict = elem_conflict_count[x]
                        candidates = [x]
                if max_conflict == 0:
                    break
            else:
                # Re-find max properly
                max_conflict = 0
                for x in S:
                    if elem_conflict_count[x] > max_conflict:
                        max_conflict = elem_conflict_count[x]
                candidates = [x for x in S if elem_conflict_count[x] == max_conflict]
            
            if not candidates:
                break
                
            best_elem = random.choice(candidates)
            
            # Remove best_elem
            S.remove(best_elem)
            
            # Update counts
            for tidx in elem_to_triples[best_elem]:
                old_cnt = triple_incount[tidx]
                if old_cnt == 3:
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    for x in (a, b, c):
                        if x != best_elem:
                            elem_conflict_count[x] -= 1
                triple_incount[tidx] = old_cnt - 1
        
        return S

    # Run greedy multiple times to find a good starting point
    best_S = set()
    best_size = 0
    
    num_restarts = 20
    for i in range(num_restarts):
        S = greedy_construct()
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Local Search / Improvement
    # We try to add elements or swap to increase size
    # Since best_S is likely maximal, we try swaps to find a different maximal set
    # that might allow an addition later? 
    # Actually, if we find a non-maximal set, we can add.
    
    # Let's try a simple local search:
    # 1. Try to add any element not in S.
    # 2. If none can be added, try swaps.
    
    current_S = best_S.copy()
    
    # Pre-compute list of elements not in S for faster access
    # We'll just iterate over all_elements
    
    # Optimization: Cache validity checks?
    
    for iteration in range(1000): # Max iterations for local search
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
            
        improved = False
        
        # Try to add
        for x in all_elements:
            if x not in current_S:
                if _is_valid_addition(current_S, x, triples, elem_to_triples):
                    current_S.add(x)
                    improved = True
                    break # Restart loop with new S
        
        if improved:
            continue
            
        # If no addition, try a swap
        # Pick random u in S, x not in S
        if not current_S:
            break
            
        u = random.choice(list(current_S))
        # Pick random x not in S
        # To be efficient, we can pick random from all_elements and check
        # Or maintain a list
        x = random.choice(all_elements)
        if x in current_S:
            continue
            
        if _find_valid_swap(current_S, x, u, triples, elem_to_triples):
            current_S.remove(u)
            current_S.add(x)
            # Accept swap
            
    # Final check: try to add again
    for x in all_elements:
        if x not in current_S:
            if _is_valid_addition(current_S, x, triples, elem_to_triples):
                current_S.add(x)
                
    if len(current_S) > best_size:
        best_S = current_S.copy()
        
    result = sorted(best_S)
    if not result:
        result = list(range(91, 181))
        
    return result

# EVOLVE-BLOCK-END
