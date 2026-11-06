"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1,180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    
    Strategy:
    1. Precompute all forbidden triples (a,b,c) with a<b<c where 1/a = 1/b + 1/c
       (equivalently a = bc/(b+c) is integer).
    2. Use a greedy + local search approach to find a large independent set
       in the 3-uniform hypergraph of forbidden triples.
    3. Use parity-based partitioning as a heuristic to guide the search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    N = 180
    
    # Precompute all forbidden triples
    # For b < c, a = b*c/(b+c) must be integer and a < b (since 1/a = 1/b + 1/c > 1/c, so a < c; and 1/a > 1/b so a < b)
    # Actually: 1/a = 1/b + 1/c => a = bc/(b+c). Since b,c >= 1, a < min(b,c).
    # So for each pair (b,c) with b < c, check if bc % (b+c) == 0 and a = bc//(b+c) < b.
    
    forbidden = []  # list of triples (a, b, c)
    triple_set = set()
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triple = (a, b, c)
                    if triple not in triple_set:
                        triple_set.add(triple)
                        forbidden.append(triple)
    
    # Build adjacency: for each element, which triples it's in
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(forbidden):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    
    def is_valid(subset):
        """Check if subset has no forbidden triple."""
        s = set(subset)
        for (a, b, c) in forbidden:
            if a in s and b in s and c in s:
                return False
        return True
    
    def count_conflicts(subset):
        """Count number of forbidden triples fully contained in subset."""
        s = set(subset)
        count = 0
        for (a, b, c) in forbidden:
            if a in s and b in s and c in s:
                count += 1
        return count
    
    # Greedy construction: start with all elements, remove elements to eliminate all triples
    # Use a heuristic: remove the element that is in the most remaining triples
    
    def greedy_remove():
        """Greedy: start with all, remove element in most triples until valid."""
        current = set(range(1, N + 1))
        # Track which triples are still "active" (all elements in current)
        active_triples = set(range(len(forbidden)))
        
        while active_triples:
            # Count for each element in current, how many active triples it's in
            counts = {e: 0 for e in current}
            for t_idx in active_triples:
                a, b, c = forbidden[t_idx]
                if a in current:
                    counts[a] += 1
                if b in current:
                    counts[b] += 1
                if c in current:
                    counts[c] += 1
            
            # Find element with max count
            if not counts:
                break
            max_count = max(counts.values())
            if max_count == 0:
                break
            
            # Remove one of the elements with max count (break ties by smaller value to keep larger)
            candidates = [e for e in current if counts[e] == max_count]
            # Prefer removing smaller elements to keep larger ones (they're in fewer triples typically)
            to_remove = min(candidates)
            
            current.discard(to_remove)
            
            # Update active triples
            new_active = set()
            for t_idx in active_triples:
                a, b, c = forbidden[t_idx]
                if a in current and b in current and c in current:
                    new_active.add(t_idx)
            active_triples = new_active
        
        return current
    
    # Local search: start from greedy, try swapping elements
    def local_search(initial_set, max_iters=50000):
        """Local search: try to add elements back or swap to improve."""
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        
        # Precompute for each element not in current, which triples would be violated if added
        # For speed, track which triples are "blocked" by missing elements
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding elements not in current
            for e in range(1, N + 1):
                if e in current:
                    continue
                # Check if adding e creates any triple
                # e is in triples elem_to_triples[e]
                can_add = True
                for t_idx in elem_to_triples[e]:
                    a, b, c = forbidden[t_idx]
                    # Check if the other two elements are in current
                    others = [x for x in (a, b, c) if x != e]
                    if len(others) == 2 and others[0] in current and others[1] in current:
                        can_add = False
                        break
                if can_add:
                    current.add(e)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    improved = True
            
            if not improved:
                # Try swapping: remove one element, add another
                for e_remove in list(current):
                    # Temporarily remove
                    current.discard(e_remove)
                    
                    # Try to add any element
                    for e_add in range(1, N + 1):
                        if e_add in current or e_add == e_remove:
                            continue
                        can_add = True
                        for t_idx in elem_to_triples[e_add]:
                            a, b, c = forbidden[t_idx]
                            others = [x for x in (a, b, c) if x != e_add]
                            if len(others) == 2 and others[0] in current and others[1] in current:
                                can_add = False
                                break
                        if can_add:
                            current.add(e_add)
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(current)
                            improved = True
                            break
                        elif can_add:
                            pass
                    
                    # Restore if we didn't improve
                    if e_remove not in current and e_remove != e_add if 'e_add' in dir() else True:
                        # This is getting complicated, let's simplify
                        pass
                    
                    # Actually, let's just restore e_remove if we didn't add anything
                    if len(current) < len(best) or len(current) == len(best_size):
                        current.add(e_remove)
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy remove
    greedy_result = greedy_remove()
    best_result = set(greedy_result)
    
    # Strategy 2: Local search from greedy
    ls_result = local_search(greedy_result)
    if len(ls_result) > len(best_result):
        best_result = ls_result
    
    # Strategy 3: Try parity-based construction
    # Odds only
    odds = set(range(1, N + 1, 2))
    if is_valid(odds):
        if len(odds) > len(best_result):
            best_result = odds
    
    # Evens only
    evens = set(range(2, N + 1, 2))
    if is_valid(evens):
        if len(evens) > len(best_result):
            best_result = evens
    
    # Strategy 4: Greedy with different tie-breaking
    def greedy_remove_v2():
        """Greedy: remove element in most triples, break ties by larger value first."""
        current = set(range(1, N + 1))
        active_triples = set(range(len(forbidden)))
        
        while active_triples:
            counts = {e: 0 for e in current}
            for t_idx in active_triples:
                a, b, c = forbidden[t_idx]
                if a in current:
                    counts[a] += 1
                if b in current:
                    counts[b] += 1
                if c in current:
                    counts[c] += 1
            
            if not counts:
                break
            max_count = max(counts.values())
            if max_count == 0:
                break
            
            candidates = [e for e in current if counts[e] == max_count]
            to_remove = max(candidates)  # Remove larger to keep smaller
            
            current.discard(to_remove)
            
            new_active = set()
            for t_idx in active_triples:
                a, b, c = forbidden[t_idx]
                if a in current and b in current and c in current:
                    new_active.add(t_idx)
            active_triples = new_active
        
        return current
    
    greedy2_result = greedy_remove_v2()
    if len(greedy2_result) > len(best_result):
        best_result = set(greedy2_result)
    else:
        # Try local search from this too
        ls2_result = local_search(greedy2_result, max_iters=20000)
        if len(ls2_result) > len(best_result):
            best_result = ls2_result
    
    # Strategy 5: Randomized restarts
    if time.time() < deadline - 5:
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            
            # Randomized greedy
            current = set(range(1, N + 1))
            active_triples = set(range(len(forbidden)))
            
            while active_triples:
                counts = {e: 0 for e in current}
                for t_idx in active_triples:
                    a, b, c = forbidden[t_idx]
                    if a in current:
                        counts[a] += 1
                    if b in current:
                        counts[b] += 1
                    if c in current:
                        counts[c] += 1
                
                if not counts:
                    break
                max_count = max(counts.values())
                if max_count == 0:
                    break
                
                candidates = [e for e in current if counts[e] == max_count]
                to_remove = random.choice(candidates)
                
                current.discard(to_remove)
                
                new_active = set()
                for t_idx in active_triples:
                    a, b, c = forbidden[t_idx]
                    if a in current and b in current and c in current:
                        new_active.add(t_idx)
                active_triples = new_active
            
            if len(current) > len(best_result):
                best_result = set(current)
            
            # Quick local search
            ls_rand = local_search(current, max_iters=5000)
            if len(ls_rand) > len(best_result):
                best_result = ls_rand
    
    # Final verification
    if not is_valid(best_result):
        # Fallback: use the simple range
        best_result = set(range(91, 181))
    
    if not best_result:
        best_result = {180}
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
