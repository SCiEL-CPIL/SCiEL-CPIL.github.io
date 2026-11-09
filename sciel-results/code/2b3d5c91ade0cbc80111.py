"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute bad_triples_mask[a][b]
    # bad_triples_mask[a][b] is an integer bitset where bit c is 1 if {a,b,c} is a bad triple
    bad_triples_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_triples_mask[a][b] = mask
            bad_triples_mask[b][a] = mask

    # Precompute conflict graph: conflict_mask[a] is bitset of elements that conflict with a
    # Two elements a,b conflict if there exists c such that {a,b,c} is a bad triple
    # Actually, for the independent set formulation, we need:
    # A set S is valid iff for every pair (a,b) in S, no c in S is in bad_triples_mask[a][b]
    # This is a 3-uniform hypergraph independent set problem.
    # For branch-and-bound, we work with the "forbidden" structure.

    def greedy(ordering):
        selected = []
        selected_set = set()
        forbidden = 0  # Bitset of all c that would form a bad triple with some pair in selected
        
        for x in ordering:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            # Check if x is forbidden
            if (forbidden >> x) & 1:
                continue
            
            # x is valid. Add it.
            selected.append(x)
            selected_set.add(x)
            
            # Update forbidden mask
            for a in selected[:-1]:
                forbidden |= bad_triples_mask[a][x]
        
        return selected

    def local_search(selected, max_iterations=30):
        """Try to improve the solution via single-element swaps."""
        if not selected or time.time() > deadline:
            return selected
        
        all_elements = set(range(1, n + 1))
        best_set = set(selected)
        best_size = len(best_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            unselected = all_elements - best_set
            
            # Try removing one element and adding another
            for rem in list(best_set):
                if time.time() > deadline:
                    break
                
                new_set = best_set - {rem}
                
                # Compute forbidden for the reduced set
                new_forbidden = 0
                new_list = list(new_set)
                for i in range(len(new_list)):
                    for j in range(i + 1, len(new_list)):
                        new_forbidden |= bad_triples_mask[new_list[i]][new_list[j]]
                
                # Try adding any unselected element that's not forbidden
                for add in unselected:
                    if add in new_set:
                        continue
                    if (new_forbidden >> add) & 1:
                        continue
                    
                    # If add is not forbidden by any pair in new_set, 
                    # then {new_set, add} is valid (no bad triple)
                    candidate_set = new_set | {add}
                    if len(candidate_set) > best_size:
                        best_set = candidate_set
                        best_size = len(best_set)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return list(best_set)

    def branch_and_bound(available_mask, selected_list, current_size, best_size_ref):
        """
        Branch and bound to find a large valid subset.
        available_mask: bitset of elements still available to add
        selected_list: list of currently selected elements
        current_size: len(selected_list)
        best_size_ref: list containing [best_size] so it can be updated in place
        Returns: list of selected elements (best found in this branch)
        """
        # Upper bound: current_size + popcount(available_mask)
        # But we can do better: only elements in available_mask that are NOT forbidden by any pair in selected_list
        # can actually be added. Let's compute the truly available set.
        
        # Compute forbidden based on selected_list
        if selected_list:
            forbidden = 0
            sel = selected_list
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    forbidden |= bad_triples_mask[sel[i]][sel[j]]
            truly_available = available_mask & ~forbidden
        else:
            truly_available = available_mask
        
        upper_bound = current_size + bin(truly_available).count('1')
        
        if upper_bound <= best_size_ref[0]:
            return selected_list[:]  # No improvement possible
        
        if time.time() > deadline:
            return selected_list[:]
        
        # If no more elements can be added, return
        if truly_available == 0:
            return selected_list[:]
        
        # Choose the next element to branch on.
        # Heuristic: pick the element with highest "degree" in the conflict structure
        # among truly_available elements.
        # For each candidate x in truly_available, its "conflict degree" is:
        #   number of pairs (a,b) in selected_list such that x is in bad_triples_mask[a][b]
        # But since x is in truly_available, it's not forbidden by any pair in selected_list.
        # Instead, we consider: if we add x, how many other truly_available elements become forbidden?
        # This is a good heuristic for branching.
        
        candidates = []
        x = 1
        while x <= n:
            if (truly_available >> x) & 1:
                # Compute how many truly_available elements would be forbidden if we add x
                if selected_list:
                    new_forbidden = 0
                    for a in selected_list:
                        new_forbidden |= bad_triples_mask[a][x]
                    new_truly_available = truly_available & ~new_forbidden & ~(1 << x)
                    lost = bin(truly_available & ~(1 << x)).count('1') - bin(new_truly_available).count('1')
                else:
                    new_truly_available = truly_available & ~(1 << x)
                    lost = 0
                
                # Prefer elements that lose fewer candidates (more "independent")
                candidates.append((lost, x))
            x += 1
        
        if not candidates:
            return selected_list[:]
        
        # Sort by lost (ascending) - try elements that are least restrictive first
        candidates.sort()
        
        best_result = selected_list[:]
        
        # Branch: try adding each candidate, and also try skipping
        for lost, x in candidates:
            if time.time() > deadline:
                break
            
            # Option 1: Add x
            new_selected = selected_list + [x]
            if selected_list:
                new_forbidden = 0
                for a in selected_list:
                    new_forbidden |= bad_triples_mask[a][x]
                new_available = truly_available & ~new_forbidden & ~(1 << x)
            else:
                new_available = truly_available & ~(1 << x)
            
            result = branch_and_bound(new_available, new_selected, current_size + 1, best_size_ref)
            if len(result) > len(best_result):
                best_result = result
                if len(result) > best_size_ref[0]:
                    best_size_ref[0] = len(result)
            
            # Option 2: Skip x
            # Only worth trying if skipping could still lead to a better solution
            skip_available = truly_available & ~(1 << x)
            if bin(skip_available).count('1') + current_size > best_size_ref[0]:
                result_skip = branch_and_bound(skip_available, selected_list, current_size, best_size_ref)
                if len(result_skip) > len(best_result):
                    best_result = result_skip
                    if len(result_skip) > best_size_ref[0]:
                        best_size_ref[0] = len(result_skip)
        
        return best_result

    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Generate more diverse base orderings
    base_orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count() * 10 + x % 7),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count() * 10 + x % 5),
        sorted(range(1, n + 1), key=lambda x: (x & 1, x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x & 3, -x.bit_count(), x)),
    ]
    
    # Phase 1: Greedy + local search to get a good initial solution
    num_restarts = 3000
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i < len(base_orderings):
            ordering = base_orderings[i]
        else:
            ordering = list(range(1, n + 1))
            random.shuffle(ordering)
            
        selected = greedy(ordering)
        
        # Apply local search to improve
        if len(selected) > best_size:
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
            else:
                best_size = len(selected)
                best_result = selected
        elif len(selected) == best_size and best_result:
            # Try local search even on ties, might find a better arrangement
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
    
    # Phase 2: Branch and bound to try to find a better solution
    if time.time() < deadline - 5.0:  # Only if we have time
        full_mask = 0
        for i in range(1, n + 1):
            full_mask |= (1 << i)
        
        best_size_ref = [best_size]
        
        # Try B&B from different starting points
        # Start with empty set
        result = branch_and_bound(full_mask, [], 0, best_size_ref)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        
        # Try B&B starting from subsets of the current best solution
        if best_result and time.time() < deadline - 3.0:
            # Try removing each element from best_result and running B&B
            for rem in best_result[:10]:  # Limit to first 10 to save time
                if time.time() > deadline - 2.0:
                    break
                
                remaining = [x for x in best_result if x != rem]
                # Compute available: all elements not in remaining
                remaining_set = set(remaining)
                avail = 0
                for i in range(1, n + 1):
                    if i not in remaining_set:
                        avail |= (1 << i)
                
                result = branch_and_bound(avail, remaining, len(remaining), best_size_ref)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = result
    
    if not best_result:
        best_result = [1]
        
    return best_result
# EVOLVE-BLOCK-END
