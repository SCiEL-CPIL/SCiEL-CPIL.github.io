"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 80.0  # 80s safety margin within 90s limit
    
    # Precompute gcd matrix and bad triples
    n = 128
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute for each pair (a,b), the set of x such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    # A triple (a,b,x) is bad if gcd(a,b) == gcd(a,x) == gcd(b,x)
    # For each pair (a,b), precompute which x make it bad
    bad_x_for_pair = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_mat[a][b]
            bad_xs = set()
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                    bad_xs.add(x)
            if bad_xs:
                bad_x_for_pair[(a, b)] = bad_xs
    
    # Also precompute: for each x, the set of pairs (a,b) that would form a bad triple with x
    bad_pairs_for_x = {x: set() for x in range(1, n + 1)}
    for (a, b), bad_xs in bad_x_for_pair.items():
        for x in bad_xs:
            bad_pairs_for_x[x].add((a, b))
    
    def is_valid_addition(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        selected_list = list(selected)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                if (a, b) in bad_pairs_for_x[x] or (b, a) in bad_pairs_for_x[x]:
                    return False
        return True
    
    def greedy_add(start_order):
        """Greedy add elements in given order."""
        selected = set()
        for x in start_order:
            if is_valid_addition(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 1: Greedy from full set removal
    def greedy_remove():
        """Start with all elements, remove those that create conflicts."""
        selected = set(range(1, n + 1))
        changed = True
        while changed:
            changed = False
            for x in list(selected):
                # Check if x conflicts with any pair in selected
                conflict_found = False
                selected_list = [y for y in selected if y != x]
                for i in range(len(selected_list)):
                    for j in range(i + 1, len(selected_list)):
                        a, b = selected_list[i], selected_list[j]
                        if (a, b) in bad_pairs_for_x[x] or (b, a) in bad_pairs_for_x[x]:
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    selected.remove(x)
                    changed = True
        return selected
    
    # Get initial solution from greedy removal
    best_set = greedy_remove()
    best_size = len(best_set)
    
    # Try multiple greedy-add restarts with different orderings
    import random
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Different ordering strategies
        if attempt % 3 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 3 == 1:
            # Sort by number of bad pairs (least constrained first)
            order = sorted(range(1, n + 1), key=lambda x: len(bad_pairs_for_x[x]))
        else:
            # Sort by value
            order = list(range(1, n + 1))
            if attempt % 6 == 2:
                order.reverse()
        
        candidate = greedy_add(order)
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
    
    # Try 1-for-2 swaps to improve
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            selected_list = list(best_set)
            
            for x in selected_list:
                if time.time() > deadline - 5.0:
                    break
                
                # Try removing x and adding two new elements
                remaining = best_set - {x}
                
                # Find candidates that can be added to remaining
                candidates = []
                for y in range(1, n + 1):
                    if y not in remaining and y != x:
                        if is_valid_addition(y, remaining):
                            candidates.append(y)
                
                # Try adding two candidates
                for i in range(len(candidates)):
                    if time.time() > deadline - 5.0:
                        break
                    y1 = candidates[i]
                    temp_set = remaining | {y1}
                    for j in range(i + 1, len(candidates)):
                        y2 = candidates[j]
                        if y2 == y1:
                            continue
                        if is_valid_addition(y2, temp_set):
                            new_set = temp_set | {y2}
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(new_set)
                                improved = True
                                break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
