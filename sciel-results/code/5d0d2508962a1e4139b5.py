import time
from itertools import combinations
from math import gcd
from collections import Counter

def construct():
    deadline = time.time() + 85
    
    n_max = 128
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i, n_max + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    bad_triple_set = set()
    conflict_pairs = [[] for _ in range(n_max + 1)]
    
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n_max + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triple_set.add((a, b, c))
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))

    def is_compatible(x, S):
        """Check if x can be added to S without forming a bad triple."""
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True

    def greedy_build(order):
        selected = set()
        for x in order:
            if is_compatible(x, selected):
                selected.add(x)
        return selected

    # Order by number of conflicts (ascending)
    triple_count = [0] * (n_max + 1)
    for t in bad_triple_set:
        triple_count[t[0]] += 1
        triple_count[t[1]] += 1
        triple_count[t[2]] += 1
        
    order_safe = sorted(range(1, n_max + 1), key=lambda x: triple_count[x])
    
    best_set = greedy_build(order_safe)
    
    def local_search(selected, time_limit):
        selected = set(selected)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # Phase 1: Add
            for x in range(1, n_max + 1):
                if x not in selected:
                    if is_compatible(x, selected):
                        selected.add(x)
                        improved = True
                        break
            if improved:
                continue
                
            # Phase 2: Swap
            # Try to find x not in S and y in S such that (S - {y}) + {x} is valid.
            # This requires is_compatible(x, S - {y}).
            
            # Optimization:
            # For a given x, find all y in S that are in conflict with x.
            # If we remove one such y, x might become compatible.
            
            for x in range(1, n_max + 1):
                if x in selected:
                    continue
                    
                # Find conflicts of x with current S
                # A conflict is a pair (a,b) in S such that (a,b,x) is bad.
                # To add x, we must break all such pairs.
                # If there is only one such pair, we can remove one element from it.
                # If there are multiple, it's harder.
                
                # Get all pairs in conflict_pairs[x] that are subsets of S
                active_conflicts = []
                for a, b in conflict_pairs[x]:
                    if a in selected and b in selected:
                        active_conflicts.append((a, b))
                        
                if not active_conflicts:
                    # Should have been added in Phase 1
                    selected.add(x)
                    improved = True
                    break
                    
                # If we have exactly one active conflict (a,b), we can try removing a or b.
                if len(active_conflicts) == 1:
                    a, b = active_conflicts[0]
                    # Try removing a
                    if is_compatible(x, selected - {a}):
                        selected.remove(a)
                        selected.add(x)
                        improved = True
                        break
                    # Try removing b
                    if is_compatible(x, selected - {b}):
                        selected.remove(b)
                        selected.add(x)
                        improved = True
                        break
                else:
                    # Multiple conflicts. 
                    # Heuristic: Try removing the element that appears in the most active conflicts.
                    counter = Counter()
                    for a, b in active_conflicts:
                        counter[a] += 1
                        counter[b] += 1
                        
                    if counter:
                        y = counter.most_common(1)[0][0]
                        if is_compatible(x, selected - {y}):
                            selected.remove(y)
                            selected.add(x)
                            improved = True
                            break
                            
            if not improved:
                break
                
        return selected

    best_set = local_search(best_set, deadline - time.time() - 5)
    
    return list(best_set)
