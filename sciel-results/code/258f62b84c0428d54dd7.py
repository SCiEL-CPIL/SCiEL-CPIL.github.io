"""Backtracking with branch-and-bound for maximum independent set in hypergraph."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin before 60s limit
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i * j // math.gcd(i, j)
    
    # Precompute conflict triples: for each pair (a,b), list of c that form conflict
    # A conflict triple is {a,b,c} where lcm(a,b)=lcm(a,c)=lcm(b,c)
    pair_conflicts = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            pair_conflicts[(a, b)] = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    pair_conflicts[(a, b)].append(c)
                    pair_conflicts[(a, c)].append(b)
                    pair_conflicts[(b, c)].append(a)
    
    # For each element x, precompute which pairs (a,b) would be "completed" by adding x
    creates_conflict = [[] for _ in range(n + 1)]
    for (a, b), c_list in pair_conflicts.items():
        for c in c_list:
            creates_conflict[c].append((a, b))
    
    # Precompute for each x: the set of elements y such that there exists z with conflict
    # This helps with faster checking
    # conflict_pairs[x] = list of (a,b) where x conflicts with both a and b
    # Actually creates_conflict already does this
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a conflict triple."""
        if x in selected_set:
            return True
        for (a, b) in creates_conflict[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    def local_search(selected, max_iters=50000):
        """Local search with 1-swap and 2-swap moves."""
        selected = set(selected)
        current_size = len(selected)
        no_improve = 0
        iteration = 0
        
        while iteration < max_iters and time.time() < deadline - 3:
            iteration += 1
            improved = False
            
            # Try 1-swap: remove one, add one
            for x in list(selected):
                for y in range(1, n + 1):
                    if y in selected:
                        continue
                    new_selected = set(selected)
                    new_selected.discard(x)
                    if can_add(new_selected, y):
                        new_selected.add(y)
                        if len(new_selected) > current_size:
                            selected = new_selected
                            current_size = len(selected)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap: remove two, add two
                improved = False
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i + 1, len(sel_list)):
                        x1, x2 = sel_list[i], sel_list[j]
                        for y1 in range(1, n + 1):
                            if y1 in selected:
                                continue
                            for y2 in range(y1 + 1, n + 1):
                                if y2 in selected:
                                    continue
                                new_selected = set(selected)
                                new_selected.discard(x1)
                                new_selected.discard(x2)
                                if can_add(new_selected, y1):
                                    new_selected.add(y1)
                                    if can_add(new_selected, y2):
                                        new_selected.add(y2)
                                        if len(new_selected) > current_size:
                                            selected = new_selected
                                            current_size = len(selected)
                                            improved = True
                                            break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve > 30:
                    break
            else:
                no_improve = 0
        
        return selected
    
    # Multiple greedy initializations with different orderings
    best = set()
    
    # Ordering 1: natural order
    best = greedy_solve(elements)
    best = local_search(best)
    
    # Ordering 2: reverse
    best2 = greedy_solve(elements[::-1])
    best2 = local_search(best2)
    if len(best2) > len(best):
        best = best2
    
    # Ordering 3: large numbers first [65,128] then rest
    order3 = list(range(65, 129)) + list(range(1, 65))
    best3 = greedy_solve(order3)
    best3 = local_search(best3)
    if len(best3) > len(best):
        best = best3
    
    # Ordering 4: conflict-count-sorted (fewest conflicts first)
    conflict_counts = [len(creates_conflict[x]) for x in range(n + 1)]
    order4 = sorted(elements, key=lambda x: conflict_counts[x])
    best4 = greedy_solve(order4)
    best4 = local_search(best4)
    if len(best4) > len(best):
        best = best4
    
    # Ordering 5: conflict-count-sorted descending (most conflicts first - exclude them)
    order5 = sorted(elements, key=lambda x: -conflict_counts[x])
    best5 = greedy_solve(order5)
    best5 = local_search(best5)
    if len(best5) > len(best):
        best = best5
    
    # Ordering 6: powers of 2 first, then others
    powers_of_2 = [2**k for k in range(8) if 2**k <= 128]
    others6 = [x for x in elements if x not in powers_of_2]
    order6 = powers_of_2 + others6
    best6 = greedy_solve(order6)
    best6 = local_search(best6)
    if len(best6) > len(best):
        best = best6
    
    # Random restarts
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        random.shuffle(elements)
        best_r = greedy_solve(elements)
        best_r = local_search(best_r, max_iters=20000)
        if len(best_r) > len(best):
            best = best_r
    
    # Final local search on best with more iterations
    if time.time() < deadline - 3:
        best = local_search(best, max_iters=100000)
    
    return list(best)
# EVOLVE-BLOCK-END
