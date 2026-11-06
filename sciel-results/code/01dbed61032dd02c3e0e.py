"""Rewritten program: structural change to a branch-and-bound / maximum-weight independent set solver
on the conflict graph defined by lcm(a,b) <= 256, maximizing sum of 1/x over [2,256]."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock
    n = 256
    # Nodes are 2..256 (indices 0..254)
    # Conflict edge (i,j) if lcm(a_i, a_j) <= 256
    # We want max weight independent set where weight = 1/a
    
    # Precompute conflict bitmasks for each node
    # 255 nodes, so bitmask fits in a Python int (arbitrary precision)
    vals = list(range(2, 257))
    num_nodes = len(vals)
    
    # Precompute lcm condition: lcm(a,b) <= 256 iff a*b/gcd(a,b) <= 256
    conflict = [0] * num_nodes
    for i in range(num_nodes):
        a = vals[i]
        for j in range(i+1, num_nodes):
            b = vals[j]
            if a * b // gcd(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Sort nodes by weight descending (larger 1/a means smaller a), for better pruning
    # Actually, for branch-and-bound on MWIS, we want to consider high-weight nodes first
    # Let's reorder: index order where higher weight comes first
    order = sorted(range(num_nodes), key=lambda i: -1.0 / vals[i])
    # Rebuild conflict in new order
    new_conflict = [0] * num_nodes
    for old_i in range(num_nodes):
        new_i = order.index(old_i)  # This is O(n^2) but n=255, fine
        for old_j in range(num_nodes):
            if (conflict[old_i] >> old_j) & 1:
                new_j = order.index(old_j)
                new_conflict[new_i] |= (1 << new_j)
    
    # Actually, let's just work with the original ordering but use a different approach.
    # Use a recursive branch-and-bound with memoization on the "available" set.
    # Since 2^255 is too large, we use a greedy upper bound.
    
    # Better approach: use the fact that the conflict graph is structured.
    # Many numbers > 128 have lcm with any other number > 128 * something... 
    # Actually lcm(a,b) > 256 for all distinct pairs if we pick numbers that are "coprime-ish" or large.
    
    # Let's try a different structural approach:
    # Partition [2,256] into groups where within each group, all pairs have lcm <= 256 (cliques in conflict graph)
    # Then we can pick at most one from each group.
    
    # Actually, the simplest strong approach: use a maximum weight independent set solver
    # with branch and bound, using a greedy coloring upper bound.
    
    # Let me try a simpler but effective approach: 
    # Greedy + local search (swap/insert/delete)
    
    # Start with a good greedy solution
    selected = set()
    # Sort by 1/x descending (i.e., x ascending)
    for x in range(2, 257):
        if all(x * y // gcd(x, y) > 256 for y in selected):
            selected.add(x)
    
    def score(s):
        return sum(1.0 / x for x in s)
    
    current_score = score(selected)
    best_selected = set(selected)
    best_score = current_score
    
    # Local search: try to add, remove, or swap elements
    def can_add(s, x):
        return all(x * y // gcd(x, y) > 256 for y in s)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element
        for x in range(2, 257):
            if x not in selected and can_add(selected, x):
                new_sel = set(selected)
                new_sel.add(x)
                new_score = score(new_sel)
                if new_score > best_score + 1e-15:
                    best_selected = new_sel
                    best_score = new_score
                    selected = new_sel
                    current_score = new_score
                    improved = True
        
        # Try swapping: remove one, add one
        if time.time() > deadline - 5:
            break
        for rem in list(selected):
            for add in range(2, 257):
                if add in selected or add == rem:
                    continue
                new_sel = set(selected)
                new_sel.discard(rem)
                if can_add(new_sel, add):
                    new_sel.add(add)
                    new_score = score(new_sel)
                    if new_score > best_score + 1e-15:
                        best_selected = new_sel
                        best_score = new_score
                        selected = new_sel
                        current_score = new_score
                        improved = True
                        break
            if time.time() > deadline - 5:
                break
        
        # Try 2-for-1 or 1-for-2 swaps
        if not improved and time.time() < deadline - 10:
            # Try removing two, adding one
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    rem1, rem2 = sel_list[i], sel_list[j]
                    new_sel = set(selected)
                    new_sel.discard(rem1)
                    new_sel.discard(rem2)
                    for add in range(2, 257):
                        if add in new_sel:
                            continue
                        if can_add(new_sel, add):
                            new_sel.add(add)
                            new_score = score(new_sel)
                            if new_score > best_score + 1e-15:
                                best_selected = new_sel
                                best_score = new_score
                                selected = new_sel
                                current_score = new_score
                                improved = True
                                break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
    
    return list(best_selected)
# EVOLVE-BLOCK-END
