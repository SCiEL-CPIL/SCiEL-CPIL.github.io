"""Optimized program for finding max subset of [1,128] avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct numbers have all pairwise LCMs equal."""
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if time.time() > deadline - 20:
            break
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements
    
    # Build conflict structures
    # For each node, store the set of (other1, other2) pairs that form a bad triple
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
    
    # Count conflicts per node
    conflict_count = [len(pairs_by_node[i]) for i in range(n)]
    
    def can_add(selected, node):
        """Check if adding node to selected doesn't complete any bad triple."""
        for other1, other2 in pairs_by_node[node]:
            if other1 in selected and other2 in selected:
                return False
        return True
    
    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected = set()
        for node in order:
            if time.time() > deadline - 10:
                break
            if can_add(selected, node):
                selected.add(node)
        return selected
    
    def local_search(selected, max_iter=5000):
        """Local search: try to add unselected nodes, then try swaps."""
        selected = set(selected)
        unselected = set(range(n)) - selected
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 15:
                break
            
            # Phase 1: Try to add any unselected node
            for node in list(unselected):
                if time.time() > deadline - 15:
                    break
                if can_add(selected, node):
                    selected.add(node)
                    unselected.remove(node)
                    improved = True
            
            # Phase 2: Try swaps - remove one, add two
            if not improved:
                # Try: for each selected node s, try removing it and adding two unselected
                # This is expensive, so limit attempts
                swap_found = False
                for s in list(selected):
                    if time.time() > deadline - 15:
                        break
                    if swap_found:
                        break
                    # Try removing s
                    selected.remove(s)
                    # Try adding pairs from unselected
                    unselected_list = list(unselected)
                    for i in range(len(unselected_list)):
                        if time.time() > deadline - 15:
                            break
                        u1 = unselected_list[i]
                        if can_add(selected, u1):
                            selected.add(u1)
                            unselected.remove(u1)
                            for j in range(i + 1, len(unselected_list)):
                                if time.time() > deadline - 15:
                                    break
                                u2 = unselected_list[j]
                                if can_add(selected, u2):
                                    selected.add(u2)
                                    unselected.remove(u2)
                                    swap_found = True
                                    improved = True
                                    break
                        if swap_found:
                            break
                        # Undo adding u1
                        selected.discard(u1)
                        unselected.add(u1)
                    # Re-add s if we didn't find a swap
                    if not swap_found:
                        selected.add(s)
                        unselected.discard(s)
        
        return selected
    
    # Multiple restarts with different orderings
    best_solution = set()
    best_size = 0
    
    # Ordering 1: by conflict count (ascending)
    order1 = sorted(range(n), key=lambda i: conflict_count[i])
    sol1 = greedy_init(order1)
    sol1 = local_search(sol1, max_iter=3000)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_solution = sol1
    
    # Ordering 2: reverse
    order2 = list(reversed(order1))
    sol2 = greedy_init(order2)
    sol2 = local_search(sol2, max_iter=3000)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_solution = sol2
    
    # Ordering 3: natural order
    order3 = list(range(n))
    sol3 = greedy_init(order3)
    sol3 = local_search(sol3, max_iter=3000)
    if len(sol3) > best_size:
        best_size = len(sol3)
        best_solution = sol3
    
    # Ordering 4: reverse natural
    order4 = list(range(n - 1, -1, -1))
    sol4 = greedy_init(order4)
    sol4 = local_search(sol4, max_iter=3000)
    if len(sol4) > best_size:
        best_size = len(sol4)
        best_solution = sol4
    
    # Additional local search passes on best solution
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            improved_sol = local_search(best_solution, max_iter=2000)
            if len(improved_sol) > len(best_solution):
                best_solution = improved_sol
    
    # Final attempt: try to add any remaining nodes
    if time.time() < deadline - 5:
        for node in range(n):
            if node not in best_solution and can_add(best_solution, node):
                best_solution.add(node)
    
    return [i + 1 for i in sorted(best_solution)]

# EVOLVE-BLOCK-END
