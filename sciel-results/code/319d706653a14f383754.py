"""Improved program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
from itertools import combinations
from math import gcd

def construct():
    """
    Find a nonempty subset of [1,128] with no three distinct elements whose
    three pairwise gcds are equal. Maximize cardinality.
    """
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    # Find all forbidden triples and store as sorted tuples
    forbidden = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    forbidden.add((a, b, c))
    
    def is_forbidden_triple(a, b, c):
        """Check if three distinct numbers form a forbidden triple."""
        t = tuple(sorted((a, b, c)))
        return t in forbidden
    
    # Precompute conflict degree for each number
    conflict_degree = [0] * (n + 1)
    for (a, b, c) in forbidden:
        conflict_degree[a] += 1
        conflict_degree[b] += 1
        conflict_degree[c] += 1
    
    # Sort numbers by conflict degree (ascending) - least conflicting first
    sorted_numbers = sorted(numbers, key=lambda x: (conflict_degree[x], x))
    
    # Greedy construction
    selected = []
    selected_set = set()
    
    for x in sorted_numbers:
        # Check if adding x creates a forbidden triple with any two in selected
        can_add = True
        for a, b in combinations(selected, 2):
            if is_forbidden_triple(a, b, x):
                can_add = False
                break
        if can_add:
            selected.append(x)
            selected_set.add(x)
    
    # Local improvement: try to replace elements to increase cardinality
    improved = True
    max_iter = 100
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i, x in enumerate(selected):
            # Try removing x and adding some y not in selected
            selected_without_x = selected[:i] + selected[i+1:]
            selected_set_without_x = selected_set - {x}
            
            for y in numbers:
                if y in selected_set_without_x:
                    continue
                # Check if adding y to selected_without_x is valid
                valid = True
                for a, b in combinations(selected_without_x, 2):
                    if is_forbidden_triple(a, b, y):
                        valid = False
                        break
                if valid:
                    # Perform the replacement
                    selected[i] = y
                    selected_set.discard(x)
                    selected_set.add(y)
                    improved = True
                    break
            if improved:
                break
    
    # Additional improvement: try adding any remaining numbers
    for y in numbers:
        if y in selected_set:
            continue
        valid = True
        for a, b in combinations(selected, 2):
            if is_forbidden_triple(a, b, y):
                valid = False
                break
        if valid:
            selected.append(y)
            selected_set.add(y)
    
    return selected

# EVOLVE-BLOCK-END
