import time
import random
from itertools import combinations, permutations

# Precompute 6-cycles for n=17
def _precompute_c6_cycles(n):
    """
    Returns a list of tuples, where each tuple contains 6 edge indices 
    corresponding to a 6-cycle in K_n.
    """
    cycles = []
    # Map edge (u, v) with u < v to index
    def edge_idx(u, v):
        if u > v: u, v = v, u
        return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
    
    # Iterate over all combinations of 6 vertices
    for comb in combinations(range(n), 6):
        # Generate all 6-cycles on these 6 vertices
        # A cycle is a permutation of the 6 vertices, but we can fix the first vertex 
        # and consider the remaining 5! / 2 = 60 permutations (due to rotation and reflection symmetry)
        # Actually, simpler: generate all permutations of the 6 vertices, 
        # and check if it forms a valid cycle (v0-v1-v2-v3-v4-v5-v0).
        # To avoid duplicates, we can fix v0 as the smallest vertex in the cycle? 
        # No, the cycle can start anywhere. 
        # Standard way: Fix v0. Permute v1..v5. 
        # There are 5! = 120 permutations. 
        # Each cycle is counted twice (clockwise and counter-clockwise).
        # So 60 unique cycles per set of 6 vertices.
        
        v0 = comb[0]
        rest = comb[1:]
        for perm in permutations(rest):
            # Cycle: v0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - v0
            # Check if this is a unique cycle representation.
            # To ensure we don't add duplicates, we can enforce an ordering constraint.
            # E.g., perm[0] < perm[4] (symmetry breaking)
            if perm[0] > perm[4]:
                continue
            
            edges = [
                edge_idx(v0, perm[0]),
                edge_idx(perm[0], perm[1]),
                edge_idx(perm[1], perm[2]),
                edge_idx(perm[2], perm[3]),
                edge_idx(perm[3], perm[4]),
                edge_idx(perm[4], v0)
            ]
            cycles.append(tuple(edges))
            
    return cycles

# Global cache for cycles
_CYCLES_CACHE = {}

def _get_cycles(n):
    if n not in _CYCLES_CACHE:
        _CYCLES_CACHE[n] = _precompute_c6_cycles(n)
    return _CYCLES_CACHE[n]

def _find_violation(colors, n):
    """
    Returns a tuple of 6 edge indices forming a monochromatic C6, or None.
    """
    cycles = _get_cycles(n)
    for cycle in cycles:
        c0 = colors[cycle[0]]
        if (c0 == colors[cycle[1]] == c0 == colors[cycle[2]] == c0 and
            c0 == colors[cycle[3]] == c0 and c0 == colors[cycle[4]] == c0 and
            c0 == colors[cycle[5]]):
            return cycle
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 0
    best_colors = []
    
    # Try n from 17 down to 1
    # We know 17 is possible.
    for target_n in range(17, 0, -1):
        if time.time() > deadline - 5:
            break
            
        total_edges = target_n * (target_n - 1) // 2
        cycles = _get_cycles(target_n)
        
        # If no cycles, any coloring is valid
        if not cycles:
            best_n = target_n
            best_colors = [0] * total_edges
            break
            
        # Local Search
        found = False
        # Number of restarts
        max_restarts = 20
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(total_edges)]
            
            # Iterate until no violation or time limit
            iters = 0
            max_iters = 5000
            while iters < max_iters:
                if time.time() > deadline - 5:
                    break
                    
                violation = _find_violation(colors, target_n)
                if not violation:
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
                    
                # Recolor a random edge in the violation
                edge_idx = random.choice(violation)
                old_c = colors[edge_idx]
                new_c = random.choice([c for c in range(4) if c != old_c])
                colors[edge_idx] = new_c
                iters += 1
                
            if found:
                break
                
        if found:
            break
        else:
            # If we didn't find a valid coloring for this n, continue to n-1
            continue
            
    # If no valid coloring found (should not happen for n<=17), fallback to n=1
    if best_n == 0:
        best_n = 1
        best_colors = []
        
    return {'n': best_n, 'colors': best_colors}
