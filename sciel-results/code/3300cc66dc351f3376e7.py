"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from math import gcd

def construct():
    """
    Solve maximum weight independent set on conflict graph for 1..511.
    Uses branch-and-bound with fractional knapsack relaxation.
    """
    n = 512
    # Vertices 1..511, weight(a) = 1/(512-a)
    # Sort by weight descending
    vertices = list(range(511, 0, -1))  # already in decreasing weight order
    
    weights = [1.0 / (n - a) for a in vertices]
    m = len(vertices)
    
    # Precompute conflict matrix as bitsets (using Python integers as bitmasks)
    # conflict_mask[i] has bit j set if vertices[i] conflicts with vertices[j]
    conflict_mask = [0] * m
    for i in range(m):
        a = vertices[i]
        mask = 0
        for j in range(i + 1, m):
            b = vertices[j]
            if gcd(a, b) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    best_weight = 0.0
    best_solution = []
    
    # Use iterative deepening or simple branch-and-bound with good pruning
    # Key: process in order, maintain a bitmask of available vertices
    
    def solve(pos, current_sum, selected, available):
        nonlocal best_weight, best_solution
        
        # Upper bound: current_sum + sum of weights of available vertices from pos onward
        # Compute quickly
        ub = current_sum
        for i in range(pos, m):
            if (available >> i) & 1:
                ub += weights[i]
        
        if ub <= best_weight:
            return
        
        if pos >= m:
            if current_sum > best_weight:
                best_weight = current_sum
                best_solution = list(selected)
            return
        
        # Option 1: select vertex pos if available
        if (available >> pos) & 1:
            new_available = available & ~(1 << pos) & ~conflict_mask[pos]
            selected.append(vertices[pos])
            solve(pos + 1, current_sum + weights[pos], selected, new_available)
            selected.pop()
        
        # Option 2: skip vertex pos
        new_available = available & ~(1 << pos)
        solve(pos + 1, current_sum, selected, new_available)
    
    initial_available = (1 << m) - 1
    solve(0, 0.0, [], initial_available)
    
    return best_solution

# EVOLVE-BLOCK-END
