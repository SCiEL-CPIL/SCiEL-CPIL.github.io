"""Rewritten program: exact maximum independent set via branch-and-bound with bitsets."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Build nodes: (residue, modulus)
    nodes = []
    for m in moduli_list:
        for r in range(m):
            nodes.append((r, m))
    
    n = len(nodes)
    
    # Build conflict bitmasks: conflict[i] is a bitset of all j that conflict with i
    # n=1004, so we need 1004-bit integers (Python handles big ints natively)
    conflict = [0] * n
    
    for i in range(n):
        r1, m1 = nodes[i]
        for j in range(i + 1, n):
            r2, m2 = nodes[j]
            if not _is_disjoint(r1, m1, r2, m2):
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Degree of each node
    degree = [bin(conflict[i]).count('1') for i in range(n)]
    
    best_set = []
    best_size = 0
    deadline_reached = [False]
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_bound(available_mask):
        """Upper bound via greedy coloring on available subgraph."""
        if available_mask == 0:
            return 0
        
        colors = 0
        uncolored = available_mask
        
        while uncolored:
            # Greedily color one color class at a time
            # Pick vertex with max degree in uncolored
            best_v = -1
            best_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                d = popcount(conflict[v] & uncolored)
                if d > best_deg:
                    best_deg = d
                    best_v = v
                temp &= temp - 1
            
            # Color class: start with best_v, add non-conflicting vertices
            color_class_mask = 1 << best_v
            uncolored &= ~(1 << best_v)
            
            # Try to add more vertices to this color class
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if not (conflict[v] & color_class_mask):
                    color_class_mask |= (1 << v)
                    uncolored &= ~(1 << v)
                temp &= temp - 1
            
            colors += 1
        
        return colors
    
    def branch(available_mask, current_set):
        nonlocal best_set, best_size, deadline_reached
        
        if deadline_reached[0]:
            return
        
        if time.time() > deadline:
            deadline_reached[0] = True
            return
        
        if available_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound
        ub = greedy_bound(available_mask)
        if len(current_set) + ub <= best_size:
            return
        
        # If upper bound equals possible max, take all
        # Select pivot: max degree vertex in available
        pivot = -1
        max_deg = -1
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            d = popcount(conflict[v] & available_mask)
            if d > max_deg:
                max_deg = d
                pivot = v
            temp &= temp - 1
        
        # Branch 1: Include pivot
        new_available = available_mask & ~(1 << pivot) & ~conflict[pivot]
        branch(new_available, current_set + [pivot])
        
        if deadline_reached[0]:
            return
        
        # Branch 2: Exclude pivot
        new_available2 = available_mask & ~(1 << pivot)
        branch(new_available2, current_set)
    
    initial_mask = (1 << n) - 1
    branch(initial_mask, [])
    
    if deadline_reached[0] and best_size == 0:
        # Fallback to greedy
        random.seed(42)
        order = list(range(n))
        random.shuffle(order)
        chosen = []
        chosen_mask = 0
        for i in order:
            if not (conflict[i] & chosen_mask):
                chosen.append(i)
                chosen_mask |= (1 << i)
        best_set = chosen
        best_size = len(chosen)
    
    result = [[nodes[i][0], nodes[i][1]] for i in best_set]
    
    if not result:
        result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return result
# EVOLVE-BLOCK-END
