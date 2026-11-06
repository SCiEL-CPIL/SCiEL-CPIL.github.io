import time
import random
import math
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 60s limit
    
    n = 256
    # Build conflict graph using bitmasks for speed
    # conflicts[i] is a bitmask where bit j is set if i conflicts with j
    conflicts = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Precompute degree for ordering
    degrees = [(conflicts[i]).bit_count() for i in range(n + 1)]
    
    # Initial solution: All odd numbers (no conflicts among odds)
    best_S = set(range(1, n + 1, 2))
    best_size = len(best_S)
    
    # Greedy coloring bound (Tomita-style)
    def greedy_color_bound(candidates):
        """Proper greedy coloring: assign colors so adjacent vertices get different colors.
        Returns the number of colors used as an upper bound on clique size (and thus MIS size)."""
        colors = {}
        color_classes = []  # each element is a bitmask of vertices with that color
        remaining = candidates
        
        while remaining:
            # Pick a vertex from remaining
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            remaining ^= v_bit
            
            # Try to assign the lowest available color
            assigned = False
            for c_idx, c_mask in enumerate(color_classes):
                # Check if v conflicts with any vertex in this color class
                if (conflicts[v] & c_mask) == 0:
                    color_classes[c_idx] = c_mask | v_bit
                    assigned = True
                    break
            
            if not assigned:
                color_classes.append(v_bit)
        
        return len(color_classes)
    
    # BnB with proper coloring bound
    def bnb(candidates, current_size, current_set):
        """Branch and bound for MIS."""
        nonlocal best_size, best_S
        if time.time() > deadline:
            return
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_S = set(current_set)
            return
        
        ub = current_size + greedy_color_bound(candidates)
        if ub <= best_size:
            return
        
        # Pick vertex with highest degree in candidates subgraph
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            deg = (candidates & conflicts[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= v_bit
        
        if best_v < 0:
            return
        
        # Branch 1: Include best_v
        current_set.append(best_v)
        new_candidates = candidates & ~(conflicts[best_v] | (1 << best_v))
        bnb(new_candidates, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        bnb(new_candidates, current_size, current_set)
    
    # Run BnB with the full candidate set
    full_candidates = 0
    for i in range(1, n + 1):
        full_candidates |= (1 << i)
    
    bnb(full_candidates, 0, [])
    
    # Efficient local search polish using bitmask
    if time.time() < deadline - 3:
        S_mask = 0
        for s in best_S:
            S_mask |= (1 << s)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try adding each vertex not in S
            for x in range(1, n + 1):
                if (S_mask >> x) & 1:
                    continue
                # Check if x conflicts with any element in S
                if (conflicts[x] & S_mask) == 0:
                    S_mask |= (1 << x)
                    improved = True
        
        # Convert back to set
        S = set()
        temp = S_mask
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            S.add(v)
            temp ^= v_bit
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    return list(best_S)
