import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    n = 256
    # Build conflict graph using bitmasks for speed
    # conflicts[i] is a bitmask where bit j is set if i conflicts with j
    conflicts = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Initial solution: All odd numbers (no conflicts among odds)
    best_S = set(range(1, n + 1, 2))
    best_size = len(best_S)
    
    # Tomita-style branch and bound for Maximum Independent Set
    def color_bound(candidates):
        """Compute greedy coloring upper bound on candidates bitmask."""
        color_classes = []
        remaining = candidates
        while remaining:
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            color_classes.append(v_bit)
            remaining &= ~(conflicts[v] | v_bit)
        return len(color_classes)
    
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
        
        ub = current_size + color_bound(candidates)
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
    
    # Local search polish: try to add vertices that don't conflict with current set
    if time.time() < deadline - 3:
        S = set(best_S)
        all_nums = list(range(1, n + 1))
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in all_nums:
                if x not in S:
                    # Check if x conflicts with any element in S
                    x_conflicts = conflicts[x]
                    conflict_with_S = False
                    for s in S:
                        if (x_conflicts >> s) & 1:
                            conflict_with_S = True
                            break
                    if not conflict_with_S:
                        S.add(x)
                        improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
                improved = True
    
    return list(best_S)
