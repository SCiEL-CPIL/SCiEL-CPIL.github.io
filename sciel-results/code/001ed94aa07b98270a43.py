"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 85  # seconds, leave 5s margin
    
    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT
    
    N = 512
    
    # Precompute compatibility: compat[x] is a set of y such that lcm(x,y) <= N
    compat = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x, N + 1):
            g = math.gcd(x, y)
            if x // g * y <= N:
                compat[x].add(y)
                compat[y].add(x)
    
    # Tomita-style Maximum Clique with greedy coloring
    best_clique = []
    best_size = 0
    nodes_explored = 0
    deadline_hit = [False]
    
    def color_sort(P):
        """Greedy coloring of candidate set P.
        Returns (ordered_P, color_bounds) where ordered_P[i] has color color_bounds[i].
        color_bounds is non-decreasing, and color_bounds[i] is an upper bound on
        clique size in {ordered_P[0], ..., ordered_P[i]}.
        """
        if not P:
            return [], []
        
        color_classes = []
        uncolored = list(P)
        
        while uncolored:
            color_classes.append([])
            for i in range(len(uncolored) - 1, -1, -1):
                v = uncolored[i]
                ok = True
                for u in color_classes[-1]:
                    if u not in compat[v]:
                        ok = False
                        break
                if ok:
                    color_classes[-1].append(v)
                    uncolored.pop(i)
        
        ordered = []
        color_bounds = []
        for ci, cls in enumerate(color_classes):
            for v in cls:
                ordered.append(v)
                color_bounds.append(ci + 1)
        
        return ordered, color_bounds
    
    def expand(R, P):
        nonlocal best_clique, best_size, nodes_explored
        
        if deadline_hit[0]:
            return
        
        nodes_explored += 1
        if nodes_explored % 3000 == 0 and check_deadline():
            deadline_hit[0] = True
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        ordered_P, color_bounds = color_sort(P)
        
        i = len(ordered_P) - 1
        while i >= 0:
            if deadline_hit[0]:
                return
            
            v = ordered_P[i]
            
            if len(R) + color_bounds[i] <= best_size:
                i -= 1
                continue
            
            R.append(v)
            new_P = [u for u in ordered_P[:i] if u in compat[v]]
            expand(R, new_P)
            R.pop()
            i -= 1
    
    # Initial greedy lower bound: repeatedly pick vertex with max degree in remaining
    P_init = list(range(1, N + 1))
    R_init = []
    while P_init:
        best_v = max(P_init, key=lambda v: len(compat[v] & set(P_init)))
        R_init.append(best_v)
        P_init = [u for u in P_init if u != best_v and u in compat[best_v]]
    
    best_size = len(R_init)
    best_clique = list(R_init)
    
    # Branch and bound
    expand([], list(range(1, N + 1)))
    
    if not best_clique:
        return list(range(1, 23))
    
    final_list = sorted(best_clique)
    
    # Verify
    valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if final_list[j] not in compat[final_list[i]]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return list(range(1, 23))
    
    return final_list

# EVOLVE-BLOCK-END
