import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Nodes are 2..256
    n = 256
    vertices = list(range(2, n + 1))
    num_vertices = len(vertices)
    
    # Weights: 1/x as Fraction for exact arithmetic
    weights = {v: Fraction(1, v) for v in vertices}
    
    # Build compatibility graph (complement of conflict graph)
    # Edge exists if lcm(i, j) > 256
    neighbors = {v: set() for v in vertices}
    
    # Precompute adjacency
    for i_idx in range(num_vertices):
        i = vertices[i_idx]
        for j_idx in range(i_idx + 1, num_vertices):
            j = vertices[j_idx]
            if lcm(i, j) > 256:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    best_sum = Fraction(0)
    best_set = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline
    
    # Weighted coloring bound: color P greedily, bound = sum of max weight per color class
    def color_bound(P):
        if not P:
            return Fraction(0)
        
        # Order by degree in P descending for better coloring
        P_list = sorted(P, key=lambda v: -len(neighbors[v] & P))
        
        color_of = {}
        color_weights = []  # color_weights[c] = max weight in color c
        
        for v in P_list:
            used = set()
            for u in (neighbors[v] & P):
                if u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c >= len(color_weights):
                color_weights.append(Fraction(0))
            if weights[v] > color_weights[c]:
                color_weights[c] = weights[v]
        
        return sum(color_weights)
    
    # Bron-Kerbosch with pivot and weighted coloring bound
    def bron_kerbosch(R, P, X):
        nonlocal best_sum, best_set
        
        if time_check():
            return
            
        if not P and not X:
            # Maximal clique found
            s = sum(weights[v] for v in R)
            if s > best_sum:
                best_sum = s
                best_set = list(R)
            return
        
        # Pruning: Upper bound using coloring
        current_sum = sum(weights[v] for v in R)
        if current_sum + color_bound(P) <= best_sum:
            return
            
        # Pivot selection
        union_px = P | X
        if union_px:
            u = max(union_px, key=lambda x: len(neighbors[x] & P))
            candidates = P - neighbors[u]
        else:
            candidates = P
            
        for v in list(candidates):
            if time_check():
                return
            R.append(v)
            new_P = P & neighbors[v]
            new_X = X & neighbors[v]
            
            bron_kerbosch(R, new_P, new_X)
            
            R.pop()
            P.remove(v)
            X.add(v)
            
    # Initial call
    initial_P = set(vertices)
    initial_X = set()
    bron_kerbosch([], initial_P, initial_X)
    
    if not best_set:
        # Fallback
        selected = []
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected
        
    return best_set
