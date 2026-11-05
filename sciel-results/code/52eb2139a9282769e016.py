import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    N = 512
    # Precompute compatibility
    # adj[i] is a set of nodes compatible with i
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Tomita's algorithm for Maximum Clique
    # P: candidate set, X: excluded set, C: current clique
    best_clique = []
    
    def expand(P, X, C):
        nonlocal best_clique
        if time.time() - start_time > timeout:
            return
        
        if len(P) == 0:
            if len(C) > len(best_clique):
                best_clique = C.copy()
            return
        
        # Greedy coloring to get an upper bound
        # We need to order P such that colors are non-decreasing
        # and the last color is an upper bound on the size of the clique in P
        
        # Simple coloring heuristic:
        # Order P by degree in the subgraph induced by P? 
        # Or just arbitrary order and color greedily.
        # For Tomita, we usually sort P by degree in the whole graph or subgraph.
        
        # Let's use a simple greedy coloring on the induced subgraph of P
        # We want to find a clique.
        # Standard Tomita:
        # 1. Color P greedily to get an ordering and colors.
        # 2. Process nodes in reverse color order.
        
        # To make it fast, we'll just use a simple heuristic.
        # Given N=512, even O(N^2) per step is fine if depth is small.
        
        # Let's implement a simplified version:
        # Pick a node from P, add to C, recurse on P intersected with adj[node].
        # To prune, we need an upper bound.
        # A simple upper bound is len(P).
        # If len(C) + len(P) <= len(best_clique), return.
        
        if len(C) + len(P) <= len(best_clique):
            return
        
        # Heuristic: pick the node with the highest degree in P
        # This is not strictly Tomita but a good heuristic.
        # To be safer and faster, let's use the standard Tomita coloring.
        
        # We'll implement the coloring step properly.
        # Color the graph induced by P.
        # We want an ordering v1, ..., vk such that color(vi) <= color(vj) for i < j
        # and color(vk) is an upper bound.
        
        # Simple greedy coloring:
        # Unassigned = P
        # Colors = {}
        # color_idx = 0
        # While Unassigned:
        #    color_idx += 1
        #    Q = Unassigned
        #    While Q:
        #       pick u from Q (e.g., first one)
        #       Colors[u] = color_idx
        #       Unassigned.remove(u)
        #       Q = Q - adj[u] (within P)
        
        # This is O(N^2).
        
        unassigned = list(P)
        colors = {}
        color_idx = 0
        while unassigned:
            color_idx += 1
            Q = set(unassigned)
            while Q:
                # Pick a node, maybe one with min degree in Q? Or just pop.
                # Popping arbitrary is fine for coloring validity.
                u = next(iter(Q))
                colors[u] = color_idx
                unassigned.remove(u)
                Q.difference_update(adj[u])
                # Q should be intersected with P, but adj[u] already only contains valid neighbors
                # and Q is subset of P.
                
        # Now we have colors.
        # Sort nodes by color.
        # Process in reverse order of colors.
        # Nodes with higher color are processed first?
        # Tomita processes in order of increasing color, but checks upper bound.
        # Actually, we process nodes in the order they were colored?
        # The standard approach:
        # Order nodes by color.
        # For each node u in reverse order of the coloring list:
        #    If len(C) + colors[u] <= len(best_clique): continue
        #    Expand with P \ {u} and C + {u}
        #    Remove u from P
        #    Add u to X
        
        # Let's create the ordered list
        # We need to handle the sets P and X carefully.
        
        # To avoid deep recursion overhead, we can use a stack, but recursion is fine for N=512.
        
        # We need to pass P and X as sets.
        # The coloring was done on the current P.
        
        # Let's refine the loop:
        # We have `unassigned` now empty.
        # We have `colors` dict.
        
        # Sort nodes by color
        nodes_in_P = list(P)
        nodes_in_P.sort(key=lambda x: colors[x])
        
        # Process in reverse?
        # The upper bound for a node u is colors[u].
        # If we process nodes with small colors first, the bound is tight?
        # No, the bound is that the max clique in the remaining set is <= max color.
        # So if we remove a node with color c, the remaining nodes have colors <= c?
        # No, the coloring is for the whole P.
        # If we remove u, the remaining P' is a subset of P.
        # The max clique in P' is <= max color in P'.
        
        # Standard Tomita:
        # 1. Color P.
        # 2. Order P by color.
        # 3. Iterate through the ordered list.
        #    Let u be the next node.
        #    If len(C) + color[u] <= best_size: return (since subsequent nodes have >= color? No, <=? )
        #    Actually, if we order by increasing color, the last node has the max color.
        #    If we process in reverse (highest color first), we can prune early?
        #    Let's stick to: 
        #    If len(C) + color[u] <= best_size: skip u? 
        #    No, the bound applies to the set of nodes with color >= color[u]?
        #    
        #    Let's use the property: 
        #    The maximum clique in P is bounded by the number of colors.
        #    When we pick u, we recurse on P_u = P ∩ N(u).
        #    The bound for the branch is len(C) + 1 + (max clique in P_u).
        #    We don't have a tight bound for P_u easily without re-coloring.
        #    
        #    However, a simple pruning:
        #    If len(C) + len(P) <= len(best_clique): return.
        #    This is what I had before.
        #    
        #    To be more efficient, we can use the color bound.
        #    If we process nodes in decreasing order of color:
        #    The remaining nodes (after removing u) have colors <= color[u]? 
        #    No, the coloring is static for this P.
        #    
        #    Let's just use the simple len(P) bound first. It's often sufficient.
        #    If it's too slow, I'll add coloring.
        
        pass

    # Let's try a simpler, robust heuristic first: Greedy + Local Search.
    # It's less likely to hit exponential blowup.
    
    # Greedy:
    # Start with empty set.
    # Add nodes one by one, preferring those with high degree in the remaining compatible set.
    
    # Actually, let's try to find the max clique using a library or a very optimized implementation.
    # Since I can't import networkx, I'll write a compact Bron-Kerbosch with pivot.
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() - start_time > timeout:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
        
        # Pivot selection
        # Choose u in P ∪ X that maximizes |P ∩ N(u)|
        # If P ∪ X is empty, u = None
        U = P.union(X)
        if not U:
            u = None
        else:
            # Heuristic: pick node with max degree in P
            max_deg = -1
            u = None
            for v in U:
                deg = len(P.intersection(adj[v]))
                if deg > max_deg:
                    max_deg = deg
                    u = v
        
        if u is None:
            candidates = P
        else:
            candidates = P - adj[u]
            
        for v in list(candidates):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)

    # Initialize
    P = set(range(1, N + 1))
    X = set()
    R = set()
    
    # To improve speed, we can order P initially by degree.
    # But Bron-Kerbosch with pivot is usually fast.
    
    # Let's run it.
    bron_kerbosch(R, P, X)
    
    if best_clique:
        return list(best_clique)
    else:
        # Fallback
        return [1]
