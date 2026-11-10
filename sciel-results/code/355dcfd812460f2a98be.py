"""Rewritten program: construct largest pairwise non-coprime subset of [1,255] containing 255."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    
    Since 255 ∈ S, for any x ∈ S, gcd(x, 255) > 1.
    Let C = {x ∈ [1, 255] | gcd(x, 255) > 1}.
    Then S ⊆ C.
    The condition "pairwise gcd > 1" means S is a clique in the graph G_C where 
    vertices are C and edges connect u,v if gcd(u,v) > 1.
    Since 255 is connected to all vertices in C (by definition of C), 
    finding the max S containing 255 is equivalent to finding the max clique 
    in the subgraph induced by C \ {255}, and adding 255 to it.
    """
    
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    # 1. Generate candidates C
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # 2. We need max clique in G[C \ {255}]
    # Let V = C \ {255}
    V = [x for x in C if x != 255]
    n = len(V)
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(V)}
    
    # Build adjacency list
    # adj[i] is a set of indices j such that gcd(V[i], V[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = V[i]
        for j in range(i + 1, n):
            vj = V[j]
            if gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Maximum Clique Algorithm (Branch and Bound with Tomita-style coloring)
    # We use a color-based upper bound for pruning, which is much stronger than
    # the simple |candidates| bound.
    
    best_clique = []
    best_size = 0
    
    # Color the graph using greedy coloring (Tomita's MCQ algorithm)
    # For a set of candidates, we assign colors such that adjacent vertices
    # have different colors. The color number of a vertex gives an upper bound
    # on the clique size that can be extended through that vertex.
    
    def color_sort(candidates):
        """
        Greedily color the candidates and return them sorted by color (descending).
        Returns a list of (vertex, color) pairs sorted by color descending.
        """
        if not candidates:
            return []
        
        # Work with a list of vertices
        verts = list(candidates)
        
        # Greedy coloring: assign smallest available color
        color_of = {}
        # We'll iterate and assign colors
        # For efficiency, we process in some order
        # Let's sort by degree descending first to help coloring
        verts.sort(key=lambda v: len(adj[v] & candidates), reverse=True)
        
        colors_used = set()
        for v in verts:
            # Find neighbors of v that are already colored
            neighbor_colors = set()
            for u in adj[v]:
                if u in color_of:
                    neighbor_colors.add(color_of[u])
            # Assign smallest color not in neighbor_colors
            c = 1
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            colors_used.add(c)
        
        # Sort by color descending
        sorted_verts = sorted(verts, key=lambda v: color_of[v], reverse=True)
        return [(v, color_of[v]) for v in sorted_verts]
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Time check
        if time.time() > deadline:
            return
        
        # Color-based upper bound
        colored = color_sort(candidates)
        if not colored:
            return
        
        # The maximum color is an upper bound on clique size extension
        max_color = colored[-1][1]  # colors are 1-indexed, max is last
        if len(clique) + max_color <= best_size:
            return
        
        # Process vertices in order of increasing color (so we try high-color ones first
        # but actually Tomita processes in order and uses the color as bound for remaining)
        # Let's process in the order given by color_sort (descending color)
        # For each vertex, the upper bound is its color (since colors are assigned
        # such that the color of v is >= any clique in the remaining subgraph that
        # doesn't include v... actually the standard approach is:
        # Process vertices in order, and for vertex at position i, the bound is
        # the color of the vertex at position i (or the max color among remaining).
        
        # Standard Tomita: process in order, and the bound for vertex v_i is color(v_i)
        # because vertices after v_i have color <= color(v_i) (since sorted descending).
        
        for idx, (v, c) in enumerate(colored):
            if time.time() > deadline:
                return
            
            # Upper bound: len(clique) + c
            if len(clique) + c <= best_size:
                break  # All remaining vertices have color <= c, so no improvement possible
            
            new_clique = clique + [v]
            new_candidates = candidates & adj[v]
            expand(new_clique, new_candidates)
            candidates.remove(v)
            
            # Update the color bound: after removing v, the max color among remaining
            # is at most c (since we're processing in descending color order)
            # Actually, the next vertex has color <= c, so the bound naturally decreases.
    
    # Start with empty clique and all V as candidates
    expand([], set(range(n)))
    
    # 4. Construct final set
    # best_clique contains indices into V
    result_indices = best_clique
    result_values = [V[i] for i in result_indices]
    result_values.append(255)
    
    # 5. Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
            
    return result_values
# EVOLVE-BLOCK-END
