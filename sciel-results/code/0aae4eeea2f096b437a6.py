"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0  # 85s out of 90s budget
    
    N = 180
    
    # 1. Precompute the conflict graph
    # A conflict exists between x and y if there exists z such that {x, y, z} is a reciprocal triple.
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # For a fixed a, pairs (b,c) are determined by divisors of a^2.
    # We need to build an adjacency list for the graph where nodes are 1..N.
    # If {a,b,c} is a triple, then a conflicts with b, a conflicts with c, and b conflicts with c.
    
    adjacency = [set() for _ in range(N + 1)]
    
    # Find all triples (a, b, c) with distinct elements in 1..N
    # Iterate a from 1 to N
    for a in range(1, N + 1):
        a_sq = a * a
        # Divisors d of a_sq
        # b = a + d, c = a + a_sq/d
        # We need b, c <= N and b != c
        # Also b, c must be distinct from a (guaranteed since d>=1)
        # And b != c => d != a_sq/d => d^2 != a_sq => d != a
        for d in range(1, a_sq + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    # Add edges
                    # a-b, a-c, b-c
                    adjacency[a].add(b)
                    adjacency[a].add(c)
                    adjacency[b].add(a)
                    adjacency[b].add(c)
                    adjacency[c].add(a)
                    adjacency[c].add(b)
    
    # 2. Tomita's Algorithm for Maximum Independent Set
    # We want to find the largest subset of vertices with no edges between them.
    
    # Greedy coloring for upper bound
    # We color the graph such that adjacent vertices have different colors.
    # The maximum size of an independent set in a subgraph is bounded by the number of colors used.
    # We use a specific coloring order to make the bound tighter.
    
    def color_subgraph(vertices):
        """
        Greedy coloring of the subgraph induced by 'vertices'.
        Returns a list of color classes (lists of vertices) and the total number of colors.
        The vertices are ordered by descending degree in the subgraph to improve bound tightness.
        """
        if not vertices:
            return [], 0
        
        # Order vertices by descending degree within the subgraph
        # This helps greedy coloring use fewer colors, providing a tighter upper bound
        subgraph = {v: adjacency[v] & set(vertices) for v in vertices}
        ordered = sorted(vertices, key=lambda v: -len(subgraph[v]))
        
        color_classes = []
        assigned = set()
        
        for v in ordered:
            if v in assigned:
                continue
            
            # Find the first color class that doesn't contain a neighbor of v
            placed = False
            for i, cls in enumerate(color_classes):
                # Check if v has any neighbor in this class
                # If no neighbor in class, v can be added
                if not (subgraph[v] & set(cls)):
                    cls.append(v)
                    assigned.add(v)
                    placed = True
                    break
            
            if not placed:
                color_classes.append([v])
                assigned.add(v)
        
        return color_classes, len(color_classes)

    # Branch and Bound
    # best_set: global best independent set found
    # current_set: current path in the search tree
    # candidates: set of vertices that can be added to current_set (not adjacent to any in current_set)
    
    best_set = set()
    
    # Initial lower bound: Use the known good seed (upper half)
    # Verify upper half is independent
    seed_set = set(range(91, 181))
    # Check validity of seed
    valid_seed = True
    for v in seed_set:
        if adjacency[v] & seed_set:
            valid_seed = False
            break
    if valid_seed:
        best_set = seed_set
    else:
        # Fallback: Greedy construction
        current = set()
        for v in range(N, 0, -1):
            if not (adjacency[v] & current):
                current.add(v)
        best_set = current

    # For Tomita's algorithm, we typically process vertices in a specific order.
    # Let's use a global ordering or dynamic ordering.
    # Standard Tomita:
    # 1. Order vertices by degree.
    # 2. Branch on vertices.
    
    # To make it efficient, we'll implement the recursive step carefully.
    # We need to handle the "candidates" set efficiently.
    
    # Let's use a bitset or just sets. N=180 is small enough for sets.
    
    def expand(current_set, candidates):
        """
        Recursively expand the independent set.
        current_set: set of vertices already in the independent set.
        candidates: set of vertices that can be added to current_set.
        """
        nonlocal best_set
        
        if not candidates:
            return
        
        # Check time
        if time.time() > deadline:
            return
            
        # Upper bound: size of current_set + max possible from candidates
        # We use the coloring bound on the subgraph induced by candidates
        color_classes, num_colors = color_subgraph(candidates)
        
        # If even taking the max color class doesn't beat best, prune
        if len(current_set) + num_colors <= len(best_set):
            return
        
        # Select a vertex to branch on.
        # Tomita suggests picking the vertex with the highest color number (last color class).
        # This vertex is in the last color class.
        if not color_classes:
            return
            
        last_class = color_classes[-1]
        # Pick a vertex from the last class.
        # Usually the first one or the one with highest degree.
        v = last_class[0]
        
        # Branch 1: Include v
        new_current = current_set | {v}
        # New candidates are those in candidates that are not adjacent to v
        new_candidates = candidates - adjacency[v] - {v}
        
        expand(new_current, new_candidates)
        
        # Branch 2: Exclude v
        # Remove v from candidates
        # Note: In standard MIS, we don't just remove v, we branch.
        # However, Tomita's algorithm structure is:
        # P1 = P \ ({v} U N(v))  (Include v)
        # P2 = P \ {v}           (Exclude v)
        
        candidates.remove(v)
        expand(current_set, candidates)

    # Initial call
    # We start with an empty set and all vertices as candidates?
    # Or we can start with the seed?
    # Starting with empty allows finding potentially larger sets, but seed helps pruning.
    # Let's start with empty but use the seed for the initial best_set bound.
    
    all_vertices = set(range(1, N + 1))
    # To improve performance, we can order the initial candidates.
    # But expand() handles coloring dynamically.
    
    # One optimization: If the graph is sparse, we might want to start with high-degree vertices excluded?
    # No, MIS wants to include vertices.
    
    # Let's run the search.
    # Since N=180, recursion depth could be 180. Python default limit is 1000, so it's safe.
    # But we need to be careful with the time limit.
    
    # To ensure we don't miss the seed if the search is slow, we already set best_set to seed.
    # The search will only update best_set if it finds something strictly larger.
    
    # We should probably run the search with a timeout.
    # The 'deadline' check inside expand() handles this.
    
    # Let's add a safeguard: if the search takes too long, we return the best found.
    
    try:
        expand(set(), all_vertices)
    except RecursionError:
        pass # Should not happen with N=180
    
    # If the search didn't find anything better, best_set is still the seed (or better if found).
    
    # Final check: ensure best_set is valid (should be by construction)
    # And non-empty
    if not best_set:
        # Fallback
        best_set = {N}
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
