"""
Q3-free graph construction on 16 vertices.

Strategy:
- Use a bipartite base graph K_{4,12} (48 edges) which is Q3-free (needs 4+4 for a cube, but the bipartite structure of Q3 requires specific edge patterns; K_{4,12} does contain Q3 actually - wait, let me reconsider).

Actually, K_{4,4} contains Q3 as a subgraph (just take all 4+4 vertices and all edges between them - that's K_{4,4} which has 16 edges, but Q3 has 12 edges. K_{4,4} contains Q3 as a subgraph because we can select 12 of the 16 edges to form a cube).

So K_{4,12} does contain Q3. Let me think more carefully.

A Q3 (3-cube) has 8 vertices with 4 in each bipartition class, and 12 edges. For a bipartite graph with parts A and B to contain Q3, we need |A| >= 4 and |B| >= 4, AND the subgraph must have the specific cube structure.

K_{4,4} contains Q3: take all 4 vertices from each side. The 16 edges of K_{4,4} include many 12-edge subsets that form Q3.

So we need a different approach. Let's use:
- Start with a random graph and use local search
- Or use a known construction

Let me use a greedy approach with a proper Q3 checker.

For Q3 detection: A subgraph isomorphic to Q3 exists if there are 8 vertices where the induced subgraph contains all 12 edges of some cube labeling.

Efficient check: For each set of 8 vertices, check if it contains Q3.
- C(16,8) = 12870 subsets
- For each, check if the 8-vertex subgraph contains Q3

For an 8-vertex subgraph to contain Q3, it needs at least 12 edges, and those edges must include a set of 12 edges forming a cube.

The number of distinct Q3 edge sets on 8 labeled vertices is 840 (as computed: 8!/48).

So for each 8-vertex subset with >= 12 edges, check all 840 Q3 patterns.

Let me implement this with bitmasks for efficiency.
"""

import itertools
import random

def construct():
    N = 16
    
    # Precompute all Q3 edge patterns on vertices 0-7
    # Q3: vertices 0-7, edges between vertices differing in exactly one bit
    q3_base_edges = set()
    for v in range(8):
        for bit in range(3):
            w = v ^ (1 << bit)
            if v < w:
                q3_base_edges.add((v, w))
    
    # Generate all distinct Q3 edge patterns (as frozensets of (i,j) with i<j)
    q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        edges = set()
        for (u, v) in q3_base_edges:
            a, b = perm[u], perm[v]
            if a > b:
                a, b = b, a
            edges.add((a, b))
        q3_patterns.add(frozenset(edges))
    
    # Convert to list for faster iteration
    q3_pattern_list = list(q3_patterns)
    
    # Precompute all 8-vertex subsets
    all_8_subsets = list(itertools.combinations(range(N), 8))
    
    def has_q3(adj):
        """Check if graph with adjacency sets contains Q3."""
        # adj is a list of sets
        for subset in all_8_subsets:
            # Build edge set for this subset using local indices 0-7
            sub_edges = set()
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = subset[i], subset[j]
                    if v in adj[u]:
                        sub_edges.add((i, j))
            
            if len(sub_edges) < 12:
                continue
            
            # Check if any Q3 pattern is a subset
            for pattern in q3_pattern_list:
                if pattern.issubset(sub_edges):
                    return True
        
        return False
    
    # Start with a moderately dense random graph
    random.seed(42)
    adj = [set() for _ in range(N)]
    edges = set()
    
    # Start with K_{3,13} as base (39 edges) - guaranteed Q3-free
    for a in range(3):
        for b in range(3, N):
            adj[a].add(b)
            adj[b].add(a)
            edges.add((a, b))
    
    # Try to add more edges
    missing = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in edges]
    random.shuffle(missing)
    
    # Greedily add edges that don't create Q3
    for (u, v) in missing:
        adj[u].add(v)
        adj[v].add(u)
        if has_q3(adj):
            # Revert
            adj[u].discard(v)
            adj[v].discard(u)
        else:
            edges.add((u, v))
    
    return [[u, v] for u, v in sorted(edges)]
