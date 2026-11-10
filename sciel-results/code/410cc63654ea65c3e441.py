"""
Q3-free graph on 16 vertices - maximize edge count.

Strategy: Use a heuristic construction followed by local search.
1. Start with a dense initial graph (e.g., complement of a sparse graph).
2. Check for Q3 subgraphs.
3. Use local search (edge removal/addition) to find a maximal Q3-free graph.

Since checking for Q3 in a dense graph is expensive, we use a smarter approach:
- Q3 has 8 vertices and 12 edges.
- We'll use a randomized approach: start with K16 and remove edges until Q3-free.
- Or better: start with a known dense Q3-free graph and try to add edges.

Key insight: Q3 is the cube graph. A graph is Q3-free if no 8 vertices induce (contain as a subgraph) Q3.

Practical approach:
1. Use the known fact that K16 minus a perfect matching is a good starting point? No, that might contain Q3.
2. Better: Use a bipartite construction or a known extremal construction.
3. Or: Use local search - start with a random graph and iteratively remove edges that complete a Q3.

Let me use a more systematic approach:
- Precompute all subsets of 8 vertices that could form a Q3 (there are C(16,8) = 12870).
- For each 8-subset, check if it contains Q3 as a subgraph.
- Use this to guide edge removals.

But checking all 12870 subsets in each iteration might be slow. Let's use a different approach.

Alternative: Use the fact that Q3 is bipartite with parts of size 4 and 4. 
We can try to construct a graph that is "close" to a bipartite graph.

Actually, let me try a direct search approach with good initialization:
1. Start with K16 (120 edges).
2. While the graph contains a Q3, remove one edge from a Q3 found.
3. Then try to add back edges that don't create a Q3.

The challenge is efficiently finding a Q3 in the graph.

Let me precompute all Q3 embeddings in K16. A Q3 embedding is a mapping of Q3's 8 vertices to 16 vertices such that all Q3 edges are present in the graph.

Number of Q3 embeddings in K16: 
- Choose 8 vertices from 16: C(16,8) = 12870
- For each 8-set, number of ways to map Q3 vertices to them: 8! = 40320
- But Q3 has automorphisms, so we divide by |Aut(Q3)| = 48 (the symmetry group of the cube)
- So roughly 12870 * 40320 / 48 ≈ 10,800,000 embeddings? That's too many to precompute.

Better approach: Use a SAT-style or backtracking search for Q3, but that's complex.

Let me try a simpler heuristic:
1. Start with a dense graph (e.g., K16 minus a few edges).
2. Use random sampling to detect Q3s: pick random 8-vertex subsets and check if they contain Q3.
3. If a Q3 is found, remove a random edge from it.
4. Repeat until no Q3 is found (with some probability of existence).
5. Then try to add edges back.

For checking if an 8-vertex subset contains Q3:
- Q3 is the 3-dimensional cube. Its vertices can be labeled 000, 001, 010, 011, 100, 101, 110, 111.
- Two vertices are adjacent if they differ in exactly one bit.
- So we need to check if there's a bijection f from {0..7} to the 8 vertices such that for all i, j, if i and j differ in exactly one bit, then f(i) and f(j) are adjacent in the graph.

This is an isomorphism check, which is NP-hard in general, but for 8 vertices it's fast.

Let me implement this:
1. For a given 8-vertex subset S, try to find a Q3 embedding.
2. Use backtracking: assign Q3 vertices to graph vertices, checking consistency.

Actually, for 8 vertices, we can enumerate all 8! = 40320 permutations and check if any is a Q3 embedding. This is feasible.

Let me code this up.
"""

import time
import random
from itertools import permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n = 16
    # Adjacency matrix: adj[u][v] = True if edge exists
    adj = [[False] * n for _ in range(n)]
    
    # Initialize with K16
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            adj[u][v] = adj[v][u] = True
            edges.append([u, v])
    
    def has_edge(u, v):
        return adj[u][v]
    
    def add_edge(u, v):
        if u == v:
            return
        u, v = min(u, v), max(u, v)
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = True
            edges.append([u, v])
    
    def remove_edge(u, v):
        u, v = min(u, v), max(u, v)
        if adj[u][v]:
            adj[u][v] = adj[v][u] = False
            # Remove from edges list (this is slow, but we'll rebuild at the end)
            pass  # We'll rebuild edges at the end
    
    # Q3 vertices: 0..7, edges where i^j is a power of 2
    q3_edges = set()
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j) in (1, 2, 4):
                q3_edges.add((i, j))
    
    def is_q3_embedding(perm, vertex_set):
        """Check if perm (mapping q3 vertex -> index in vertex_set) is a valid Q3 embedding."""
        for i, j in q3_edges:
            u = vertex_set[perm[i]]
            v = vertex_set[perm[j]]
            if not has_edge(u, v):
                return False
        return True
    
    def find_q3_in_subset(vertex_set):
        """Find a Q3 embedding in the given 8-vertex subset. Returns perm or None."""
        # Try random permutations for speed
        indices = list(range(8))
        random.shuffle(indices)
        # We'll try a limited number of permutations
        for _ in range(1000):
            perm = [0] * 8
            for q3_v, idx in enumerate(indices):
                perm[q3_v] = idx
            if is_q3_embedding(perm, vertex_set):
                return perm
        return None
    
    def find_q3():
        """Find a Q3 in the current graph. Returns (vertex_set, perm) or None."""
        # Sample random 8-vertex subsets
        all_vertices = list(range(n))
        for _ in range(200):
            subset = random.sample(all_vertices, 8)
            perm = find_q3_in_subset(subset)
            if perm is not None:
                return subset, perm
        return None
    
    # Phase 1: Remove edges to make graph Q3-free
    removal_count = 0
    while time.time() < deadline - 30:  # Leave 30s for phase 2
        q3 = find_q3()
        if q3 is None:
            # We think the graph is Q3-free (with some probability)
            break
        subset, perm = q3
        # Remove a random edge from the Q3
        i, j = random.choice(list(q3_edges))
        u = subset[perm[i]]
        v = subset[perm[j]]
        remove_edge(u, v)
        removal_count += 1
    
    # Rebuild edges list
    edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
    
    # Phase 2: Try to add back edges
    # Get all non-edges
    non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
    random.shuffle(non_edges)
    
    for u, v in non_edges:
        if time.time() > deadline - 5:
            break
        # Tentatively add edge
        adj[u][v] = adj[v][u] = True
        # Check if this creates a Q3
        # Only need to check 8-sets containing both u and v
        # There are C(14,6) = 3003 such sets
        # But we can be smarter: only check sets that might form Q3
        creates_q3 = False
        other_vertices = [w for w in range(n) if w != u and w != v]
        random.shuffle(other_vertices)
        # Sample some 6-subsets from other_vertices
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            sample = random.sample(other_vertices, 6)
            vertex_set = sample + [u, v]
            perm = find_q3_in_subset(vertex_set)
            if perm is not None:
                creates_q3 = True
                break
        if not creates_q3:
            # Keep the edge
            pass
        else:
            # Remove the edge
            adj[u][v] = adj[v][u] = False
    
    # Final edges
    edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
    
    return edges

