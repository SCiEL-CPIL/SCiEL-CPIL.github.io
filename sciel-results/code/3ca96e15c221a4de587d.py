"""Fewest four-cycles in graphs with 31 vertices and 91 edges.

We use a construction based on the polarity graph of the projective plane of order q=5,
which is known to have ex(31; C4) = 90 edges with a controlled number of C4s.
We then add one edge and try to minimize the total C4 count via local search.

The polarity graph of PG(2,5) has 31 vertices and 31*5/2 + 31/2 = ... actually:
For a polarity graph of PG(2,q) with q odd, the number of edges is q(q+1)/2 * q? 
Let me recall: The polarity graph of PG(2,q) has n = q^2+q+1 vertices.
Each vertex is adjacent to q+1 vertices (the absolute points of the line at infinity? no).

Actually, for a symplectic polarity of PG(2,q) with q odd:
- Number of absolute points: q+1
- Each vertex has degree: q+1 (if absolute) or q (if non-absolute)? 

Let me use a simpler approach: use the known construction for C4-free or near-C4-free graphs.

For n=31, ex(31;C4) = 90. The Reiman bound gives ex(n;C4) <= n/4(1+sqrt(4n-3)).
For n=31: 31/4 * (1+sqrt(121)) = 31/4 * 12 = 93. So 90 is below Reiman.

A well-known construction: take the incidence graph of points and lines in PG(2,q) and 
use a polarity to create a graph. Alternatively, use the graph from the finite field.

Let me use the following construction:
Vertices: (x, y) for x in GF(5)^2 plus one point at infinity, plus... no, 5^2=25, need 31.

PG(2,5) has 31 points. A polarity (symplectic) gives a graph where each point is 
adjacent to the points on its polar line. For a symplectic polarity of PG(2,q), q odd:
- q+1 = 6 absolute points
- 25 non-absolute points
- Absolute points have degree q+1 = 6
- Non-absolute points have degree q = 5

Total edges: (6*6 + 25*5)/2 = (36+125)/2 = 161/2 = 80.5? That's not integer.

Let me reconsider. For a symplectic polarity of PG(2,q):
- The number of absolute points is q+1
- Each absolute point is incident to its polar line which contains q+1 points, all of which 
  are adjacent to it. But some of those are absolute too.

Actually, let me just use a known explicit construction. The graph with 31 vertices and 
90 edges that is C4-free (or has few C4s) can be constructed using the following:

Use vertices 0..30. Define adjacency using finite field GF(5).
Let vertices be (a, b) for a,b in {0,1,2,3,4} (25 vertices) plus 6 more.

Alternatively, use the following simpler approach: start with a random graph with 91 edges 
and do local search to minimize C4 count. Given the time limit, we can afford a modest 
local search.

Let me implement:
1. Generate a random graph with 91 edges.
2. Compute C4 count efficiently.
3. Local search: try swapping edges to reduce C4 count.

For C4 count: number of 4-cycles = (1/8) * sum over pairs of vertices of (common neighbors)^2 - ... 
Actually, the number of 4-cycles (as subgraphs, not closed walks) is:
For each pair of vertices {u,v}, let d(u,v) = number of common neighbors.
Then the number of 4-cycles containing edge {u,v} is C(d(u,v), 2).
But a 4-cycle has 4 edges, so total C4s = (1/4) * sum over all edges {u,v} of C(d(u,v), 2).

Wait, that's not right either. Let me think again.

A 4-cycle is a subgraph isomorphic to C4. The number of C4 subgraphs can be computed as:
For each unordered pair of vertices {u, v}, let c(u,v) = |N(u) ∩ N(v)|.
The number of 4-cycles is (1/8) * sum_{u < v} c(u,v) * (c(u,v) - 1) ... no.

Each 4-cycle has two pairs of opposite vertices. For a 4-cycle (a,b,c,d), the pairs of 
opposite vertices are {a,c} and {b,d}. Each pair contributes 1 to the count of common 
neighbors being at least 2. 

The formula is: number of C4 = (1/8) * sum_{u} sum_{v} C(common(u,v), 2) where the sum 
is over all unordered pairs {u,v}, but this counts each 4-cycle exactly twice (once for 
each pair of opposite vertices). So:

C4_count = (1/8) * sum_{u < v} C(c(u,v), 2)

Wait, no. Let me be more careful. For a fixed 4-cycle with vertices a,b,c,d in order, 
the pairs of opposite vertices are {a,c} and {b,d}. For pair {a,c}, the common neighbors 
include b and d, so C(c(a,c), 2) >= 1. Similarly for {b,d}. 

The sum over all pairs {u,v} of C(c(u,v), 2) counts each 4-cycle exactly 2 times (once 
for each pair of opposite vertices). So:

C4_count = (1/2) * (1/2) * sum_{u < v} C(c(u,v), 2) = (1/4) * sum_{u < v} C(c(u,v), 2)

Hmm, let me just verify with a small example. C4 itself: 4 vertices, 4 edges. 
Pairs: {0,2} has common neighbors {1,3}, C(2,2)=1. {1,3} has common neighbors {0,2}, 
C(2,2)=1. Other pairs have 0 or 1 common neighbors. Sum = 2. C4_count should be 1. 
So C4_count = sum/2 = 1. Yes, so C4_count = (1/2) * sum_{u < v} C(c(u,v), 2).

Let me implement this.
"""

import time
import random
from itertools import combinations

def count_c4s(adj_matrix, n=31):
    """Count the number of C4 subgraphs using the common-neighbor formula."""
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            common = 0
            for w in range(n):
                if adj_matrix[u][w] and adj_matrix[v][w]:
                    common += 1
            if common >= 2:
                total += common * (common - 1) // 2
    return total // 2

def construct():
    deadline = time.time() + 140  # 140s to leave safety margin
    
    n = 31
    m = 91
    
    # Start with a structured graph: use a near-C4-free construction
    # Use the polarity graph of PG(2,5) as a starting point if possible,
    # otherwise start random and do local search.
    
    # Let's start with a random graph and do local search
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    
    # Initial random graph
    edge_set = set(random.sample(all_pairs, m))
    adj = [[0]*n for _ in range(n)]
    for u, v in edge_set:
        adj[u][v] = 1
        adj[v][u] = 1
    
    best_c4 = count_c4s(adj, n)
    best_edge_set = set(edge_set)
    
    # Local search: try to improve by swapping edges
    # An edge swap: remove edge e1, add edge e2, keeping m constant
    # Try to find swaps that reduce C4 count
    
    # Precompute common neighbor counts for efficiency
    def common_neighbor_counts(adj, n):
        cn = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        c += 1
                cn[u][v] = c
                cn[v][u] = c
        return cn
    
    cn = common_neighbor_counts(adj, n)
    
    # Local search iterations
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge to remove
        edges = list(edge_set)
        e1 = random.choice(edges)
        u1, v1 = e1
        
        # Try to find an edge to add that reduces C4
        # Candidate edges to add: non-edges
        best_delta = 0
        best_e2 = None
        
        # For efficiency, only try a subset of candidate edges
        # Edges that could reduce C4 are those connecting vertices with low common neighbor counts
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edge_set:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        # Limit candidates per iteration for speed
        for e2 in candidates[:200]:
            u2, v2 = e2
            
            # Estimate the change in C4 count
            # Removing e1=(u1,v1) affects C4s containing this edge
            # Adding e2=(u2,v2) affects C4s containing this edge
            
            # This is complex; let's just do a full recompute for simplicity
            # but that's too slow. Instead, use the common neighbor formula incrementally.
            
            # For now, do a quick heuristic: prefer adding edges between vertices 
            # with few common neighbors
            if cn[u2][v2] <= cn[u1][v1]:
                # Tentatively apply
                adj[u2][v2] = 1
                adj[v2][u2] = 1
                adj[u1][v1] = 0
                adj[v1][u1] = 0
                new_c4 = count_c4s(adj, n)
                adj[u2][v2] = 0
                adj[v2][u2] = 0
                adj[u1][v1] = 1
                adj[v1][u1] = 1
                
                if new_c4 < best_c4:
                    best_delta = new_c4 - best_c4
                    best_e2 = e2
                    best_c4_new = new_c4
        
        if best_e2 is not None:
            # Apply the best swap
            edge_set.discard(e1)
            edge_set.add(best_e2)
            u1, v1 = e1
            u2, v2 = best_e2
            adj[u1][v1] = 0
            adj[v1][u1] = 0
            adj[u2][v2] = 1
            adj[v2][u2] = 1
            best_c4 = best_c4_new
            best_edge_set = set(edge_set)
            cn = common_neighbor_counts(adj, n)
        else:
            # No improvement found in this iteration; try a few more random swaps
            # If stuck, try a random restart
            if iterations % 5000 == 0:
                # Random restart
                edge_set = set(random.sample(all_pairs, m))
                adj = [[0]*n for _ in range(n)]
                for u, v in edge_set:
                    adj[u][v] = 1
                    adj[v][u] = 1
                best_c4 = count_c4s(adj, n)
                cn = common_neighbor_counts(adj, n)
    
    # Return the best edge set found
    return [list(e) for e in sorted(best_edge_set)]
