"""
Strategy: Construct a C4-free base graph (90 edges) using a greedy algorithm with
incremental C4 tracking, then perform local search (edge swaps) to minimize C4s
in the 91-edge graph.

Key insight: The extremal number ex(31, C4) = 90, so there exists a C4-free graph
with 91-1=90 edges. We can construct such a graph greedily, then add one edge
and optimize via swaps.

For speed, we maintain:
- adj: adjacency matrix
- common[u,v]: number of common neighbors of u and v
- c4_count: total number of C4s = (1/4) * sum_{u,v} C(common[u,v], 2)

When adding edge (a,b):
- For each w in N(a) ∩ N(b) (common neighbors), nothing changes to common counts
  except that a and b become adjacent.
- Actually, adding edge (a,b) increases common(x,w) for pairs where x is neighbor of a 
  and w is neighbor of b, or vice versa.
  
Let me reconsider. The formula is:
C4_count = (1/4) * sum_{u,v} C(common(u,v), 2)

When we add edge (a,b):
- common(u,v) changes only if u is a neighbor of a or b, or v is a neighbor of a or b.
- Specifically, for any pair (u,v), common(u,v) increases by 1 if:
  - u is neighbor of a and v is neighbor of b (and u != b, v != a, u != v)
  - OR u is neighbor of b and v is neighbor of a (and u != a, v != b, u != v)
  
This is getting complex. Let me use a simpler approach for n=31:
- Maintain common matrix
- When adding edge (a,b):
  - For each x in N(a) \ {b}: common(x,b) += 1 (since a is now a common neighbor of x and b)
    Wait, no. common(x,b) = |N(x) ∩ N(b)|. Adding edge (a,b) means b's neighbor set gains a.
    So common(x,b) increases by 1 iff a is in N(x), i.e., x is neighbor of a.
    So for each x in N(a) \ {b}: common(x,b) += 1, common(b,x) += 1
  - For each y in N(b) \ {a}: common(a,y) += 1, common(y,a) += 1
  
  The change in C4 count:
  For each affected pair (x,b) where x in N(a)\{b}:
    Old contribution: C(common(x,b), 2)
    New contribution: C(common(x,b)+1, 2)
    Delta: common(x,b)
  Similarly for (b,x): same delta, but we count each unordered pair once? 
  
  Actually, the sum is over all ordered pairs (u,v) with u≠v? Or all pairs u,v?
  Let's use: C4_count = (1/4) * sum_{u=0}^{n-1} sum_{v=0, v≠u}^{n-1} C(common(u,v), 2)
  
  No, the standard formula is: number of C4s = (1/4) * sum_{u,v} C(common(u,v), 2)
  where the sum is over all ordered pairs (u,v), u≠v. Each C4 is counted 4 times
  (once for each pair of opposite vertices, and each pair appears in 2 orders, so 4 total).
  
  Wait, let me verify: A C4 has vertices a-b-c-d-a. The opposite pairs are (a,c) and (b,d).
  For pair (a,c), common(a,c) ≥ 2 (neighbors b and d). C(common(a,c),2) counts the number
  of ways to choose 2 common neighbors, which gives the number of C4s where a,c are opposite.
  Each C4 is counted once for (a,c) and once for (c,a) [if ordered], and once for (b,d) and (d,b).
  So if we sum over all ordered pairs (u,v), u≠v, each C4 is counted 4 times.
  Hence C4_count = (1/4) * sum_{u≠v} C(common(u,v), 2).
  
  But common(u,v) = common(v,u), so we can also write:
  C4_count = (1/2) * sum_{u<v} C(common(u,v), 2) * 2 / 4? No.
  
  Let me just use: C4_count = (1/4) * sum_{u=0}^{n-1} sum_{v=0}^{n-1, v≠u} C(common(u,v), 2)
  
  Since common(u,v) = common(v,u), this equals (1/2) * sum_{u≠v} C(common(u,v), 2) / 2? 
  
  Actually, let me just compute it directly. For n=31, I'll maintain the common matrix
  and compute the C4 count as needed, or incrementally.

Given the complexity, let me use a different approach:
1. Build a C4-free graph with 90 edges using greedy with backtracking or just greedy.
2. Add the 91st edge that creates the fewest C4s.
3. Perform local search: swap edges to reduce C4 count.

For the greedy C4-free construction, I'll use the fact that we need a graph where
no two vertices have 2 or more common neighbors. This is equivalent to a graph where
the adjacency matrix A satisfies that A^2 has entries ≤ 1 for all off-diagonal entries
(and diagonal entries are degrees, which can be anything).

Actually, for a C4-free graph, no two vertices can have 2 common neighbors. So
common(u,v) ≤ 1 for all u≠v.

Let me construct such a graph greedily.
"""

import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140
    
    rng = random.Random(42)
    
    def count_c4(adj):
        """Count C4s in graph given by adjacency matrix."""
        A = adj.astype(np.int32)
        C = A @ A
        # C[u,v] = number of common neighbors of u and v
        # C4 count = (1/4) * sum_{u,v} C(C[u,v], 2) for u≠v
        # But C[u,u] = degree(u), and C(degree, 2) is not a C4.
        # We need to exclude u=v.
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = C[u, v]
                if c >= 2:
                    total += c * (c - 1) // 2
        # Each C4 is counted twice (once for each pair of opposite vertices)
        return total // 2
    
    def build_c4_free_graph():
        """Build a C4-free graph with as many edges as possible (aim for 90)."""
        adj = np.zeros((n, n), dtype=np.int8)
        common = np.zeros((n, n), dtype=np.int32)
        
        # Try to add edges in a smart order
        # Use a greedy approach: try all possible edges, add if it doesn't create a C4
        # To get 90 edges, we need a specific structure.
        
        # Let's use a known construction: the polarity graph of PG(2,5)
        # PG(2,5) has 31 points. A polarity graph has edges between point and its polar line.
        
        # Finite field GF(5)
        # Points are 1D subspaces of GF(5)^3, so 31 points.
        # Represent points as (x, y, z) with (x,y,z) ≠ (0,0,0), modulo scaling.
        
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero coordinate 1
                    if x != 0:
                        inv = pow(x, -1, 5)
                        pts = (1, (y * inv) % 5, (z * inv) % 5)
                    elif y != 0:
                        inv = pow(y, -1, 5)
                        pts = (0, 1, (z * inv) % 5)
                    else:
                        pts = (0, 0, 1)
                    if pts not in points:
                        points.append(pts)
        
        # There should be 31 points
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        # Define a polarity: a symmetric bilinear form B(p, q) = p0*q0 + p1*q1 + p2*q2
        # Point p is adjacent to point q if B(p, q) = 0
        # This gives a self-adjacent graph (polarity graph)
        
        def dot_product(p, q):
            return (p[0] * q[0] + p[1] * q[1] + p[2] * q[2]) % 5
        
        adj = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dot_product(points[i], points[j]) == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edge_count += 1
            # Self-adjacency: check if point is on its polar line
            if dot_product(points[i], points[i]) == 0:
                # This would be a self-loop, but we don't count self-loops as edges
                pass
        
        # The polarity graph of PG(2,5) with this bilinear form should have
        # each vertex adjacent to its polar line, which has 6 points in PG(2,5).
        # So it should be 6-regular, giving 31*6/2 = 93 edges? 
        # But some points may be self-adjacent.
        
        # Let's just use whatever we get and see the edge count.
        # If it's not 90, we can adjust.
        
        return adj, edge_count
    
    # First, try to build a good base graph
    base_adj, base_edges = build_c4_free_graph()
    
    # If we don't have exactly 90 edges, adjust
    # Let's check the C4 count of the base graph
    base_c4 = count_c4(base_adj)
    
    # Now we need to get to 91 edges. If base has 90 edges, add 1.
    # If base has different number, we need to adjust.
    
    # Let's just do a general approach: start with a random 91-edge graph and optimize.
    # But use the polarity graph as a starting point if it's close.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_adj = None
    best_c4 = float('inf')
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Use polarity graph base, add best edge
    if 85 <= base_edges <= 95:
        strategies.append(('polarity', base_adj.copy()))
    
    # Strategy 2: Random graph
    for _ in range(3):
        edges = rng.sample(all_pairs, m)
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        strategies.append(('random', adj))
    
    for strategy_name, adj in strategies:
        if time.time() > deadline - 10:
            break
        
        # Ensure we have exactly 91 edges
        current_edges = adj.sum() // 2
        if current_edges < m:
            # Add random edges
            existing = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        existing.add((i, j))
            missing = [p for p in all_pairs if p not in existing]
            to_add = rng.sample(missing, m - current_edges)
            for u, v in to_add:
                adj[u, v] = 1
                adj[v, u] = 1
        elif current_edges > m:
            # Remove random edges
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            to_remove = rng.sample(existing, current_edges - m)
            for u, v in to_remove:
                adj[u, v] = 0
                adj[v, u] = 0
        
        current_c4 = count_c4(adj)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
        
        # Local search: edge swaps
        T = 100.0
        T_min = 0.5
        cooling = 0.9995
        
        while T > T_min and time.time() < deadline - 5:
            # Pick random edge to remove
            while True:
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u < v and adj[u, v] == 1:
                    break
                if v < u and adj[v, u] == 1:
                    u, v = v, u
                    break
            
            # Pick random non-edge to add
            while True:
                c = rng.randint(0, n - 1)
                d = rng.randint(0, n - 1)
                if c < d and adj[c, d] == 0:
                    break
                if d < c and adj[d, c] == 0:
                    c, d = d, c
                    break
            
            # Compute delta C4
            # Remove (u,v), add (c,d)
            # Temporarily modify adj
            adj[u, v] = 0
            adj[v, u] = 0
            adj[c, d] = 1
            adj[d, c] = 1
            
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[c, d] = 0
                adj[d, c] = 0
            
            T *= cooling
    
    # Extract edges from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
    
    # Ensure we have exactly 91 edges
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    
    return final_edges

if __name__ == "__main__":
    edges = construct()
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A = adj.astype(np.int32)
    C = A @ A
    total = 0
    for u in range(31):
        for v in range(u + 1, 31):
            c = C[u, v]
            if c >= 2:
                total += c * (c - 1) // 2
    print(f"C4: {total // 2}")
