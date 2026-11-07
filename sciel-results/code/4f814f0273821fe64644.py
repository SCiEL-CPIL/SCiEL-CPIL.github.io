"""
Constructs a maximally diameter-2 graph on 16 vertices where every edge is critical.

The graph must be triangle-free. If a triangle exists (u, v, w), removing edge (u, v) 
leaves the path u-w-v of length 2, so the distance between u and v remains 2, 
violating the criticality constraint. 

By Turán's theorem, the maximum number of edges in a triangle-free graph on n vertices 
is floor(n^2 / 4). For n = 16, this is 64. The unique extremal graph is the complete 
bipartite graph K_{8,8}.

In K_{8,8}:
- Diameter is 2.
- Every edge (u, v) with u in A, v in B has no common neighbors (since A and B are 
  disjoint and bipartite). Removing (u, v) leaves no path of length 2 between u and v. 
  The shortest path becomes length 3 (e.g., u-b'-a'-v). Thus, diameter increases to 3.

Therefore, K_{8,8} is the optimal solution with 64 edges.
"""

def construct():
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}
    # Partition vertices into two sets: A = {0, ..., 7}, B = {8, ..., 15}
    # Connect every vertex in A to every vertex in B.
    for i in range(half):
        for j in range(half, n):
            # Ensure values are Python int
            edges.append([int(i), int(j)])
            
    return edges
