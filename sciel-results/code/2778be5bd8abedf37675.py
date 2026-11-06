"""
Clique Partition via Edge Conflict Graph Coloring.
Strategy:
1. Construct the "edge conflict graph" G_E.
   - Vertices of G_E are the edges of the original graph G.
   - Two vertices in G_E are adjacent if the corresponding edges in G are NOT adjacent (i.e., they do not share a common vertex in G).
   - Note: If two edges in G share a vertex, they are compatible and can be in the same clique (provided all other pairs in the clique are also compatible).
   - Wait, the condition for a set of edges to form a clique in G is that all pairs of edges in the set are adjacent in G.
   - Therefore, in the conflict graph G_E, a valid clique in G corresponds to an independent set in G_E.
   - We want to partition the edges of G into cliques. This is equivalent to coloring the vertices of G_E such that adjacent vertices in G_E have different colors.
   - Why? If two edges are adjacent in G_E (non-adjacent in G), they cannot be in the same clique. Thus, they must have different colors.
   - If two edges are NOT adjacent in G_E (adjacent in G), they *can* be in the same clique, but they might not have to be. However, if we assign them the same color, we must ensure that ALL edges with that color form a clique in G.
   - Standard graph coloring ensures that adjacent vertices have different colors. It does NOT ensure that non-adjacent vertices with the same color form a clique in the original sense?
   - Let's re-verify.
   - A clique in G is a set of vertices $V_C$ such that every pair is connected.
   - The edges of a clique $K_k$ are all pairs of vertices in $V_C$.
   - A partition of edges into cliques means we partition $E(G)$ into sets $E_1, E_2, \dots$ such that each $E_i$ is the edge set of a clique in $G$.
   - Let $C_i$ be the clique in $G$ corresponding to $E_i$.
   - The condition "every edge must lie in exactly one listed clique" is satisfied by the partition.
   - The condition "every listed set must be a clique" means for each $i$, $E_i$ induces a complete graph on its vertex set.
   - This is equivalent to: For any two edges $e, f \in E_i$, $e$ and $f$ must be adjacent in $G$.
   - In $G_E$, $e$ and $f$ are adjacent if they are NOT adjacent in $G$.
   - So, for any two edges $e, f \in E_i$, they must NOT be adjacent in $G_E$.
   - This means each color class in a proper coloring of $G_E$ is an independent set in $G_E$.
   - An independent set in $G_E$ is a set of edges in $G$ such that no two are non-adjacent in $G$. i.e., all pairs are adjacent in $G$.
   - Thus, any proper coloring of $G_E$ yields a partition of $E(G)$ into sets where each set is a clique in $G$.
   - Minimizing the number of cliques is equivalent to minimizing the number of colors in a proper coloring of $G_E$.
   - This is the Chromatic Number problem on $G_E$.
   - We will use a greedy coloring heuristic on $G_E$.

2. Greedy Coloring of G_E:
   - Vertices of $G_E$: Edges of $G$.
   - Adjacency in $G_E$: $e=(u,v)$ and $f=(x,y)$ are adjacent in $G_E$ iff $\{u,v\} \cap \{x,y\} = \emptyset$.
   - Algorithm:
     - Maintain a set of uncolored vertices (edges of G).
     - In each step, pick the vertex (edge) with the highest degree in the remaining conflict graph.
     - Assign it the smallest color (clique ID) such that no neighbor in the remaining conflict graph has that color.
       - Wait, standard greedy coloring assigns the smallest available color considering ALL neighbors, colored or not? No, usually it considers the colors of already colored neighbors.
       - Actually, if we process vertices one by one, we assign a color $c$ to $v$ if no neighbor $u$ of $v$ (that has already been colored) has color $c$.
       - However, to ensure the set of all vertices with color $c$ forms an independent set, we must ensure that no two vertices with color $c$ are adjacent.
       - If we use the standard sequential greedy coloring (assign smallest available color based on colored neighbors), it guarantees a proper coloring.
       - Thus, each color class is an independent set in $G_E$, which corresponds to a clique in $G$.
     - We need to track the actual vertices of the clique in $G$ to output them.
     - For each color $c$, we maintain the set of vertices in $G$ involved in the edges assigned color $c$.
     - At the end, for each color $c$, the set of vertices forms a clique (since all edges with color $c$ are mutually adjacent in $G$).
     - We output these vertex sets as the cliques.

3. Optimization:
   - $G_E$ has $M$ vertices (edges of $G$). For $n=64, p=0.55$, $M \approx 1000$.
   - The conflict graph $G_E$ is dense?
     - Two edges are adjacent in $G_E$ if they are disjoint in $G$.
     - Probability two random edges are disjoint in $G$:
       - Edges are $(u,v)$ and $(x,y)$. Disjoint if $\{u,v\} \cap \{x,y\} = \emptyset$.
       - This happens if the 4 vertices are distinct.
       - In a dense graph, most edges are disjoint. So $G_E$ is dense.
       - Coloring a dense graph with greedy might use many colors.
       - However, we are minimizing the number of cliques.
       - Is this formulation correct?
       - Yes. Partition into cliques = Coloring of Conflict Graph.
       - The quality of the coloring depends on the order of vertices.
       - "Highest degree in remaining graph" is a standard heuristic (Welsh-Powell or similar).
       - Since $G_E$ is dense, the degree is high.
       - Let's implement this.

4. Implementation Details:
   - Represent edges of $G$ by indices $0 \dots M-1$.
   - Build adjacency list for $G_E$.
     - For each pair of edges $(e_i, e_j)$, check if they are disjoint in $G$.
     - If disjoint, add edge to $G_E$.
     - $M \approx 1000$. $M^2 \approx 10^6$. Checking disjointness is $O(1)$.
     - Building adjacency list takes $O(M^2)$. This is fast.
   - Greedy Coloring:
     - `remaining_edges`: Set of edge indices not yet colored.
     - `colors`: List of sets of edge indices for each color.
     - `color_vertex_sets`: List of sets of vertices in $G$ for each color.
     - Loop while `remaining_edges` is not empty:
       - Calculate degree of each edge in `remaining_edges` within `remaining_edges` in $G_E$.
         - Degree of $e$ = number of edges $f$ in `remaining_edges` such that $e, f$ are disjoint in $G$.
       - Pick $e_{max}$ with max degree.
       - Determine available colors for $e_{max}$.
         - A color $c$ is available if no neighbor $f$ of $e_{max}$ in $G_E$ (that is already colored) has color $c$.
         - Wait, the neighbors in $G_E$ that are already colored are those not in `remaining_edges`.
         - Let `colored_neighbors` = $\{ f \mid f \notin \text{remaining\_edges}, f \text{ adjacent to } e_{max} \text{ in } G_E \}$.
         - `used_colors` = $\{ \text{color}(f) \mid f \in \text{colored\_neighbors} \}$.
         - `available_colors` = $\{ 0, 1, \dots \} \setminus \text{used\_colors}$.
         - Pick smallest $c \in \text{available\_colors}$.
       - Assign color $c$ to $e_{max}$.
       - Add $e_{max}$ to `colors[c]`.
       - Add vertices of $e_{max}$ to `color_vertex_sets[c]`.
       - Remove $e_{max}$ from `remaining_edges`.
   - Output:
     - For each color $c$, the clique is `sorted(color_vertex_sets[c])`.
     - Note: We must ensure the clique has at least 2 vertices.
     - If a color class has only 1 edge, the vertex set has 2 vertices. This is valid.
     - If a color class has 0 edges, skip. (Should not happen).

5. Verification:
   - The greedy coloring produces a proper coloring of $G_E$.
   - Thus, each color class is an independent set in $G_E$.
   - Thus, all edges in a color class are adjacent in $G$.
   - Thus, the vertices of each color class form a clique in $G$.
   - The partition is valid.

6. Complexity:
   - Building $G_E$: $O(M^2)$.
   - Greedy Coloring: $M$ steps.
     - In each step, finding max degree: $O(M \cdot \text{deg})$.
     - Finding available colors: $O(\text{deg})$.
     - Total: $O(M^2 \cdot \text{avg\_deg})$.
     - For $M=1000$, $M^2=10^6$. If avg deg is 500, $5 \cdot 10^8$ ops.
     - This might be too slow in Python.
     - Optimization:
       - Instead of recalculating degrees from scratch, maintain degrees dynamically.
       - When removing $e$, decrease degree of its neighbors in `remaining_edges`.
       - Use a heap or sorted list for max degree?
       - Or just scan `remaining_edges` each time. $1000 \times 1000 = 10^6$ checks.
       - Each check is $O(1)$ if we have adjacency matrix of $G_E$.
       - Adjacency matrix of $G_E$: $1000 \times 1000$ bits/bools. Memory ~ 1MB. Fine.
       - Scanning 1000 items 1000 times = $10^6$ iterations.
       - Inside iteration: check adjacency (O(1)), update max.
       - Then find available colors: check neighbors of $e_{max}$.
       - Total ops ~ $10^6$. This is very fast in Python (< 1s).

7. Refinement:
   - The greedy order matters. "Highest degree in remaining graph" is a good heuristic.
   - We can also try "Largest First" (by original degree) but dynamic is better.
   - We will implement the dynamic degree version.

8. Edge Cases:
   - Empty graph: Return [].
   - Single edge: Return [[u, v]].

Let's write the code.
"""

import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    """Generate the same graph as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _solve_coloring(adj, edge_list, n):
    """
    Solve the clique partition problem by coloring the edge conflict graph.
    """
    M = len(edge_list)
    if M == 0:
        return []
    
    # 1. Build Conflict Graph G_E
    # Vertices are 0..M-1 (indices in edge_list)
    # Adjacency: e_i, e_j are adjacent in G_E if they are disjoint in G.
    # We use a list of sets or a bitset? List of sets is fine for M=1000.
    # Or a 2D list of bools for O(1) lookup.
    # M=1000 -> 1M bools. Fine.
    
    # Precompute edge endpoints for fast disjointness check
    # edge_list[i] = (u, v)
    
    # conflict_adj[i] = set of j such that edge i and edge j are disjoint
    conflict_adj = [set() for _ in range(M)]
    
    for i in range(M):
        u1, v1 = edge_list[i]
        for j in range(i + 1, M):
            u2, v2 = edge_list[j]
            # Check if disjoint
            if u1 != u2 and u1 != v2 and v1 != u2 and v1 != v2:
                conflict_adj[i].add(j)
                conflict_adj[j].add(i)
                
    # 2. Greedy Coloring
    # remaining: set of uncolored edge indices
    remaining = set(range(M))
    
    # colors: list of lists of edge indices
    colors = []
    # color_vertices: list of sets of vertices in G
    color_vertices = []
    
    # degree_in_remaining: dict or list?
    # Since M is small, we can just calculate degree on the fly or maintain a list.
    # Let's maintain a list `deg` of length M, initialized to len(conflict_adj[i]).
    # When we remove a node, we update degrees of its neighbors in `remaining`.
    
    deg = [len(s) for s in conflict_adj]
    
    while remaining:
        if _remaining() < 5:
            # Fallback: assign each remaining edge a new unique color
            for e in remaining:
                c = len(colors)
                colors.append([e])
                color_vertices.append(set(edge_list[e]))
            break
            
        # Find edge with max degree in remaining
        max_deg = -1
        best_e = -1
        for e in remaining:
            if deg[e] > max_deg:
                max_deg = deg[e]
                best_e = e
                
        # Find available colors for best_e
        # A color c is unavailable if any neighbor of best_e in conflict_adj has color c.
        # Neighbors of best_e are in conflict_adj[best_e].
        # We only care about neighbors that are ALREADY colored (i.e., not in remaining).
        
        used_colors = set()
        for neighbor in conflict_adj[best_e]:
            if neighbor not in remaining:
                # Find the color of this neighbor
                # We need a map from edge index to color index
                # Let's maintain `edge_color` array.
                pass
        
        # Optimization: Maintain `edge_color` array.
        # Initialize with -1.
        
        # Let's restart the loop logic with `edge_color`.
        
    # Re-implementing with edge_color for clarity and speed
    
    edge_color = [-1] * M
    colors = []
    color_vertices = []
    remaining = set(range(M))
    deg = [len(s) for s in conflict_adj]
    
    while remaining:
        if _remaining() < 5:
            for e in remaining:
                c = len(colors)
                colors.append([e])
                color_vertices.append(set(edge_list[e]))
                edge_color[e] = c
            remaining.clear()
            break
            
        # Find max degree
        max_deg = -1
        best_e = -1
        for e in remaining:
            if deg[e] > max_deg:
                max_deg = deg[e]
                best_e = e
                
        # Find smallest available color
        used_colors = set()
        for neighbor in conflict_adj[best_e]:
            if neighbor not in remaining:
                used_colors.add(edge_color[neighbor])
                
        c = 0
        while c in used_colors:
            c += 1
            
        # Assign color
        edge_color[best_e] = c
        if c >= len(colors):
            colors.append([])
            color_vertices.append(set())
        colors[c].append(best_e)
        color_vertices[c].update(edge_list[best_e])
        
        # Remove best_e from remaining
        remaining.remove(best_e)
        
        # Update degrees of neighbors in remaining
        for neighbor in conflict_adj[best_e]:
            if neighbor in remaining:
                deg[neighbor] -= 1
                
    # Construct result
    result = []
    for i in range(len(colors)):
        if colors[i]:
            verts = sorted(list(color_vertices[i]))
            # Verify it's a clique? 
            # The coloring guarantees it. But let's double check if time permits?
            # No, trust the logic.
            if len(verts) >= 2:
                result.append(verts)
                
    return result

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_cover = [0] * n_edges
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    
    for clique in cliques:
        # Check clique validity
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if not adj[clique[i]][clique[j]]:
                    return False
        # Mark edges
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                e = (min(clique[a], clique[b]), max(clique[a], clique[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
                else:
                    return False
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # Solve via Coloring
        cliques = _solve_coloring(adj, edge_list, n)
        
        # Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback
            cliques = [[u, v] for u, v in edge_list]
            
        results.append(cliques)
        
    return results

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
