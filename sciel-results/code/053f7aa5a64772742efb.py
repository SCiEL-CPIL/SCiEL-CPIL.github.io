"""
Construct a dense edge-coloured graph on 20 vertices (0..19) with 20 colours (0..19)
such that every 4-cycle is rainbow (four distinct colours).

Strategy:
- Start with complete graph K20 (190 edges).
- Use a greedy/local-search approach to find a valid edge-colouring,
  or if K20 is impossible, find the densest valid subgraph.

Given the constraint that every 4-cycle must be rainbow, and we have 20 colours,
K20 has C(20,4) = 4845 4-cycles. Each 4-cycle uses 4 edges, and those 4 edges must
have distinct colours. This is a very strong constraint.

Let me think about whether K20 can work. With 190 edges and 20 colours, each colour
is used ~9.5 times on average. A colour class is a matching (since in a simple graph,
two edges of the same colour cannot share a vertex... wait, the problem doesn't say
that! It only says every 4-cycle must have four distinct colours. Two edges of the
same colour CAN share a vertex, as long as they don't form part of a 4-cycle together
with two other edges of the same colour... actually, if two edges of the same colour
share a vertex, they can be part of a 4-cycle with two other edges. Let me re-read.

"every 4-cycle is rainbow" means for every 4-cycle (u,v,w,x,u), the four edges (u,v),(v,w),(w,x),(x,u)
must have four distinct colours.

So two edges of the same colour can share a vertex, as long as they're not in the same 4-cycle
in a way that causes a conflict. Actually, if two edges share a vertex and have the same colour,
they could potentially be in many 4-cycles. For example, if edges (u,v) and (u,w) have the same
colour c, then any 4-cycle containing both edges (u,v) and (u,w) would have two edges of colour c,
violating the rainbow property. But a 4-cycle is a cycle of length 4, so it has 4 edges. If two
adjacent edges in the 4-cycle have the same colour, that's a violation.

So actually, adjacent edges in any 4-cycle must have distinct colours. This means that for any
vertex v, and any two neighbours u and w of v, if u-v-w is a path of length 2, then edge (u,v)
and (v,w) must have different colours... wait, no. They only need to be different if they appear
in the same 4-cycle. Two edges sharing vertex v appear in a 4-cycle together iff there's a path
from u to w of length 2 that doesn't use v, i.e., there exists x such that (u,x) and (x,w) are edges.

This is getting complex. Let me just try a computational approach.

Plan:
1. Try K20 with a smart colouring.
2. If K20 doesn't work, remove edges and recolor.

Actually, let me think about a known construction. For the anti-Ramsey problem on C4,
there are known results. For n vertices and q colours, the maximum number of edges in a
C4-anti-Ramsey graph... 

Let me try a different approach: use a proper edge-colouring first (adjacent edges different colours),
which is a necessary condition. K20 has maximum degree 19, so it needs at least 19 colours for a
proper edge-colouring (by Vizing's theorem, at most 20). We have 20 colours, so a proper
edge-colouring exists.

But proper edge-colouring is necessary but not sufficient for the 4-cycle rainbow property.

Let me just write code that:
1. Starts with K20.
2. Assigns colours greedily or via local search.
3. Checks all 4-cycles.
4. If invalid, removes problematic edges and tries again.

Given time constraints, let me try a structured approach:
- Use a 1-factorisation of K20 (K20 is even, so it has a 1-factorisation into 19 perfect matchings).
- Colour each edge in matching i with colour i. This gives a proper edge-colouring.
- Check if this satisfies the 4-cycle rainbow property.
- If not, try other colourings.

Actually, with a 1-factorisation, any two edges of the same colour are disjoint (they're in a perfect matching). So two edges of the same colour never share a vertex. This means they can never be adjacent in a 4-cycle. So the only way a 4-cycle could have two edges of the same colour is if two opposite edges have the same colour.

In a 4-cycle u-v-w-x-u, the opposite edge pairs are (u,v)&(w,x) and (v,w)&(x,u). With a 1-factorisation, (u,v) and (w,x) have the same colour iff they're in the same perfect matching. Similarly for (v,w) and (x,u).

So the 4-cycle is NOT rainbow iff (u,v) and (w,x) have the same colour, OR (v,w) and (x,u) have the same colour.

For a 1-factorisation, this happens when u and v are matched together in some round, AND w and x are also matched in the same round. This means u-v and w-x are both in the same perfect matching.

So I need a 1-factorisation where for any 4 distinct vertices u,v,w,x, it's never the case that {u,v} and {w,x} are both edges in the same perfect matching.

This is equivalent to saying: no perfect matching contains two edges that are "opposite" in some 4-cycle... but every pair of disjoint edges in a matching is "opposite" in some 4-cycle (as long as the cross edges exist).

In K20, for any two disjoint edges {u,v} and {w,x}, the 4-cycle u-v-w-x-u exists iff (v,w) and (x,u) are edges, which they are in K20. So every pair of disjoint edges in a matching creates a non-rainbow 4-cycle.

Therefore, a 1-factorisation-based colouring will NEVER work for K20, because every matching has 10 edges, and any two of them form a non-rainbow 4-cycle.

So K20 is impossible. We need to remove edges.

The maximum number of edges... Let me think. If we use a proper edge-colouring with 20 colours, each colour class is a matching. Let the matching sizes be m_0, ..., m_19. Sum m_i = number of edges. For the 4-cycle property, we need that for any two edges of the same colour (which are in the same matching), they don't form opposite edges of a 4-cycle. But in a matching, any two edges are disjoint, so they form opposite edges of a 4-cycle iff the four cross edges exist.

To avoid this, for any two edges {u,v} and {w,x} in the same colour class, at least one of (v,w), (w,u), (u,x), (x,v) must be missing... wait, the 4-cycle is u-v-w-x-u, so the cross edges are (v,w) and (x,u). Or u-v-x-w-u, cross edges (v,x) and (w,u). There are two 4-cycles on four vertices: u-v-w-x-u and u-v-x-w-u.

So for edges {u,v} and {w,x} to not be in any 4-cycle together, we need that for both orderings, at least one cross edge is missing. This is complex.

Let me just try a computational search with a time limit.
"""

import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 20
    num_colours = 20
    
    # Build adjacency matrix for K20
    # adj[u][v] = colour or -1 if no edge
    def is_valid_adj(adj):
        """Check if all 4-cycles are rainbow."""
        # For each 4-cycle u-v-w-x, check edges (u,v),(v,w),(w,x),(x,u) exist and have distinct colours
        for u in range(n):
            for v in range(n):
                if v == u or adj[u][v] == -1:
                    continue
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w or adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count
    
    def make_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    edges.append([u, v, adj[u][v]])
        return edges
    
    # Try to find a dense valid graph
    # Start with a structured approach: try complete bipartite graphs or other dense structures
    
    # Let's try: use a proper edge-colouring of a dense subgraph
    # Strategy: try K_{10,10} first (100 edges), then try to add edges
    
    # Actually, let me try a greedy approach:
    # 1. Start with empty graph
    # 2. Add edges one by one, choosing colours to maintain validity
    # This is too slow for K20
    
    # Better: try a known dense structure
    # Complete bipartite K_{10,10}: vertices 0-9 and 10-19
    # Edges only between the two sets
    # Number of 4-cycles: C(10,2)*C(10,2) = 45*45 = 2025
    # With proper edge-colouring using 10 colours (each colour is a perfect matching),
    # opposite edges in a 4-cycle are in the same matching iff... 
    
    # Let me just try a randomised local search with time limit
    
    best_edges = 0
    best_adj = None
    
    # Try several random proper edge-colourings of K20, then remove edges that cause conflicts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random proper edge-colouring of K20
        # Use a greedy approach: colour edges one by one
        adj = [[-1]*n for _ in range(n)]
        
        # Order edges randomly
        edge_list = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edge_list)
        
        valid = True
        for u, v in edge_list:
            # Find a colour that doesn't conflict
            # For a proper edge-colouring, adjacent edges must have different colours
            # Adjacent edges to (u,v) are all edges incident to u or v (except each other)
            forbidden = set()
            for w in range(n):
                if w != v and adj[u][w] != -1:
                    forbidden.add(adj[u][w])
                if w != u and adj[w][v] != -1:
                    forbidden.add(adj[w][v])
            
            available = [c for c in range(num_colours) if c not in forbidden]
            if not available:
                valid = False
                break
            c = random.choice(available)
            adj[u][v] = c
            adj[v][u] = c
        
        if not valid:
            continue
        
        # Now check 4-cycle property and remove edges that violate it
        # Find all violating 4-cycles and remove one edge from each
        removed = True
        iterations = 0
        while removed and iterations < 1000:
            if time.time() > deadline:
                break
            removed = False
            iterations += 1
            
            # Find a violating 4-cycle
            found = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(n):
                    if v == u or adj[u][v] == -1:
                        continue
                    for w in range(n):
                        if w == u or w == v or adj[v][w] == -1:
                            continue
                        for x in range(n):
                            if x == u or x == v or x == w or adj[w][x] == -1 or adj[x][u] == -1:
                                continue
                            c1 = adj[u][v]
                            c2 = adj[v][w]
                            c3 = adj[w][x]
                            c4 = adj[x][u]
                            if len({c1, c2, c3, c4}) < 4:
                                # Remove one of these edges
                                # Remove the edge with the most alternatives (least degree impact)
                                # Just remove (u,v) for simplicity
                                adj[u][v] = -1
                                adj[v][u] = -1
                                found = True
                                removed = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if found:
                continue
            break
        
        if time.time() > deadline:
            break
        
        # Verify
        if is_valid_adj(adj):
            edges = count_edges(adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = [row[:] for row in adj]
    
    if best_adj is not None:
        return make_edge_list(best_adj)
    
    # Fallback: star (19 edges)
    return [[0, v, 0] for v in range(1, 20)]
