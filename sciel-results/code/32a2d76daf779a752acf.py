import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    n = 20
    colors_count = 20
    deadline = time.time() + 115.0 # Safety margin
    
    # 1. Constructive Baseline:
    # Use a Latin Square of order 20.
    # Let vertices be 0..19.
    # We construct a graph G_L based on a cyclic group Z_20 (or any group of order 20).
    # However, a standard Latin square coloring of K_n is:
    # Color of edge (u, v) = (v - u) mod n.
    # Let's check if this satisfies the "rainbow 4-cycle" condition.
    # A 4-cycle is u-v-w-x-u.
    # Edges:
    # e1: (u, v) -> (v - u)
    # e2: (v, w) -> (w - v)
    # e3: (w, x) -> (x - w)
    # e4: (x, u) -> (u - x)
    # Sum of colors: (v-u) + (w-v) + (x-w) + (u-x) = 0 mod n.
    # For the colors to be distinct, we need:
    # v-u != w-v  => 2v != u+w
    # v-u != x-w  => v+w != u+x
    # v-u != u-x  => 2u != v+x
    # w-v != x-w  => 2w != v+x
    # w-v != u-x  => v+x != u+w  (same as 2nd)
    # x-w != u-x  => 2x != u+w
    #
    # If we pick vertices such that u+w = 2v, then c1 == c2. Violation.
    # Example: u=0, v=1, w=2. Then 0+2 = 2*1.
    # Cycle 0-1-2-3-0:
    # c(0,1)=1, c(1,2)=1. Duplicate.
    # So the simple group table coloring of K_n is NOT valid for n>3.
    #
    # Alternative Constructive Strategy:
    # Use a bipartite graph. Bipartite graphs with parts of size 10.
    # If the graph is bipartite, all cycles have even length.
    # We can color edges based on a Latin Square structure on the bipartite graph.
    # Let U = {0..9}, V = {10..19}.
    # Edge (u, v) where u in U, v in V.
    # Let u' = u, v' = v - 10.
    # Color c(u, v) = (u' + v') mod 20.
    # Does this ensure rainbow 4-cycles?
    # A 4-cycle in a bipartite graph is u1-v1-u2-v2-u1.
    # Edges:
    # e1: (u1, v1) -> c1 = u1 + v1
    # e2: (u2, v1) -> c2 = u2 + v1
    # e3: (u2, v2) -> c3 = u2 + v2
    # e4: (u1, v2) -> c4 = u1 + v2
    #
    # We need c1, c2, c3, c4 to be distinct.
    # c1 = c3  => u1+v1 = u2+v2 => u1-u2 = v2-v1
    # c2 = c4  => u2+v1 = u1+v2 => u2-u1 = v2-v1 => u1-u2 = v1-v2
    # If c1=c3 and c2=c4, then v2-v1 = -(v2-v1) => 2(v2-v1)=0.
    # Since we are mod 20, if v1 != v2, v2-v1 is not 0 mod 10 (unless diff is 10, but v's are 0..9, so diff is in -9..9).
    # So c1 != c3 and c2 != c4 automatically for distinct vertices in bipartite sets?
    # Wait, if v1=0, v2=10? No, v2 is 0..9.
    # So c1 != c3 and c2 != c4 is guaranteed if u1!=u2 and v1!=v2.
    #
    # What about c1 = c2?
    # u1+v1 = u2+v1 => u1 = u2. But u1 != u2 in a 4-cycle. So c1 != c2.
    # Similarly c3 != c4.
    #
    # What about c1 = c4?
    # u1+v1 = u1+v2 => v1 = v2. But v1 != v2. So c1 != c4.
    # Similarly c2 != c3.
    #
    # What about c1 = c3? (Checked, requires u1-u2 = v2-v1)
    # What about c2 = c4? (Checked, requires u2-u1 = v2-v1)
    #
    # So the only potential collisions are c1=c3 and c2=c4.
    # c1=c3 <=> u1+v1 = u2+v2.
    # c2=c4 <=> u2+v1 = u1+v2.
    #
    # If we use c(u,v) = (u + v) mod 20, collisions happen if u1+v1 = u2+v2.
    # This is a "sum-free" or "distinct sums" problem.
    # With 10x10 edges, we have 100 edges. Colors are 0..19.
    # By Pigeonhole, many collisions are inevitable in the color assignment itself?
    # No, the constraint is only on 4-cycles.
    # But if c1=c3, the 4-cycle is NOT rainbow.
    # So we must avoid 4-cycles where u1+v1 = u2+v2 AND u2+v1 = u1+v2?
    # Actually, if c1=c3, the cycle has colors {c1, c2, c1, c4}. Not rainbow.
    # So we need to ensure that for ANY 4-cycle, u1+v1 != u2+v2.
    # This is impossible for a complete bipartite graph K_{10,10} with additive coloring mod 20?
    # Let's check:
    # u1=0, v1=0 -> c=0
    # u2=1, v2=1 -> c=2
    # u1=0, v2=1 -> c=1
    # u2=1, v1=0 -> c=1
    # Cycle 0-0-1-1-0 (using indices 0,1 in U and 0,1 in V):
    # Edges: (0,0)c=0, (1,0)c=1, (1,1)c=2, (0,1)c=1.
    # Colors: 0, 1, 2, 1. Duplicate 1. Violation.
    #
    # So additive coloring fails.
    #
    # Try multiplicative? Or XOR?
    # c(u,v) = u XOR v?
    # u1=0, v1=0 -> 0
    # u2=1, v1=0 -> 1
    # u2=1, v2=1 -> 0
    # u1=0, v2=1 -> 1
    # Colors: 0, 1, 0, 1. Violation.
    #
    # It seems hard to color K_{10,10} with 20 colors such that ALL 4-cycles are rainbow.
    # Number of 4-cycles in K_{10,10} is C(10,2)*C(10,2) = 45*45 = 2025.
    #
    # Let's step back.
    # The task asks to MAXIMIZE edges.
    # If K_{10,10} is not easily colorable, maybe a sparser graph is better?
    # Or maybe a different structure?
    #
    # What if we use a random greedy approach to build a dense graph?
    # Start with empty graph.
    # Add edges one by one.
    # For each edge, try all 20 colors.
    # If a color maintains the property (all 4-cycles in the current subgraph are rainbow), keep it.
    # If no color works, skip the edge.
    #
    # This is a "Maximal Rainbow 4-cycle-free" construction.
    # Complexity:
    # When adding edge e, we only need to check 4-cycles involving e.
    # A 4-cycle involving e=(u,v) is of the form u-x-y-v-u.
    # We need x, y such that (u,x), (x,y), (y,v) are in the graph.
    #
    # This is a very different strategy: Greedy Construction vs Local Search on K_n.
    # It guarantees validity at every step.
    # It might produce a graph with significantly fewer than 190 edges, but it's guaranteed valid.
    # If the local search on K_20 fails (which it likely does, given the high density constraint), this is a robust fallback that might actually score better than a star (19 edges) if it can get to, say, 50-100 edges.
    #
    # Let's implement this Greedy Construction.
    
    def greedy_construct():
        # Adjacency list for fast neighbor lookup
        adj = {i: set() for i in range(n)}
        # Edge color map
        edge_col = {}
        
        # List of all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        num_edges_added = 0
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            
            # Find all 4-cycles that would be formed if we add (u,v)
            # Cycle: u - x - y - v - u
            # We need x in adj[u] and y in adj[v] and (x,y) in graph.
            # Also x != y, x != v, y != u.
            
            # Optimization:
            # Iterate over neighbors x of u.
            # For each x, iterate over neighbors y of v.
            # Check if (x,y) is an edge.
            
            valid_colors = [True] * colors_count
            
            # To check validity, we need to ensure that for EVERY 4-cycle containing (u,v),
            # the color of (u,v) is distinct from the other 3 edges.
            
            # Let's collect all "forbidden" colors for (u,v).
            forbidden = [False] * colors_count
            
            # Get neighbors
            neighbors_u = adj[u]
            neighbors_v = adj[v]
            
            # To make intersection fast, ensure sets
            # Iterate over smaller set?
            # Let's just iterate over all pairs (x, y) where x in N(u), y in N(v)
            
            # Optimization: Precompute edge lookup
            # edge_exists(x, y)
            
            for x in neighbors_u:
                if x == v: continue
                for y in neighbors_v:
                    if y == u: continue
                    if x == y: continue
                    
                    # Check if (x, y) is an edge
                    if x > y:
                        key_xy = (x, y)
                    else:
                        key_xy = (y, x)
                        
                    if key_xy in edge_col:
                        # Cycle found: u-x-y-v-u
                        # Edges:
                        # e1: (u, x)
                        # e2: (x, y)
                        # e3: (y, v)
                        # e4: (u, v) - this is the one we are coloring
                        
                        c1 = edge_col[(min(u,x), max(u,x))]
                        c2 = edge_col[key_xy]
                        c3 = edge_col[(min(y,v), max(y,v))]
                        
                        # c4 must be distinct from c1, c2, c3
                        forbidden[c1] = True
                        forbidden[c2] = True
                        forbidden[c3] = True
            
            # If no 4-cycles are formed, any color is valid.
            # If 4-cycles are formed, we must pick a color not in forbidden.
            
            # Find a valid color
            chosen_color = -1
            for c in range(colors_count):
                if not forbidden[c]:
                    chosen_color = c
                    break
            
            if chosen_color != -1:
                # Add edge
                edge_col[(u, v)] = chosen_color
                adj[u].add(v)
                adj[v].add(u)
                num_edges_added += 1
            else:
                # Cannot add this edge with any color
                pass
        
        return edge_col, num_edges_added

    # Run greedy construction
    edge_col, num_edges = greedy_construct()
    
    # If we got a decent number of edges, return them.
    # Convert to required format
    result = []
    for (u, v), c in edge_col.items():
        result.append([u, v, c])
        
    # If the greedy result is very poor (e.g. < 20 edges), try the local search fallback?
    # But local search on K20 is slow and likely fails.
    # Greedy is safe.
    
    return result

# EVOLVE-BLOCK-END
