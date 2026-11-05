"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 20
    # We want a dense graph on 20 vertices with 20 colours such that every 4-cycle is rainbow.
    # Strategy: Use a greedy approach to add edges, choosing colours to avoid monochromatic/non-rainbow 4-cycles.
    # A 4-cycle is rainbow if its four edges have four distinct colours.
    # 
    # Key insight: If we colour edges such that each colour class is a matching (i.e., no two edges of the same colour share a vertex),
    # then any cycle of length 4 would have four edges, and if the colour classes are matchings, then no two edges in the same cycle can share a colour?
    # Actually, if each colour is a matching, then in a 4-cycle, no two adjacent edges can have the same colour (since they share a vertex).
    # But non-adjacent edges (opposite) could have the same colour. So we need to ensure that opposite edges in any 4-cycle have different colours.
    #
    # A stronger condition: if each colour class is a matching, then a 4-cycle has the form a-b-c-d-a. Edges ab, bc, cd, da.
    # ab and cd are non-adjacent, bc and da are non-adjacent. So we need colour(ab) != colour(cd) and colour(bc) != colour(da).
    #
    # With 20 colours and a matching per colour, the maximum degree per colour is 1, so the graph is a union of 20 matchings.
    # This means the maximum degree of the graph is at most 20, but since we have 20 vertices, the complete graph K20 has degree 19.
    # So we can potentially use all 20 colours to cover K20 if we can decompose K20 into 20 matchings? 
    # Wait, K20 has 190 edges. 20 matchings can cover at most 20*10 = 200 edges (each matching on 20 vertices has at most 10 edges).
    # So it's possible in terms of capacity.
    #
    # However, the condition is not just that colour classes are matchings. We also need the rainbow condition for 4-cycles.
    # Let's try a simpler approach: use a proper edge colouring where each colour is a matching, and then verify the 4-cycle condition.
    #
    # Actually, let's think about this differently. The condition "every 4-cycle is rainbow" is equivalent to saying that for any 4-cycle, the four edges have distinct colours.
    # This is a strong condition.
    #
    # Let's try a greedy algorithm:
    # 1. Start with an empty graph.
    # 2. Consider adding edges in some order.
    # 3. For each edge (u,v), try each colour c. If adding (u,v) with colour c creates a non-rainbow 4-cycle, skip this colour.
    # 4. If no colour works, skip the edge.
    #
    # To make this efficient, we need to quickly check if adding (u,v,c) creates a non-rainbow 4-cycle.
    # A 4-cycle containing (u,v) would be u-w-x-v-u or u-w-x-v-u... wait, a 4-cycle containing edge (u,v) must have the form u-v-a-b-u where a,b are distinct vertices.
    # The four edges are (u,v), (v,a), (a,b), (b,u).
    # For this to be a non-rainbow 4-cycle, at least two of these four edges must share a colour.
    # Since we're adding (u,v) with colour c, we need to check if there exist vertices a,b such that:
    #   - (v,a) exists with colour c, OR (a,b) exists with colour c, OR (b,u) exists with colour c
    #   - AND the other edges have colours that cause a collision.
    #
    # Actually, the condition is that the set of colours of the four edges has size < 4.
    # So we need to check all 4-cycles that include (u,v) and see if any would become non-rainbow.
    #
    # Number of 4-cycles containing a given edge (u,v): choose 2 other vertices from the remaining 18, but they must form a path v-a-b-u.
    # So we need a and b such that (v,a) and (b,u) exist, and (a,b) exists.
    #
    # This is O(n^2) per edge addition, which is 400 per edge. With 190 edges, that's 76,000 checks. Each check is O(1) if we have adjacency info.
    # This should be fast enough.
    #
    # Let's implement this.

    # State:
    # adj[u][v] = colour of edge (u,v), or None if no edge.
    adj = [[None]*n for _ in range(n)]
    edges = []  # list of [u, v, c]

    # Precompute: for each vertex, list of neighbours and their edge colours.
    # We'll maintain this incrementally.

    def is_valid_colour(u, v, c):
        """Check if adding edge (u,v) with colour c keeps all 4-cycles rainbow."""
        # Find all 4-cycles containing (u,v): u-v-a-b-u
        # a is a neighbour of v (other than u), b is a neighbour of u (other than v), and (a,b) must exist.
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[v][a] is None:
                continue
            col_va = adj[v][a]
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj[u][b] is None:
                    continue
                col_ub = adj[u][b]
                if adj[a][b] is None:
                    continue
                col_ab = adj[a][b]
                # The four edges are: (u,v) with colour c, (v,a) with col_va, (a,b) with col_ab, (b,u) with col_ub
                cols = [c, col_va, col_ab, col_ub]
                if len(set(cols)) < 4:
                    return False
        return True

    # Order of edges to try: we'll try all pairs in a fixed order.
    # To maximize density, we should try to add as many edges as possible.
    # Let's try edges in a random order to escape local optima, or in a structured order.
    # Let's try a structured order first: all pairs (i,j) with i<j.
    
    # Actually, let's try multiple random restarts and keep the best.
    
    best_edges = []
    best_count = 0

    # Try a few random permutations of edge orders.
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    for attempt in range(50):  # up to 50 attempts
        if time.time() > deadline - 5.0:
            break
        
        random.shuffle(all_pairs)
        
        adj = [[None]*n for _ in range(n)]
        edges = []
        
        for (u,v) in all_pairs:
            if time.time() > deadline - 2.0:
                break
            # Try each colour
            found = False
            for c in range(n):
                if is_valid_colour(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    found = True
                    break
            if not found:
                # Skip this edge
                pass
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Verify the best solution
    if best_edges:
        # Quick verification
        adj = [[None]*n for _ in range(n)]
        for e in best_edges:
            u, v, c = e
            adj[u][v] = c
            adj[v][u] = c
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] is None:
                    continue
                c = adj[u][v]
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if adj[v][a] is None:
                        continue
                    col_va = adj[v][a]
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if adj[u][b] is None:
                            continue
                        col_ub = adj[u][b]
                        if adj[a][b] is None:
                            continue
                        col_ab = adj[a][b]
                        cols = [c, col_va, col_ab, col_ub]
                        if len(set(cols)) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return best_edges
    
    # Fallback: return a star (always valid)
    return [[0, v, 0] for v in range(1, n)]

# EVOLVE-BLOCK-END
