"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n = 20
    # adjacency: adj[u][v] = colour or -1
    adj = [[-1]*n for _ in range(n)]
    
    # For each colour, track which vertices have an edge of that colour
    # color_edges[c] = set of (u,v) with colour c
    color_edges = [set() for _ in range(n)]
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Try to add edges in a good order
    # Start with a proper edge-colouring of K_20 using 20 colours (round-robin)
    # Then verify and remove bad edges
    
    # Round-robin tournament scheduling for 20 teams
    # This gives a proper edge-colouring with 19 colours for K_20... 
    # Actually K_20 needs 19 colours for proper edge-colouring (odd degree 19)
    # We have 20 colours available
    
    # Let's use a 1-factorization of K_20
    # For even n, K_n can be edge-coloured with n-1 colours (proper)
    # 20 is even, so we need 19 colours for proper edge-colouring
    
    # Build a 1-factorization of K_20
    # Standard method: label vertices 0..19, use round-robin
    # Actually for even n, we can use the "circle method"
    
    # Let me just assign colours greedily to get a proper edge-colouring
    # Then check 4-cycles
    
    # Simpler: assign colour to edge (u,v) as (u+v) % 20? No, that's not proper.
    
    # Let me use: for the complete graph, assign colours via 1-factorization
    # K_20 has 190 edges, 20 colours, so average 9.5 edges per colour
    
    # Greedy proper edge-colouring:
    # Sort edges, assign smallest available colour that doesn't conflict at u or v
    
    # But we also need 4-cycles to be rainbow, which is stronger than proper
    
    # Let me try: colour = (v - u) % 20 for u < v... but that gives 19 distinct values per vertex
    # Actually for fixed u, v-u for v in {u+1,...,u+19} mod 20 gives 19 distinct values
    
    # Let's just build it and check
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            c = (v - u) % n
            edges.append((u, v, c))
    
    # Check if this works: every 4-cycle rainbow?
    # Build adjacency
    adj2 = [[-1]*n for _ in range(n)]
    for u, v, c in edges:
        adj2[u][v] = c
        adj2[v][u] = c
    
    # Check all 4-cycles
    def check_all_4_cycles(adj_mat, n):
        bad = 0
        for a in range(n):
            for b in range(a+1, n):
                if adj_mat[a][b] == -1:
                    continue
                for c in range(b+1, n):
                    if adj_mat[b][c] == -1:
                        continue
                    for d in range(c+1, n):
                        if adj_mat[c][d] == -1:
                            continue
                        if adj_mat[d][a] == -1:
                            continue
                        # 4-cycle a-b-c-d-a
                        colors = [adj_mat[a][b], adj_mat[b][c], adj_mat[c][d], adj_mat[d][a]]
                        if len(set(colors)) != 4:
                            bad += 1
        return bad
    
    bad = check_all_4_cycles(adj2, n)
    if bad == 0:
        return [[u, v, c] for u, v, c in edges]
    
    # If not, remove edges that cause violations
    # For each edge, check if removing it reduces bad cycles
    # This is expensive, so let's do it smarter
    
    # Alternative: start with fewer edges and add greedily
    
    # Greedy insertion: try each edge, add if it doesn't create a bad 4-cycle
    adj3 = [[-1]*n for _ in range(n)]
    result = []
    
    # Try edges in order of... let's try all, in a fixed order
    for u in range(n):
        for v in range(u+1, n):
            if time.time() > deadline:
                break
            # Check if adding (u,v) with some colour creates a bad 4-cycle
            # Try each colour
            found = False
            for c in range(n):
                # Check if colour c is available at u and v (proper edge-colouring)
                ok = True
                for w in range(n):
                    if w != u and w != v and adj3[u][w] == c:
                        ok = False
                        break
                    if w != u and w != v and adj3[v][w] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check 4-cycles: for any w, if (w,u) and (w,v) exist, or paths
                # A 4-cycle involving edge (u,v): u-v-w-x-u or u-v-x-w-u
                # Actually, adding edge (u,v) can create 4-cycles where (u,v) is one edge
                # The 4-cycle is u-v-w-x-u where w,x are other vertices
                # So we need: (v,w) exists, (w,x) exists, (x,u) exists, and the four colours distinct
                # Also u-v-x-w-u is same cycle
                
                bad_cycle = False
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj3[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj3[w][x] == -1:
                            continue
                        if adj3[x][u] == -1:
                            continue
                        # 4-cycle: u-v-w-x-u
                        colors = [c, adj3[v][w], adj3[w][x], adj3[x][u]]
                        if len(set(colors)) != 4:
                            bad_cycle = True
                            break
                    if bad_cycle:
                        break
                if not bad_cycle:
                    # Add edge
                    adj3[u][v] = c
                    adj3[v][u] = c
                    result.append([u, v, c])
                    found = True
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Ensure we have at least the star
    if len(result) < 19:
        return [[0, v, 0] for v in range(1, n)]
    
    return result
# EVOLVE-BLOCK-END
