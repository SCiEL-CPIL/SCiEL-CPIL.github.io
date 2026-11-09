"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 20
    best_edges = [[0, v, 0] for v in range(1, n)]  # star fallback

    def check_rainbow_4cycles(edges):
        """Check if every 4-cycle has four distinct colours. Returns True if valid."""
        # Build adjacency: adj[u] = set of (v, c)
        adj = {i: set() for i in range(n)}
        for u, v, c in edges:
            adj[u].add((v, c))
            adj[v].add((u, c))
        
        # For each pair of vertices (u, v), find common neighbours
        # A 4-cycle is u-a-v-b-u where a != b, a != u,v, b != u,v
        # More efficiently: for each pair (u, v) that are NOT adjacent, 
        # check all pairs of common neighbours (a, b) with a < b
        # Actually, a 4-cycle u-a-v-b exists when u and v share two common neighbours a and b
        # AND the cycle is u-a-v-b-u, which requires edges u-a, a-v, v-b, b-u
        
        # Let's enumerate 4-cycles properly:
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges a-b, b-c, c-d, d-a
        # We need to check all such cycles.
        
        # Build adjacency matrix for quick lookup
        # adj_matrix[u][v] = c or None
        adj_matrix = [[None]*n for _ in range(n)]
        for u, v, c in edges:
            adj_matrix[u][v] = c
            adj_matrix[v][u] = c
        
        # For each 4-tuple (a,b,c,d) with a<b<c<d, check if it forms a 4-cycle
        # A 4-cycle on vertices a,b,c,d can be: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
        # We need to check all 3 possible 4-cycles on 4 vertices
        
        for a, b, c, d in itertools.combinations(range(n), 4):
            # Check the 3 possible 4-cycles on these 4 vertices
            # Cycle 1: a-b-c-d-a
            if (adj_matrix[a][b] is not None and adj_matrix[b][c] is not None and 
                adj_matrix[c][d] is not None and adj_matrix[d][a] is not None):
                colours = [adj_matrix[a][b], adj_matrix[b][c], adj_matrix[c][d], adj_matrix[d][a]]
                if len(set(colours)) != 4:
                    return False
            
            # Cycle 2: a-b-d-c-a
            if (adj_matrix[a][b] is not None and adj_matrix[b][d] is not None and 
                adj_matrix[d][c] is not None and adj_matrix[c][a] is not None):
                colours = [adj_matrix[a][b], adj_matrix[b][d], adj_matrix[d][c], adj_matrix[c][a]]
                if len(set(colours)) != 4:
                    return False
            
            # Cycle 3: a-c-b-d-a
            if (adj_matrix[a][c] is not None and adj_matrix[c][b] is not None and 
                adj_matrix[b][d] is not None and adj_matrix[d][a] is not None):
                colours = [adj_matrix[a][c], adj_matrix[c][b], adj_matrix[b][d], adj_matrix[d][a]]
                if len(set(colours)) != 4:
                    return False
        
        return True

    def try_random_construction():
        """Try to build a dense valid graph by adding edges one at a time with random colours."""
        # Start with all possible edges, try to add them in random order
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        edges = []
        adj_matrix = [[None]*n for _ in range(n)]
        
        # For efficiency, maintain a set of existing edges
        edge_set = set()
        
        for u, v in all_edges:
            # Try each colour
            for c in range(n):
                # Check if adding edge (u,v) with colour c would create a bad 4-cycle
                # A new 4-cycle involving (u,v) must use u and v as opposite vertices or adjacent
                # If u-v is an edge, then 4-cycles containing it: u-v-x-y-u where x,y are common neighbours... 
                # Actually, u-v is part of the cycle, so cycles are: u-v-x-y-u where x adjacent to v, y adjacent to u, x!=y, x!=u,v, y!=u,v, and x-y not needed... 
                # Wait, 4-cycle u-v-x-y-u requires edges: u-v, v-x, x-y, y-u
                # So we need: v-x exists, y-u exists, x-y exists, and all four colours distinct
                
                # Let's check all possible 4-cycles that include the new edge (u,v)
                # Case 1: u and v are adjacent in the cycle: u-v-x-y-u
                #   Need: edge v-x, edge x-y, edge y-u all exist
                # Case 2: u and v are opposite in the cycle: u-x-v-y-u
                #   Need: edge u-x, edge x-v, edge v-y, edge y-u all exist
                #   But u-x and x-v means x is common neighbour of u and v
                #   Similarly y is common neighbour
                #   So we need two common neighbours x, y of u and v with x-y not required... 
                #   Wait, cycle is u-x-v-y-u, so edges are u-x, x-v, v-y, y-u
                #   x and y don't need to be connected
                
                valid = True
                
                # Case 1: u-v-x-y-u (u,v adjacent in cycle)
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj_matrix[v][x] is None:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if adj_matrix[y][u] is None:
                            continue
                        if adj_matrix[x][y] is None:
                            continue
                        # Check colours: c (for u-v), adj_matrix[v][x], adj_matrix[x][y], adj_matrix[y][u]
                        colours = [c, adj_matrix[v][x], adj_matrix[x][y], adj_matrix[y][u]]
                        if len(set(colours)) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Case 2: u-x-v-y-u (u,v opposite in cycle)
                # Find all common neighbours of u and v
                common = []
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj_matrix[u][x] is not None and adj_matrix[x][v] is not None:
                        common.append(x)
                
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        x = common[i]
                        y = common[j]
                        # Cycle: u-x-v-y-u
                        # Edges: u-x, x-v, v-y, y-u
                        # Colours: adj_matrix[u][x], adj_matrix[x][v], adj_matrix[v][y], adj_matrix[y][u]
                        colours = [adj_matrix[u][x], adj_matrix[x][v], adj_matrix[v][y], adj_matrix[y][u]]
                        if len(set(colours)) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # If we get here, this colour works
                edges.append([u, v, c])
                adj_matrix[u][v] = c
                adj_matrix[v][u] = c
                edge_set.add((min(u,v), max(u,v)))
                break  # Move to next edge
            
            if time.time() > deadline - 5.0:
                break
        
        return edges

    # Try multiple random constructions
    best_len = len(best_edges)
    
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(attempt * 7919 + 1)
        edges = try_random_construction()
        
        if len(edges) > best_len:
            # Verify the full solution
            if check_rainbow_4cycles(edges):
                best_edges = edges
                best_len = len(edges)
        
        if time.time() > deadline - 5.0:
            break
    
    return best_edges

# EVOLVE-BLOCK-END
