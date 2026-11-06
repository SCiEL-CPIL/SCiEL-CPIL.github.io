"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def get_edge_index(i, j, n):
        """Map (i, j) with i < j to index in the flat array."""
        # Number of edges before row i is i * (2*n - i - 1) / 2
        # Index in row i is j - i - 1
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def find_cycles(n, colors):
        """
        Find all monochromatic C6 cycles.
        Returns a list of lists of 6 tuples (u, v) representing the edges in the cycle.
        """
        # Precompute edge colors in a 2D list for O(1) access
        # colors[i][j] = color
        # We only store i < j
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        
        cycles = []
        # Iterate over all combinations of 6 vertices
        # This is O(n^6) which is 38760 for n=20.
        # For each set, check if it forms a C6 in any color.
        # To be efficient, we can just check if the edges form a cycle.
        # A set of 6 vertices forms a C6 if there is a permutation (v0..v5) such that
        # color(v0,v1)==color(v1,v2)==...==color(v5,v0) and all these colors are the same.
        
        # Optimization: Only check if all 6 vertices are distinct.
        # We can iterate through all 6-subsets.
        for verts in itertools.combinations(range(n), 6):
            # There are 5 distinct C6 structures on 6 labeled vertices (up to rotation/reflection)
            # Actually, it's easier to just check all permutations? No, too slow.
            # Let's check if the induced subgraph on these 6 vertices contains a monochromatic C6.
            # The induced subgraph has 15 edges.
            # We can just try to find a cycle of length 6 in the induced subgraph for each color.
            
            # For each color c in 0..3:
            # Build adjacency list for these 6 vertices using only color c edges.
            # Then check for C6.
            
            for c in range(4):
                # Check if there is a C6 in color c using these vertices
                # We can use a simple DFS on the 6 vertices.
                # Since there are only 6 vertices, we can just check all 5! = 120 permutations?
                # Or simpler: check if any vertex has degree 2 in the induced subgraph of color c?
                # No, C6 is a specific structure.
                
                # Let's just check all 6! / 12 = 60 unique cycles?
                # Actually, let's just check if the graph formed by color c edges on these 6 vertices is a C6.
                # It must have exactly 6 edges, all vertices degree 2, and be connected.
                
                edges_in_c = []
                degrees = [0] * 6
                # Map vertex value to index 0-5
                v_map = {v: i for i, v in enumerate(verts)}
                
                for i in range(6):
                    for j in range(i + 1, 6):
                        u = verts[i]
                        v = verts[j]
                        if color_matrix[u][v] == c:
                            edges_in_c.append((i, j))
                            degrees[i] += 1
                            degrees[j] += 1
                
                # If not exactly 6 edges, it's not a C6 (it could be a C6 with chords, but we want simple C6)
                # Wait, the condition is "No colour class may contain a six-cycle".
                # This means if the graph contains *any* cycle of length 6, it's invalid.
                # So if the induced subgraph on these 6 vertices contains a C6, it's a hit.
                # A C6 is a subgraph, not necessarily induced.
                # So we just need to check if there exists a cycle of length 6 using only these 6 vertices.
                # This is equivalent to checking if the graph (V, E_c) contains a C6.
                
                # For 6 vertices, we can just check all permutations?
                # No, we can check if there is a path of length 5 from u to v where (u,v) is an edge.
                # With only 6 vertices, this is very fast.
                
                # Let's build the adjacency list for the 6 vertices in color c
                adj = [[] for _ in range(6)]
                for u, v in edges_in_c:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check for C6
                # Fix edge (0, 1). Look for path 1-2-3-4-5-0.
                # We can just do a DFS from 1 to 0 of length 4, avoiding 0 until the end.
                def dfs(cur, depth, visited):
                    if depth == 0:
                        return cur == 0
                    for nxt in adj[cur]:
                        if nxt not in visited:
                            if dfs(nxt, depth - 1, visited | {nxt}):
                                return True
                    return False
                
                found = False
                for u, v in edges_in_c:
                    # Try to find path v -> ... -> u of length 4
                    # Remove u and v from the path search temporarily?
                    # The path is v - x - y - z - w - u.
                    # x,y,z,w must be distinct from u,v and each other.
                    # So we search for path from v to u of length 4 in the graph where u is not an intermediate node.
                    # Since we only have 6 nodes, we can just check all permutations of the other 4 nodes.
                    # Or just use the DFS above.
                    
                    # The DFS above checks if there is a path of length 4 from cur to 0.
                    # We need to be careful with the starting node and target.
                    
                    # Let's restart the check for this specific edge (u,v)
                    # We want a path v - a - b - c - d - u.
                    # a,b,c,d are distinct and not in {u,v}.
                    
                    # Since there are only 4 other nodes, we can just check all permutations of the 4 other nodes.
                    # Other nodes: all 6 nodes except u and v.
                    others = [k for k in range(6) if k != u and k != v]
                    
                    for perm in itertools.permutations(others):
                        a, b, c, d = perm
                        # Check edges: (v,a), (a,b), (b,c), (c,d), (d,u)
                        if (a in adj[v] and b in adj[a] and c in adj[b] and d in adj[c] and u in adj[d]):
                            found = True
                            break
                    if found:
                        break
                
                if found:
                    # Reconstruct the cycle edges in original vertex labels
                    # The cycle is v - a - b - c - d - u - v
                    # We need to return the edges in the cycle.
                    # We found 'perm' and 'u','v'.
                    # The edges are:
                    # (verts[v], verts[a]), (verts[a], verts[b]), (verts[b], verts[c]), (verts[c], verts[d]), (verts[d], verts[u]), (verts[u], verts[v])
                    # Note: u and v are indices in the 0-5 range.
                    # We need to store these edges to recolor them.
                    cycle_edges = [
                        (verts[v], verts[a]),
                        (verts[a], verts[b]),
                        (verts[b], verts[c]),
                        (verts[c], verts[d]),
                        (verts[d], verts[u]),
                        (verts[u], verts[v])
                    ]
                    cycles.append(cycle_edges)
                    # We can break early if we just want one cycle, but for fitness we might want count.
                    # For repair, we just need one.
                    # Let's return immediately if we just want one cycle to save time.
                    # But the function name is find_cycles, implying list.
                    # For the repair step, we can just return the first one found.
                    return [cycle_edges]
        
        return cycles

    def repair_search(n, time_limit):
        """
        Local search with repair.
        """
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        deadline = time.time() + time_limit
        
        while time.time() < deadline:
            cycles = find_cycles(n, colors)
            if not cycles:
                return colors
            
            # Pick a random cycle
            cycle = random.choice(cycles)
            # Pick a random edge in the cycle
            u, v = random.choice(cycle)
            
            # Recolor this edge to a random other color
            old_c = colors[get_edge_index(min(u,v), max(u,v), n)]
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
            
            colors[get_edge_index(min(u,v), max(u,v), n)] = new_c
        
        return None

    # Try n=17
    res = repair_search(17, 60)
    if res:
        # Verify
        if not find_cycles(17, res):
            return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 30:
        res = repair_search(16, 25)
        if res:
            if not find_cycles(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 15:
        res = repair_search(15, 12)
        if res:
            if not find_cycles(15, res):
                return {'n': 15, 'colors': res}

    # Try n=14
    if time.time() < hard_deadline - 8:
        res = repair_search(14, 6)
        if res:
            if not find_cycles(14, res):
                return {'n': 14, 'colors': res}

    # Try n=13
    if time.time() < hard_deadline - 4:
        res = repair_search(13, 3)
        if res:
            if not find_cycles(13, res):
                return {'n': 13, 'colors': res}

    # Fallback to n=10
    if time.time() < hard_deadline - 2:
        res = repair_search(10, 1.5)
        if res:
            if not find_cycles(10, res):
                return {'n': 10, 'colors': res}

    # Absolute fallback: n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if not find_cycles(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
