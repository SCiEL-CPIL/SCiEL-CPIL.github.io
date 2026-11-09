import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # We want to maximize n.
    # Known: n=26 is possible.
    # Try n from 29 down to 26.
    
    def check_c4_creation(adj, c, u, v, d, n):
        """
        Check if adding edge (u, v) to color c creates a C4.
        adj is the adjacency matrix (list of sets or bitmask) BEFORE adding (u,v).
        We need to check if there is a path u-a-b-v in adj[c] with distinct vertices.
        """
        # Get neighbors of u and v in color c
        # adj[c] is a list of sets of neighbors
        nu = adj[c][u]
        nv = adj[c][v]
        
        # We need a, b such that a in nu, b in nv, and (a,b) in adj[c].
        # And u, a, b, v are distinct.
        # Iterate over a in nu
        for a in nu:
            if a == v: continue # u-v edge not in adj yet, but a cannot be v for path u-a-b-v? 
                               # If a=v, path is u-v-b-v? No, a must be intermediate.
            # If a == u, impossible as u not in nu (no self loops)
            # Check if a is distinct from v
            if a == v:
                continue
            # Now look for b in nv such that (a,b) is an edge in adj[c]
            # b must be distinct from u, v, a.
            na = adj[c][a]
            # b must be in nv AND b in na
            # Also b != u, b != v, b != a
            for b in nv:
                if b == u or b == v or b == a:
                    continue
                if b in na:
                    return True
        return False

    def solve_cyclic(n):
        """
        Find a valid cyclic 5-coloring for K_n.
        Returns list of colors for differences 1..m, or None.
        """
        m = n // 2
        # colors[d] for d in 1..m
        colors = [None] * (m + 1)
        
        # adj[c][i] is set of neighbors of i in color c
        adj = [[set() for _ in range(n)] for _ in range(5)]
        
        def backtrack(d_idx):
            if time.time() > deadline:
                return False
            if d_idx > m:
                return True
            
            d = d_idx
            # Try each color
            for c in range(5):
                # Check if adding difference d to color c is valid
                # We add edges (i, i+d) for all i.
                # Due to symmetry, we just need to check one representative?
                # No, the graph is circulant, so if one edge creates a C4, all do?
                # Actually, the structure is translation invariant.
                # So we just check if adding edge (0, d) creates a C4?
                # Wait, adding (0, d) adds the edge. But the other edges (i, i+d) are also added.
                # However, due to symmetry, if a C4 is formed, it will be formed involving (0, d) or a translate.
                # Since the graph is vertex-transitive, checking if (0, d) is part of a new C4 is sufficient?
                # Let's check if adding (0, d) creates a C4 in the context of the existing graph.
                # The existing graph is also circulant.
                # So we just check if there is a path 0-a-b-d in adj[c].
                
                # Let's verify this logic.
                # If we add the full set of edges for difference d, a C4 might be formed by edges
                # (i, i+d), (i+d, i+2d), ...
                # But since the graph is circulant, the existence of ANY C4 implies the existence
                # of a C4 containing the edge (0, d) (by translation).
                # So yes, we just check for a path 0-a-b-d in adj[c].
                
                # Path 0 -> a -> b -> d
                # a in adj[c][0]
                # b in adj[c][d]
                # (a, b) in adj[c]
                # 0, a, b, d distinct.
                
                valid = True
                # a must be neighbor of 0
                # b must be neighbor of d
                # (a,b) must be edge
                # Distinctness: a != 0, a != d, b != 0, b != d, a != b.
                # Since 0 and d are distinct (d>=1, d<=m < n), and no self loops, a!=0, b!=d is implicit if we check sets?
                # adj[c][0] does not contain 0.
                # adj[c][d] does not contain d.
                
                for a in adj[c][0]:
                    if a == d:
                        continue
                    # a is distinct from 0 and d
                    for b in adj[c][d]:
                        if b == 0 or b == d or b == a:
                            continue
                        # Check if (a,b) is an edge
                        if b in adj[c][a]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    colors[d] = c
                    # Add edges for difference d to adj[c]
                    # Edges: (i, i+d) for i in 0..n-1
                    # We add to adj sets
                    added_edges = []
                    for i in range(n):
                        j = (i + d) % n
                        if i < j:
                            adj[c][i].add(j)
                            adj[c][j].add(i)
                            added_edges.append((i, j))
                        else:
                            # i > j, so edge is (j, i)
                            adj[c][i].add(j)
                            adj[c][j].add(i)
                            added_edges.append((j, i))
                    
                    if backtrack(d_idx + 1):
                        return True
                    
                    # Undo
                    for u, v in added_edges:
                        adj[c][u].remove(v)
                        adj[c][v].remove(u)
                    colors[d] = None
                    
            return False
        
        return backtrack(1)

    # Try n from 29 down to 26
    best_n = 0
    best_colors = []
    
    for n in range(29, 25, -1):
        if time.time() > deadline:
            break
            
        # Optional: Symmetry breaking.
        # The color of difference 1 can be fixed to 0?
        # In a cyclic coloring, permuting colors is an automorphism.
        # So we can fix colors[1] = 0.
        
        # Let's modify solve_cyclic to allow fixing colors[1]=0
        def solve_cyclic_fixed(n):
            m = n // 2
            colors = [None] * (m + 1)
            adj = [[set() for _ in range(n)] for _ in range(5)]
            
            # Fix colors[1] = 0
            colors[1] = 0
            # Add edges for d=1 to color 0
            for i in range(n):
                j = (i + 1) % n
                if i < j:
                    adj[0][i].add(j)
                    adj[0][j].add(i)
                else:
                    adj[0][i].add(j)
                    adj[0][j].add(i)
            
            def backtrack(d_idx):
                if time.time() > deadline:
                    return False
                if d_idx > m:
                    return True
                
                d = d_idx
                for c in range(5):
                    if d == 1:
                        continue # Already fixed
                    
                    valid = True
                    for a in adj[c][0]:
                        if a == d:
                            continue
                        for b in adj[c][d]:
                            if b == 0 or b == d or b == a:
                                continue
                            if b in adj[c][a]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        colors[d] = c
                        added_edges = []
                        for i in range(n):
                            j = (i + d) % n
                            if i < j:
                                adj[c][i].add(j)
                                adj[c][j].add(i)
                                added_edges.append((i, j))
                            else:
                                adj[c][i].add(j)
                                adj[c][j].add(i)
                                added_edges.append((j, i))
                        
                        if backtrack(d_idx + 1):
                            return True
                        
                        for u, v in added_edges:
                            adj[c][u].remove(v)
                            adj[c][v].remove(u)
                        colors[d] = None
                        
                return False
            
            return backtrack(2)
        
        result = solve_cyclic_fixed(n)
        if result:
            best_n = n
            # Construct the full coloring
            # colors[d] is defined for d=1..m
            # Edge (i, j) has difference d = (j-i) % n.
            # If d > n/2, use n-d.
            # Color is colors[min(d, n-d)]
            
            full_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = (j - i) % n
                    d_min = min(d, n - d)
                    full_colors.append(colors[d_min])
            
            # Verify
            # Check for C4 in each color
            valid = True
            for c in range(5):
                # Build adj matrix for color c
                adj_mat = [[0]*n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if full_colors[idx] == c:
                            adj_mat[i][j] = 1
                            adj_mat[j][i] = 1
                        idx += 1
                # Check C4
                # For each pair (u, v), count common neighbors
                for u in range(n):
                    for v in range(u+1, n):
                        common = 0
                        for k in range(n):
                            if adj_mat[u][k] and adj_mat[v][k]:
                                common += 1
                        if common >= 2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_colors = full_colors
                break # Found max n, stop
            else:
                # Should not happen if backtracking is correct
                print(f"Error: Invalid coloring found for n={n}", file=sys.stderr)
        
        # If we didn't find a solution for n, try smaller n
        # Note: solve_cyclic_fixed might take a long time.
        # If it times out, we break the inner loop?
        # The deadline check is inside backtrack.
        # If it returns False due to time, we move to next n.
        
    # If no cyclic solution found, fallback to n=11 or known small solution?
    # But n=26 is known to exist. The cyclic construction might not find it if it's not cyclic?
    # Lazebnik-Woldar is not necessarily cyclic in the simple sense?
    # Actually, many such constructions are based on finite fields or groups.
    # If cyclic fails, we might need a more general search.
    
    # However, given the constraints, cyclic is the best bet for a quick strong solution.
    # If n=26 is not found cyclically, we might return a smaller n.
    
    if best_n == 0:
        # Fallback: Return a valid small coloring
        # n=5 is always possible (K5 can be 5-colored trivially, no C4)
        # Let's just return n=1 with empty colors?
        # Or a known small valid one.
        # For n=1, colors is empty.
        best_n = 1
        best_colors = []
        
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
