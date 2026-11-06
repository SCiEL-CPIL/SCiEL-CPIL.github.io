import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns a set of valid colors for edge (u, v).
        If the set is empty, the edge cannot be added.
        """
        forbidden = set()
        
        # Check all 4-cycles u-v-w-x-u
        # w is neighbor of v, x is neighbor of u
        # w != u, x != v, w != x
        # w ~ x must hold
        
        for w in neighbors[v]:
            if w == u: continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                # Cycle u-v-w-x-u
                # Edges: (u,v) [new], (v,w), (w,x), (x,u)
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # The 3 existing edges must have distinct colors among themselves
                # If not, the 4-cycle cannot be rainbow regardless of c(u,v)
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                # The new color must be distinct from the existing 3
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        valid_colors = set(range(num_colors)) - forbidden
        return valid_colors

    def verify_and_fix(edges):
        """Verify all 4-cycles are rainbow; if not, return None (shouldn't happen with correct construction)"""
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        # Check all 4-cycles
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[b][c]: continue
                    for d in range(c+1, n):
                        if not adj[c][d]: continue
                        if not adj[d][a]: continue
                        # Check if a-b-c-d-a is a 4-cycle (no chords)
                        if adj[a][c] or adj[b][d]: continue
                        # Check rainbow
                        colors = [color_mat[a][b], color_mat[b][c], color_mat[c][d], color_mat[d][a]]
                        if len(set(colors)) != 4:
                            return None
        return edges

    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
            
            if valid_colors:
                # Pick a random valid color
                c = random.choice(list(valid_colors))
                
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            
        return edges

    def local_search(edges, seed=None):
        """Try edge swaps to potentially improve the solution"""
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        current_edges = list(edges)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try removing an edge and adding a different one
            for i in range(len(current_edges)):
                if time.time() > deadline - 5:
                    break
                    
                u1, v1, c1 = current_edges[i]
                # Try removing this edge
                adj[u1][v1] = 0
                adj[v1][u1] = 0
                color_mat[u1][v1] = -1
                color_mat[v1][u1] = -1
                neighbors[u1].discard(v1)
                neighbors[v1].discard(u1)
                
                # Try adding a new edge
                added = False
                for u2 in range(n):
                    for v2 in range(u2+1, n):
                        if adj[u2][v2]: continue
                        if u2 == u1 and v2 == v1: continue
                        
                        valid_colors = get_valid_colors(u2, v2, adj, color_mat, neighbors)
                        if valid_colors:
                            c2 = random.choice(list(valid_colors))
                            adj[u2][v2] = 1
                            adj[v2][u2] = 1
                            color_mat[u2][v2] = c2
                            color_mat[v2][u2] = c2
                            neighbors[u2].add(v2)
                            neighbors[v2].add(u2)
                            added = True
                            break
                    if added:
                        break
                
                if added:
                    # Build new edge list
                    new_edges = []
                    for u, v, c in current_edges:
                        if not (u == u1 and v == v1) and not (u == v1 and v == u1):
                            new_edges.append([u, v, c])
                    
                    # Find the added edge
                    for u2 in range(n):
                        for v2 in range(u2+1, n):
                            if adj[u2][v2] and not ((u2, v2) in [(a, b) for a, b, _ in new_edges] and (v2, u2) in [(a, b) for a, b, _ in new_edges]):
                                if u2 != u1 or v2 != v1:
                                    new_edges.append([u2, v2, color_mat[u2][v2]])
                                    break
                        else:
                            continue
                        break
                    
                    current_edges = new_edges
                    improved = True
                    break
                else:
                    # Restore the removed edge
                    adj[u1][v1] = 1
                    adj[v1][u1] = 1
                    color_mat[u1][v1] = c1
                    color_mat[v1][u1] = c1
                    neighbors[u1].add(v1)
                    neighbors[v1].add(u1)
        
        return current_edges

    restarts = 0
    # Run as many restarts as possible
    while time.time() < deadline:
        edges = run_greedy(seed=restarts)
        
        # Try local search to improve
        if time.time() < deadline - 10:
            improved_edges = local_search(edges, seed=restarts)
            if improved_edges and len(improved_edges) > len(edges):
                edges = improved_edges
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        if best_count == 190:
            break
            
        restarts += 1

    return best_edges
