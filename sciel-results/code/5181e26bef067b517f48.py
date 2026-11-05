import random
import time
import itertools

def construct():
    deadline = time.time() + 110
    n = 20
    num_colors = 20
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # adj[u][v] = color if edge exists, -1 otherwise
    # neighbors[u] = set of v such that adj[u][v] != -1
    
    def get_forbidden_colors(u, v, adj, neighbors):
        """
        Returns the set of colors that cannot be used for edge (u, v)
        such that all 4-cycles involving (u, v) are rainbow.
        """
        forbidden = set()
        # 4-cycles are u-a-v-b-u
        # a is neighbor of u, b is neighbor of v
        # a != v, b != u, a != b
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                # Check if (a, b) is an edge
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    # The 4-cycle edges are (u,a), (a,v), (v,b), (b,u) -- Wait.
                    # Cycle is u-a-v-b-u.
                    # Edges: (u,a), (a,v), (v,b), (b,u).
                    # Colors: ca, cv, adj[v][b], adj[u][b]?
                    # No, adj[v][b] is the edge (v,b). adj[u][b] is not in the cycle unless b is neighbor of u.
                    # Wait, the cycle is u-a-v-b-u.
                    # Edges: (u,a), (a,v), (v,b), (b,u).
                    # (u,a) color: adj[u][a]
                    # (a,v) color: adj[a][v]
                    # (v,b) color: adj[v][b]
                    # (b,u) color: adj[b][u]
                    
                    # My previous logic in the prompt's code was:
                    # ca = adj[u][a]
                    # cb = adj[a][b]  <-- This assumes (a,b) is an edge, which it is.
                    # cv = adj[b][v]
                    # This checks if (u,a), (a,b), (b,v) are distinct?
                    # No, a 4-cycle is u-a-v-b-u.
                    # The edges are (u,a), (a,v), (v,b), (b,u).
                    # The prompt's code checked:
                    # if adj[a][b] != -1:
                    #   cb = adj[a][b]
                    #   cv = adj[b][v]
                    #   if ca == cb or ca == cv or cb == cv:
                    #   forbidden.add(ca); forbidden.add(cb); forbidden.add(cv)
                    
                    # This logic in the prompt seems to be checking triangles or something else?
                    # Or is it checking 4-cycles of the form u-a-b-v-u?
                    # If the cycle is u-a-b-v-u, edges are (u,a), (a,b), (b,v), (v,u).
                    # Then the colors are adj[u][a], adj[a][b], adj[b][v], adj[v][u].
                    # The prompt's code:
                    # ca = adj[u][a]
                    # cb = adj[a][b]
                    # cv = adj[b][v]
                    # It checks if these 3 are distinct.
                    # And forbids them for the new edge (u,v).
                    # This implies the new edge (u,v) completes the cycle u-a-b-v-u.
                    # Yes! The cycle is u-a-b-v-u.
                    # So the 4th edge is (u,v).
                    # The other 3 edges are (u,a), (a,b), (b,v).
                    # For the cycle to be rainbow, all 4 colors must be distinct.
                    # So c_new cannot be ca, cb, or cv.
                    # Also ca, cb, cv must be distinct among themselves.
                    # If ca == cb, the cycle is already non-rainbow regardless of c_new.
                    # But we assume the current graph is valid (all existing 4-cycles are rainbow).
                    # So ca, cb, cv are distinct.
                    
                    # So the logic is:
                    # If (a,b) is an edge, and a is neighbor of u, b is neighbor of v:
                    # Then u-a-b-v-u is a 4-cycle.
                    # The colors of existing edges are ca, cb, cv.
                    # The new edge (u,v) has color c.
                    # We need c != ca, c != cb, c != cv.
                    # So forbidden.add(ca), forbidden.add(cb), forbidden.add(cv).
                    
                    # This logic is CORRECT for the cycle u-a-b-v-u.
                    
                    # My previous analysis of "u-a-v-b-u" was a different cycle structure.
                    # The prompt's code correctly identifies 4-cycles formed by adding (u,v) 
                    # where the path is u-a-b-v.
                    
                    # So I will stick to this logic.
                    
                    if ca == cb or ca == cv or cb == cv:
                        # This shouldn't happen if the graph is currently valid.
                        # But if it does, we can't fix it by choosing c.
                        # So we return "impossible"
                        return set(range(num_colors))
                    
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
        
        return forbidden

    def greedy_build(seed):
        rng = random.Random(seed)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in shuffled_edges:
            forbidden = get_forbidden_colors(u, v, adj, neighbors)
            
            if len(forbidden) < num_colors:
                # Find a valid color
                for c in range(num_colors):
                    if c not in forbidden:
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        break
        
        return edges

    def local_search(edges, adj, neighbors, seed):
        rng = random.Random(seed)
        # Try to add missing edges
        # and maybe remove/add to swap
        
        missing_edges = [e for e in all_edges if adj[e[0]][e[1]] == -1]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            rng.shuffle(missing_edges)
            for (u, v) in missing_edges:
                if time.time() > deadline - 5:
                    break
                forbidden = get_forbidden_colors(u, v, adj, neighbors)
                if len(forbidden) < num_colors:
                    for c in range(num_colors):
                        if c not in forbidden:
                            adj[u][v] = c
                            adj[v][u] = c
                            neighbors[u].add(v)
                            neighbors[v].add(u)
                            edges.append([u, v, c])
                            missing_edges.remove((u, v))
                            improved = True
                            break

        return edges, adj, neighbors

    best_edges = None
    best_count = 0
    
    # Phase 1: Greedy with multiple seeds
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    # Phase 2: Local Search on the best found so far
    if best_edges is not None:
        # Reconstruct adj and neighbors from best_edges
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for e in best_edges:
            u, v, c = e
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        # Try local search with a few seeds
        for i in range(10):
            if time.time() > deadline - 2:
                break
            edges_ls, adj_ls, neighbors_ls = local_search(best_edges[:], adj, neighbors, 1000+i)
            if len(edges_ls) > best_count:
                best_count = len(edges_ls)
                best_edges = edges_ls
                adj = adj_ls
                neighbors = neighbors_ls

    if best_edges is None:
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
