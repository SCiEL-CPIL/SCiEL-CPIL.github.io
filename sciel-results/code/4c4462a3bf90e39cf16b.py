import random
import time
import math
import sys

def construct():
    n = 16
    # Start with K_8,8 + edge (0,1)
    # Vertices 0-7 in A, 8-15 in B
    adj = [[False] * n for _ in range(n)]
    
    # Create K_8,8
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = True
            adj[v][u] = True
            
    # Add edge (0,1)
    adj[0][1] = True
    adj[1][0] = True
    
    # Precompute degrees
    deg = [sum(adj[i]) for i in range(n)]
    
    # Helper to get edges
    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append((u, v))
        return edges
    
    # Helper to get non-edges
    def get_non_edges():
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
        return non_edges
    
    # Calculate score
    def calc_score():
        max_sum = 0
        # Find all triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    current_score = calc_score()
    best_score = current_score
    best_adj = [row[:] for row in adj]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    T = 5.0  # Temperature
    cooling = 0.999
    
    edges = get_edges()
    non_edges = get_non_edges()
    
    while time.time() < deadline:
        # Propose a 2-switch
        # Pick two random edges (u,v) and (x,y)
        # Try to swap to (u,x) and (v,y) or (u,y) and (v,x)
        
        # To speed up, pick a random edge (u,v) and a random non-edge (x,y) is NOT a 2-switch.
        # A 2-switch removes two edges and adds two non-edges.
        # We need to pick two edges (u,v), (x,y) and two non-edges (u,x), (v,y) [or u,y v,x].
        
        # Strategy: Pick random u, v (edge). Pick random x, y (edge).
        # Check if (u,x) and (v,y) are non-edges and distinct from existing.
        
        # Simpler: Pick random u, v (edge).
        # Pick random x from non-neighbors of u.
        # Pick random y from non-neighbors of v.
        # Check if (x,y) is an edge.
        # If yes, we can remove (u,v) and (x,y) and add (u,x) and (v,y).
        
        u, v = random.choice(edges)
        
        # Get non-neighbors of u
        non_neigh_u = [i for i in range(n) if not adj[u][i] and i != u]
        if not non_neigh_u:
            continue
        x = random.choice(non_neigh_u)
        
        # Get non-neighbors of v
        non_neigh_v = [i for i in range(n) if not adj[v][i] and i != v]
        if not non_neigh_v:
            continue
        y = random.choice(non_neigh_v)
        
        # Check if (x,y) is an edge
        if not adj[x][y]:
            continue
        
        # Check if we are creating duplicate edges or self loops
        # (u,x) and (v,y) must be non-edges (ensured by selection)
        # (u,x) != (v,y) and (u,x) != (y,v) etc.
        # Since u,v,x,y are distinct?
        # u,v are edge. x is non-neighbor of u. y is non-neighbor of v.
        # If x == v, adj[u][v] is true, so x cannot be v.
        # If y == u, adj[v][u] is true, so y cannot be u.
        # If x == y, then (x,y) is self-loop? No, adj[x][y] check fails if x==y?
        # adj[x][x] is False. So if x==y, adj[x][y] is False. So x != y.
        # Are u,v,x,y distinct?
        # u != v (edge).
        # x != u (non-neighbor).
        # y != v (non-neighbor).
        # x != v? If x==v, adj[u][v] is true, so x not in non_neigh_u. So x != v.
        # y != u? If y==u, adj[v][u] is true, so y not in non_neigh_v. So y != u.
        # x != y? Checked adj[x][y]. If x==y, adj[x][x]=False. So x!=y.
        
        # Perform swap
        adj[u][v] = False
        adj[v][u] = False
        adj[x][y] = False
        adj[y][x] = False
        
        adj[u][x] = True
        adj[x][u] = True
        adj[v][y] = True
        adj[y][v] = True
        
        # Update degrees
        deg[u] -= 1
        deg[v] -= 1
        deg[x] -= 1
        deg[y] -= 1
        
        deg[u] += 1
        deg[x] += 1
        deg[v] += 1
        deg[y] += 1
        # Net change is 0 for all degrees! 2-switches preserve degree sequence.
        
        # Update edge lists
        # Remove (u,v) and (x,y) from edges
        # Add (u,x) and (v,y) to edges
        # This is O(N) or O(E). E=65. Small.
        # But doing it every step might be slow.
        # Alternative: Don't maintain lists, just sample from range?
        # No, sampling from range(n)^2 and checking adj is O(1).
        # Let's restart the sampling strategy to be O(1) per attempt.
        
        # Actually, let's just recalc score.
        new_score = calc_score()
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
        else:
            # Reject: Undo swap
            adj[u][v] = True
            adj[v][u] = True
            adj[x][y] = True
            adj[y][x] = True
            
            adj[u][x] = False
            adj[x][u] = False
            adj[v][y] = False
            adj[y][v] = False
            
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Return best result
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                best_edges.append([u, v])
    
    return {'n': n, 'edges': best_edges}
