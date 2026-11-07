import random
import time

def construct():
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_degrees(adj):
        deg = [0] * n
        for i in range(n):
            deg[i] = len(adj[i])
        return deg

    def is_valid(adj, deg):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    degs_u = [deg[x] for x in adj[u] if x != v]
                    degs_v = [deg[y] for y in adj[v] if y != u]
                    if set(degs_u) & set(degs_v):
                        return False
        return True

    def can_add_edge_fast(u, v, adj, deg):
        # Check 1: New edge (u,v) itself
        degs_u = [deg[x] for x in adj[u] if x != v]
        degs_v = [deg[y] for y in adj[v] if y != u]
        if set(degs_u) & set(degs_v):
            return False
            
        # Check 2: Existing edges incident to u or v might break
        # Because deg[u] and deg[v] increase by 1.
        
        # For any neighbor w of u (w != v), check edge (u, w)
        for w in adj[u]:
            if w == v: continue
            # Edge (u, w)
            # Paths s-u-w-t
            # s in N(u)\{w}. Note: v is in N(u) now.
            # t in N(w)\{u}.
            
            degs_s = set()
            for s in adj[u]:
                if s == w: continue
                if s == v:
                    degs_s.add(deg[v] + 1)
                else:
                    degs_s.add(deg[s])
            
            degs_t = set()
            for t in adj[w]:
                if t == u: continue
                if t == v:
                    degs_t.add(deg[v] + 1)
                else:
                    degs_t.add(deg[t])
            
            if degs_s & degs_t:
                return False
                
        # Similarly for neighbors of v
        for z in adj[v]:
            if z == u: continue
            # Edge (v, z)
            # Paths s-v-z-t
            # s in N(v)\{z}. u is in N(v) now.
            # t in N(z)\{v}.
            
            degs_s = set()
            for s in adj[v]:
                if s == z: continue
                if s == u:
                    degs_s.add(deg[u] + 1)
                else:
                    degs_s.add(deg[s])
            
            degs_t = set()
            for t in adj[z]:
                if t == v: continue
                if t == u:
                    degs_t.add(deg[u] + 1)
                else:
                    degs_t.add(deg[t])
            
            if degs_s & degs_t:
                return False
                
        return True

    def local_search_with_kicks(edges, time_limit):
        adj = get_adj(edges)
        deg = get_degrees(adj)
        edge_set = set(edges)
        
        while True:
            if time.time() > time_limit:
                break
            
            # Phase 1: Greedy Add
            added_any = True
            while added_any:
                added_any = False
                missing = [e for e in all_edges if e not in edge_set]
                random.shuffle(missing)
                for u, v in missing:
                    if can_add_edge_fast(u, v, adj, deg):
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        edge_set.add((u, v))
                        added_any = True
            
            if time.time() > time_limit:
                break

            # Phase 2: Kick (if stuck)
            # Remove 1 or 2 random edges to allow different configuration
            if len(edge_set) > 0:
                kick_edges = random.sample(list(edge_set), min(2, len(edge_set)))
                for u, v in kick_edges:
                    if u in adj[v]: # Safety check
                        adj[u].remove(v)
                        adj[v].remove(u)
                        deg[u] -= 1
                        deg[v] -= 1
                        edge_set.remove((u, v))
                        
                # Try to add edges again (loop continues)
            else:
                break # Can't kick empty graph

        return list(edge_set)

    best_edges = []
    best_count = 0
    start_time = time.time()
    max_time = 50
    
    # Number of restarts
    for i in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with empty graph
        current_edges = []
        
        # Local search to maximize
        current_edges = local_search_with_kicks(current_edges, max_time - (time.time() - start_time) - 0.1)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    return [list(e) for e in best_edges]
