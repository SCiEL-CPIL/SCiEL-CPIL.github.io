import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 15
    # Graph representation: list of sets for adjacency, and a set of tuples for edges
    adj = [set() for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if (u, v) in edges or (v, u) in edges: return
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    def remove_edge(u, v):
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
        elif (v, u) in edges:
            edges.remove((v, u))
            adj[v].remove(u)
            adj[u].remove(v)

    def is_valid():
        # Check all paths of length 3
        # s-u-v-t
        # We can iterate over middle edge (u,v)
        for (u, v) in list(edges):
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u or t == s: continue
                    if len(adj[s]) == len(adj[t]):
                        return False
        return True

    def try_add_edge(u, v):
        # Optimized check for adding (u,v)
        # 1. Check if (u,v) already exists
        if (u, v) in edges or (v, u) in edges:
            return False
        
        # Temporarily add
        adj[u].add(v)
        adj[v].add(u)
        deg_u_new = len(adj[u])
        deg_v_new = len(adj[v])
        
        # 2. Check new paths created by (u,v)
        # Case A: (u,v) is middle edge. s-u-v-t
        # s in N(u) \ {v}, t in N(v) \ {u}
        # Note: N(u) now includes v, N(v) now includes u.
        for s in adj[u]:
            if s == v: continue
            for t in adj[v]:
                if t == u or t == s: continue
                if len(adj[s]) == len(adj[t]):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    return False
        
        # Case B: (u,v) is first edge. u-v-w-t
        # v in N(u) (yes), w in N(v) \ {u}, t in N(w) \ {v}
        # Endpoints u, t.
        for w in adj[v]:
            if w == u: continue
            for t in adj[w]:
                if t == v or t == u: continue
                # Check degrees of u and t
                if deg_u_new == len(adj[t]):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    return False
                    
        # Case C: (u,v) is last edge. s-w-u-v
        # u in N(v) (yes), w in N(u) \ {v}, s in N(w) \ {u}
        # Endpoints s, v.
        for w in adj[u]:
            if w == v: continue
            for s in adj[w]:
                if s == u or s == v: continue
                # Check degrees of s and v
                if len(adj[s]) == deg_v_new:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    return False
        
        # 3. Check if existing paths become invalid due to degree change of u or v
        # Only paths with endpoint u or v are affected.
        # Existing paths:
        # a) s-...-u where u is endpoint.
        #    Path s-x-y-u. s in N(x), x in N(y), y in N(u).
        #    We need to check if deg(s) == deg(u_new) and previously deg(s) != deg(u_old).
        #    Actually, if deg(s) == deg(u_new), it's invalid now.
        #    It doesn't matter if it was valid before, it's invalid NOW.
        #    So we just check all paths ending in u or starting from u, etc.
        
        # Let's just re-verify all paths involving u or v as endpoints.
        # This is safer than trying to diff.
        # Paths ending in u: s-x-y-u. y in N(u), x in N(y), s in N(x).
        for y in adj[u]:
            for x in adj[y]:
                if x == u: continue
                for s in adj[x]:
                    if s == y or s == u: continue
                    if len(adj[s]) == deg_u_new:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        return False
                        
        # Paths starting from u: u-x-y-t. x in N(u), y in N(x), t in N(y).
        for x in adj[u]:
            for y in adj[x]:
                if y == u: continue
                for t in adj[y]:
                    if t == x or t == u: continue
                    if deg_u_new == len(adj[t]):
                        adj[u].remove(v)
                        adj[v].remove(u)
                        return False
                        
        # Paths ending in v: s-x-y-v.
        for y in adj[v]:
            for x in adj[y]:
                if x == v: continue
                for s in adj[x]:
                    if s == y or s == v: continue
                    if len(adj[s]) == deg_v_new:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        return False
                        
        # Paths starting from v: v-x-y-t.
        for x in adj[v]:
            for y in adj[x]:
                if y == v: continue
                for t in adj[y]:
                    if t == x or t == v: continue
                    if deg_v_new == len(adj[t]):
                        adj[u].remove(v)
                        adj[v].remove(u)
                        return False
                        
        # If we get here, it's valid.
        edges.add((u, v))
        return True

    # Initial Graph: Star centered at 0
    for i in range(1, n):
        add_edge(0, i)
        
    best_edges = len(edges)
    best_graph = set(edges)
    
    while time.time() < deadline:
        # Try to add random missing edges
        improved = False
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges and (j,i) not in edges]
        random.shuffle(missing)
        
        for u, v in missing:
            if try_add_edge(u, v):
                improved = True
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_graph = set(edges)
        
        if not improved:
            # Local optimum reached. Try a kick.
            # Remove a random edge to allow restructuring
            if edges:
                e = random.choice(list(edges))
                remove_edge(e[0], e[1])
            else:
                break

    # Return the best graph found
    # Convert to list of lists of ints
    result = []
    for u, v in best_graph:
        result.append([int(u), int(v)])
        
    return result
