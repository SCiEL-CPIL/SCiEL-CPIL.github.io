"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 31
    target_edges = 91
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4s(adj):
        # adj is a list of sets of neighbors
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    common = adj[i].intersection(adj[j])
                    c = len(common)
                    total += c * (c - 1) // 2
        return total // 4

    def get_common_count(adj, u, v):
        return len(adj[u].intersection(adj[v]))

    def random_graph():
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return edges, adj

    best_c4 = float('inf')
    best_edges = None
    
    restarts = 50
    max_iters = 2000
    
    for r in range(restarts):
        edges, adj = random_graph()
        current_c4 = count_c4s(adj)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = [list(e) for e in edges]
            
        if best_c4 == 0:
            break
            
        for it in range(max_iters):
            idx_remove = random.randint(0, target_edges - 1)
            u, v = edges[idx_remove]
            
            added = False
            for _ in range(20):
                w = random.randint(0, n - 1)
                x = random.randint(0, n - 1)
                if w == x:
                    continue
                if w > x:
                    w, x = x, w
                if w in adj[x]:
                    continue
                if (w, x) == (u, v):
                    continue
                
                delta = 0
                
                # Remove (u, v)
                c_uv = get_common_count(adj, u, v)
                delta -= c_uv * (c_uv - 1) // 2
                
                for a in list(adj[u]):
                    if a == v:
                        continue
                    if v in adj[a]:
                        k = get_common_count(adj, a, u)
                        delta -= (k - 1)
                        
                for b in list(adj[v]):
                    if b == u:
                        continue
                    if u in adj[b]:
                        k = get_common_count(adj, b, v)
                        delta -= (k - 1)

                # Add (w, x)
                c_wx = get_common_count(adj, w, x)
                delta += c_wx * (c_wx - 1) // 2
                
                for c in list(adj[w]):
                    if c == x:
                        continue
                    if x in adj[c]:
                        k = get_common_count(adj, c, w)
                        delta += k
                        
                for d in list(adj[x]):
                    if d == w:
                        continue
                    if w in adj[d]:
                        k = get_common_count(adj, d, x)
                        delta += k

                if delta < 0:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges[idx_remove] = (w, x)
                    adj[w].add(x)
                    adj[x].add(w)
                    
                    current_c4 += delta // 4
                    
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_edges = [list(e) for e in edges]
                        
                    added = True
                    break
            if not added:
                continue
                
    return best_edges

# EVOLVE-BLOCK-END
