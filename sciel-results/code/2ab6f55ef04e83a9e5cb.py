import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    # Fast K4 check using bitmasks
    def creates_k4(adj, u, v):
        common = adj[u] & adj[v]
        m = common
        while m:
            lb = m & (-m)
            x = lb.bit_length() - 1
            m ^= lb
            if adj[x] & common:
                return True
        return False
    
    # Evaluate largest triangle-free induced subgraph using B&B
    def eval_max_tf(adj):
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    cm = common & (~((1 << (v + 1)) - 1))
                    while cm:
                        lb = cm & (-cm)
                        w = lb.bit_length() - 1
                        cm ^= lb
                        triangles.append((u, v, w))
        
        # tri_partners[v] = bitmask of vertices w such that v,w are in some triangle
        tri_partners = [0] * n
        for (u, v, w) in triangles:
            tri_partners[u] |= (1 << v) | (1 << w)
            tri_partners[v] |= (1 << u) | (1 << w)
            tri_partners[w] |= (1 << u) | (1 << v)
        
        def can_add(mask, v):
            nm = mask & adj[v]
            rm = nm
            while rm:
                lb = rm & (-rm)
                u = lb.bit_length() - 1
                rm ^= lb
                if adj[u] & mask & adj[v] & ~(1 << u):
                    return False
            return True
        
        best = 0
        verts = list(range(n))
        verts.sort(key=lambda v: adj[v].bit_count(), reverse=True)
        
        def bb(pos, current_mask, current_size):
            nonlocal best
            if pos >= n:
                if current_size > best:
                    best = current_size
                return
            if current_size + (n - pos) <= best:
                return
            v = verts[pos]
            if can_add(current_mask, v):
                bb(pos + 1, current_mask | (1 << v), current_size + 1)
            bb(pos + 1, current_mask, current_size)
        
        bb(0, 0, 0)
        return best
    
    best_score = n
    best_edges = []
    
    # Start with 3-partite structure: 3 groups of 6
    # Within each group: clique (all edges)
    # Between groups: initially complete bipartite, then we'll selectively remove edges
    # to maintain K4-free while breaking up large triangle-free sets
    
    # Actually, let's use a different approach:
    # Start with complete 3-partite T_3(18) (no within-part edges, all between-part edges)
    # This is K4-free and has max TF = 12
    # Then add edges within parts carefully to reduce max TF while staying K4-free
    
    # Better: use random K4-free graphs with structure bias
    # Generate random K4-free graphs using greedy with random edge order
    
    for restart in range(200):
        if time.time() > deadline - 20:
            break
        
        adj = [0] * n
        edges = []
        
        # Random edge order
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        score = eval_max_tf(adj)
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
        
        # Local search: try edge swaps to improve
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            # Try removing edges
            for ei in range(len(edges) - 1, -1, -1):
                if time.time() > deadline - 25:
                    break
                u, v = edges[ei]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(ei)
                
                new_score = eval_max_tf(adj)
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_edges = edges.copy()
                    improved = True
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.insert(ei, [u, v])
            
            # Try adding edges that don't create K4
            all_edges2 = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges2)
            for u, v in all_edges2:
                if time.time() > deadline - 25:
                    break
                if adj[u] & (1 << v):
                    continue
                if not creates_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    new_score = eval_max_tf(adj)
                    if new_score < score:
                        score = new_score
                        if score < best_score:
                            best_score = score
                            best_edges = edges.copy()
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.pop()
    
    return best_edges
