import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = 0
    best_adj = None
    
    # Precompute 4-sets
    subsets4 = []
    mask_to_verts = {}
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    subsets4.append(mask)
                    mask_to_verts[mask] = [i, j, k, l]
                    
    def has_k44(adj):
        # adj is a list of 16 ints
        for mask in subsets4:
            verts = mask_to_verts[mask]
            common = (1 << 16) - 1
            for v in verts:
                common &= adj[v]
            common &= ~mask
            if bin(common).count('1') >= 4:
                return True, mask, common
        return False, None, None

    def get_edge_count(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2

    for restart in range(200):
        if time.time() > deadline:
            break
            
        adj = [0] * 16
        # Start with p=0.8 to get dense graph
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < 0.8:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        # Phase 1: Remove edges to break K44s
        changed = True
        while changed and time.time() < deadline:
            changed = False
            has, mask, common = has_k44(adj)
            if has:
                common_verts = [i for i in range(16) if common & (1 << i)]
                mask_verts = mask_to_verts[mask]
                
                # Pick a random edge between mask and common
                # To be more effective, pick an edge that is in many K44s?
                # For now, random.
                u = random.choice(mask_verts)
                v = random.choice(common_verts)
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                changed = True
                
        # Phase 2: Add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to avoid bias
            edges_to_try = [(u, v) for u in range(16) for v in range(u+1, 16) if not (adj[u] & (1 << v))]
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                if adj[u] & (1 << v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                has, _, _ = has_k44(adj)
                if has:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    improved = True
                    
        curr_edges = get_edge_count(adj)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj.copy()
            
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges
