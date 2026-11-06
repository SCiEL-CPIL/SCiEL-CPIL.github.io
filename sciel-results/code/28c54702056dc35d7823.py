import random
import itertools

def construct():
    n = 18
    
    def build_adj_mask(edges):
        adj = [[False]*n for _ in range(n)]
        adj_mask = [0]*n
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj, adj_mask

    def can_add_edge(u, v, adj_mask):
        # Check if adding (u,v) creates a K4
        # K4 exists if there is an edge in N(u) & N(v)
        common = adj_mask[u] & adj_mask[v]
        # Check if common has any edge
        # i.e., is there x in common such that adj_mask[x] & common != 0?
        # Actually, if x, y in common and adj[x][y], then u,v,x,y is K4.
        # So we just need to check if the induced subgraph on 'common' has any edge.
        temp = common
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            temp ^= lsb
            if adj_mask[x] & common:
                return False
        return True

    def find_max_tf_set(adj, adj_mask):
        # Returns the set of vertices of the largest triangle-free induced subgraph
        # Iterate sizes from n down to 1
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if mask is triangle-free
                is_tf = True
                # Extract vertices
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                k = len(verts)
                for i in range(k):
                    u = verts[i]
                    for j in range(i+1, k):
                        v = verts[j]
                        if adj[u][v]:
                            # Check if there is w in verts connected to both u and v
                            # w must be in verts
                            # We can check: (adj_mask[u] & adj_mask[v] & mask)
                            # If this is not empty, does it imply a triangle?
                            # No, it implies there is a common neighbor.
                            # If there is a common neighbor w, and w is in mask, then u-v-w is a triangle.
                            common_w = adj_mask[u] & adj_mask[v] & mask
                            if common_w:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                if is_tf:
                    return list(subset)
        return []

    def evaluate(edges):
        adj, adj_mask = build_adj_mask(edges)
        # Check K4
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    if not can_add_edge(u, v, adj_mask):
                        # This check is for ADDING. 
                        # For EXISTING graph, if it's K4-free, this logic is slightly different.
                        # Actually, if the graph is K4-free, then for any edge (u,v), N(u)&N(v) must be an independent set.
                        # So can_add_edge(u,v) returning False means N(u)&N(v) has an edge, which means K4 exists.
                        # So if we call this on an existing edge, and it returns False, the graph is NOT K4-free.
                        # But we start with K4-free and only add edges if can_add_edge is True.
                        # So we can skip this check if we maintain invariant.
                        pass
        
        max_tf_set = find_max_tf_set(adj, adj_mask)
        return len(max_tf_set), max_tf_set

    # Initialize with T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
    
    current_edges = edges[:]
    current_score, current_set = evaluate(current_edges)
    best_edges = current_edges[:]
    best_score = current_score
    
    random.seed(42)
    
    for _ in range(1000):
        adj, adj_mask = build_adj_mask(current_edges)
        # Re-evaluate to get fresh set
        current_score, current_set = evaluate(current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = current_edges[:]
            if best_score <= 6: # Theoretical lower bound is likely around 6
                break
                
        if not current_set:
            continue
            
        # Try to add an edge inside current_set
        # Candidates: pairs (u,v) in current_set not connected
        candidates = []
        for i in range(len(current_set)):
            u = current_set[i]
            for j in range(i+1, len(current_set)):
                v = current_set[j]
                if not adj[u][v]:
                    candidates.append((u, v))
                    
        if not candidates:
            # Try swapping? Or just break
            continue
            
        # Pick a random candidate
        random.shuffle(candidates)
        moved = False
        for u, v in candidates:
            if can_add_edge(u, v, adj_mask):
                # Check if this edge creates a triangle in current_set
                # i.e., is there w in current_set connected to both u and v?
                common_w = adj_mask[u] & adj_mask[v]
                # Filter common_w to be in current_set
                set_mask = 0
                for w in current_set:
                    set_mask |= (1 << w)
                if common_w & set_mask:
                    # This edge breaks the set
                    # Add it
                    current_edges.append([u, v])
                    adj[u][v] = True
                    adj[v][u] = True
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    moved = True
                    break
            # If no edge breaks the set, we might still add one to change structure
            # But adding an edge that doesn't create a triangle in S doesn't reduce |S| immediately.
            # It might reduce it later.
            # Let's allow adding any valid edge in S with some probability
            else:
                if random.random() < 0.1:
                    current_edges.append([u, v])
                    adj[u][v] = True
                    adj[v][u] = True
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    moved = True
                    break
                    
        if not moved:
            # Try to remove an edge? 
            # Removing edges makes TF sets larger usually.
            # So we should avoid removing edges unless necessary.
            # Maybe swap?
            # For simplicity, skip
            continue
            
        # Re-evaluate
        new_score, new_set = evaluate(current_edges)
        
        # Acceptance criterion
        # If new_score < current_score, accept.
        # If new_score == current_score, accept with probability 0.5 (to explore)
        # If new_score > current_score, accept with small probability (annealing)
        
        if new_score < current_score:
            current_score = new_score
            current_set = new_set
        elif new_score == current_score:
            if random.random() < 0.5:
                current_score = new_score
                current_set = new_set
            else:
                # Revert
                # We added [u,v]
                current_edges.remove([u, v])
                # Rebuild adj/adj_mask
                adj, adj_mask = build_adj_mask(current_edges)
        else:
            # Revert
            current_edges.remove([u, v])
            adj, adj_mask = build_adj_mask(current_edges)
            
    return best_edges
