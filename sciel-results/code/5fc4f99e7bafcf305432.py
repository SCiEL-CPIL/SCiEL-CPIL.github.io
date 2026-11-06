import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m_total = len(all_edges)
    
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    
    def has_q3_with_edge(adj, u, v):
        # Check if the graph (with edge u-v already added) contains a Q3 containing (u,v)
        # This is an incremental check assuming (u,v) is the "new" edge.
        # Q3 structure:
        # u is connected to v, w1, w2
        # v is connected to u, z1, z2
        # w1, w2 connected to u, y1
        # z1, z2 connected to v, y2
        # w1, w2 connected to z1, z2 (crossed) and y1
        # z1, z2 connected to w1, w2 (crossed) and y2
        # y1, y2 connected to w1, w2 (or z1, z2) and each other
        
        # More precisely, the Q3 subgraph induced by {u, v, w1, w2, z1, z2, y1, y2}
        # Edges:
        # (u,v), (u,w1), (u,w2)
        # (v,z1), (v,z2)
        # (w1,y1), (w2,y1)
        # (z1,y2), (z2,y2)
        # (w1,z1), (w2,z2)  OR  (w1,z2), (w2,z1)
        # (y1,y2)
        # Note: w1, w2 are not connected to each other. z1, z2 are not connected.
        # w1, w2 are not connected to z2, z1 (in the first case).
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
            
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # y1 is a common neighbor of w1 and w2, not u
                # In Q3, w1 and w2 are distance 2 from u.
                # They share exactly one common neighbor other than u?
                # In Q3, u=000, w1=010, w2=100. Common neighbors of 010 and 100: 110.
                # 110 is distance 2 from u.
                # So y1 = N(w1) & N(w2) & ~{u}
                # There might be multiple candidates in a general graph.
                
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                
                for y1 in y1_candidates:
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # y2 is a common neighbor of z1 and z2, not v
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                            
                            for y2 in y2_candidates:
                                # Distinctness check
                                # Vertices: u, v, w1, w2, z1, z2, y1, y2
                                # Must be all distinct.
                                # u, v distinct from others by construction (nu/nv exclude v/u)
                                # w1, w2 distinct (loop)
                                # z1, z2 distinct (loop)
                                # w1, w2 not in nv (since nu & nv = 0? No, they can be common neighbors)
                                # Wait, if w1 is in nv, then w1 is a neighbor of v.
                                # In Q3, w1 (010) is not neighbor of v (001).
                                # So we must ensure w1, w2 are not in nv_list?
                                # No, the definition of Q3 requires specific adjacencies.
                                # If w1 is neighbor of v, it might still form a Q3?
                                # Let's check the edges.
                                # If w1 ~ v, then we have an extra edge not in Q3.
                                # But the question is "contains a subgraph isomorphic to Q3".
                                # So extra edges are fine.
                                # However, the vertices must be distinct.
                                # So w1 != z1, w1 != z2, etc.
                                
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 in (z1, z2): continue
                                if w2 in (z1, z2): continue
                                
                                # Check edges
                                # 1. y1-y2
                                if not ((adj[y1] >> y2) & 1): continue
                                
                                # 2. Cross edges
                                # Case A: (w1,z1) and (w2,z2)
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                # Case B: (w1,z2) and (w2,z1)
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
                                    
        return False

    def run_search(seed):
        random.seed(seed)
        adj = [0] * n
        edge_set = set()
        
        # 1. Greedy Construction
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            # Temporarily add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not has_q3_with_edge(adj, u, v):
                edge_set.add((u, v))
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # 2. Local Search (Hill Climbing)
        # Try to add missing edges
        # Try to swap edges to escape local maxima
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any missing edge
            for u, v in all_edges:
                if (u, v) not in edge_set:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if not has_q3_with_edge(adj, u, v):
                        edge_set.add((u, v))
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            # If we couldn't add any edge, try swaps
            if not improved and len(edge_set) > 0:
                # Pick a random edge to remove
                u, v = random.choice(list(edge_set))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.remove((u, v))
                
                # Try to add any missing edge (excluding the one we just removed? No, we can re-add it, but that's a no-op. 
                # We want to add a DIFFERENT edge to change structure)
                added = False
                for u2, v2 in all_edges:
                    if (u2, v2) not in edge_set and (u2, v2) != (u, v):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        if not has_q3_with_edge(adj, u2, v2):
                            edge_set.add((u2, v2))
                            added = True
                            break
                        else:
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                
                if not added:
                    # Revert swap
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                else:
                    # We swapped. Now try to add MORE edges
                    improved = True
                    # The loop will continue and try to add more edges in the next iteration
                    # because 'improved' is True, so it will go back to the top of the while loop
                    # and run the "Try to add any missing edge" block again.
        
        return len(edge_set), list(edge_set)

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_search(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        
        # If we find a very good solution, we can stop or continue.
        # For N=16, max edges is likely around 60-70.
        if best_count >= 65:
            break

    # If no Q3-free graph found (should not happen with greedy fallback, but SA might fail),
    # we should probably run a simple greedy to ensure validity.
    if best_count == 0:
        # Fallback greedy
        random.seed(0)
        edges = all_edges[:]
        random.shuffle(edges)
        adj = [0] * n
        edge_set = set()
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3_with_edge(adj, u, v):
                edge_set.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        best_count = len(edge_set)
        best_edges = list(edge_set)

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
