import random
import time

def construct():
    n = 16
    A = [0]
    B = list(range(1, 16))
    
    edges = []
    # Add all edges between A and B
    for a in A:
        for b in B:
            edges.append([a, b])
            
    # Greedy C4-free graph on B
    # B has 15 vertices.
    # We want to maximize edges in G[B] such that G[B] is C4-free.
    
    best_edges_B = []
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Pre-compute all pairs in B
    pairs = []
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            pairs.append((B[i], B[j]))
            
    # Map vertex to index in B for bitset ops?
    # 15 vertices, so we can use Python integers as bitsets.
    # Let's map B vertices to 0..14
    
    b_to_idx = {v: i for i, v in enumerate(B)}
    
    def check_c4(adj, u, v):
        # adj is a list of bitsets (int) where adj[i] has bit j set if (i,j) in G
        # u, v are indices in B
        # Check if adding (u,v) creates C4
        # Need path u-w-x-v in old graph
        # w in N(u), x in N(v), (w,x) in G
        nu = adj[u]
        nv = adj[v]
        # Iterate through bits in nu
        w = nu
        while w:
            # Get lowest set bit
            lsb = w & -w
            wi = lsb.bit_length() - 1
            w ^= lsb
            # Check if (wi, x) is edge for any x in nv
            # We need to check if adj[wi] & nv != 0
            if adj[wi] & nv:
                return True
        return False

    # Greedy loop
    while time.time() < deadline:
        random.shuffle(pairs)
        adj = [0] * 15
        current_edges = []
        
        for u_val, v_val in pairs:
            u = b_to_idx[u_val]
            v = b_to_idx[v_val]
            
            if not check_c4(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append([u_val, v_val])
                
        if len(current_edges) > len(best_edges_B):
            best_edges_B = current_edges
            # If we reach a high number, we might stop early?
            # Max possible is roughly 15*14/2 = 105.
            # C4-free limit is much lower.
            
    edges.extend(best_edges_B)
    return edges
