import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    
    # 1. Construct Complement of Rook Graph on 4x4
    # Vertices (i, j) where i = v // 4, j = v % 4
    # Rook Edge: i == i' or j == j'
    # Complement Edge: i != i' and j != j'
    
    adj = [0] * n
    edges_rook_comp = []
    for u in range(n):
        iu = u // 4
        ju = u % 4
        for v in range(u + 1, n):
            iv = v // 4
            jv = v % 4
            if iu != iv and ju != jv:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges_rook_comp.append([u, v])
                
    # 2. Precompute Q3 Masks in K8 for verification
    # Standard Q3 edges
    standard_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    def edge_to_idx(u, v):
        if u > v: u, v = v, u
        return u * 7 - u * (u + 1) // 2 + (v - u - 1)
        
    standard_mask = 0
    for u, v in standard_q3_edges:
        standard_mask |= 1 << edge_to_idx(u, v)
        
    q3_masks = set()
    import itertools
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in standard_q3_edges:
            nu, nv = perm[u], perm[v]
            if nu > nv: nu, nv = nv, nu
            idx = edge_to_idx(nu, nv)
            mask |= 1 << idx
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    def is_q3_free(adj):
        # Check all 8-subsets
        for S in combinations(range(n), 8):
            S_list = list(S)
            # Map global vertices to local 0..7
            # We need to check if the induced subgraph on S is Q3
            local_mask = 0
            # Build mapping from global vertex to local index
            # S is sorted, so S[0] is local 0, etc.
            # But we need to check all pairs in S
            for i in range(8):
                for j in range(i+1, 8):
                    gu = S_list[i]
                    gv = S_list[j]
                    if (adj[gu] >> gv) & 1:
                        idx = edge_to_idx(i, j)
                        local_mask |= 1 << idx
            
            # Check if local_mask is a superset of any Q3 mask
            # Since we are checking induced subgraph, we need local_mask to BE a Q3 mask?
            # No, "subgraph isomorphic to Q3" means there exists a Q3 subgraph.
            # So we check if any Q3 mask is a subset of local_mask.
            for m in q3_masks_list:
                if (local_mask & m) == m:
                    return False
        return True

    # 3. Check Rook Complement
    if is_q3_free(adj):
        return edges_rook_comp
        
    # 4. If not Q3-free, try to fix it by removing edges (Local Search)
    # We want to maximize edges, so we remove the minimum number of edges to make it Q3-free.
    # Since it's likely very close to Q3-free, we can just iterate over Q3s and remove edges.
    
    # Find a Q3
    def find_q3(adj):
        for S in combinations(range(n), 8):
            S_list = list(S)
            local_mask = 0
            for i in range(8):
                for j in range(i+1, 8):
                    gu = S_list[i]
                    gv = S_list[j]
                    if (adj[gu] >> gv) & 1:
                        idx = edge_to_idx(i, j)
                        local_mask |= 1 << idx
            
            for m in q3_masks_list:
                if (local_mask & m) == m:
                    # Found Q3 in S with mask m
                    # Return the edges of this Q3 in global coords
                    q3_edges = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            idx = edge_to_idx(i, j)
                            if (m >> idx) & 1:
                                gu = S_list[i]
                                gv = S_list[j]
                                q3_edges.append((gu, gv))
                    return q3_edges
        return None

    # Greedy removal
    # We have 72 edges. We remove edges until Q3-free.
    # To keep as many as possible, we can try removing one edge from each Q3 found.
    
    current_adj = adj[:]
    current_edges = edges_rook_comp[:]
    
    # This loop might run a few times if new Q3s appear? No, removing edges doesn't create Q3s.
    # So we just need to find ALL Q3s? No, just remove one edge from each Q3 found.
    # But removing one edge might break multiple Q3s.
    
    # Simple approach:
    # While there is a Q3:
    #   Find one.
    #   Remove one edge from it (randomly or strategically).
    #   Update adj.
    
    # To be efficient, we can collect all Q3s first?
    # There might be many.
    
    # Let's just do a while loop.
    # To avoid infinite loops or excessive time, we limit iterations.
    
    max_iters = 100
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        q3 = find_q3(current_adj)
        if q3 is None:
            break
            
        # Remove one edge from q3
        # To maximize remaining edges, we should pick an edge that is in the most other Q3s?
        # Or just pick randomly?
        # Given the structure, any edge removal works.
        # Let's pick the first one.
        u, v = q3[0]
        
        # Remove edge
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        # Remove from current_edges
        # This is slow, so we'll just rebuild edges at the end if needed,
        # or maintain a set.
        
    # Rebuild edge list from current_adj
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (current_adj[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges
