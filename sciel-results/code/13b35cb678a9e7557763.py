import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Q3 structure
    q3_adj = [0] * 8
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_adj[i] |= (1 << j)
                q3_adj[j] |= (1 << i)
                q3_edges.append((i, j))
                
    # Initial Graph
    adj_mask = [0] * n
    edges = []
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
    # A = {0, 1, 2}, B = {3..15}
    # Add K3,13
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
            
    # Add Triangle in A
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(0, 2)
    
    # Add Matching in B
    for i in range(3, 15, 2):
        add_edge(i, i+1)
        
    current_edge_count = len(edges)
    best_edges = list(edges)
    best_count = current_edge_count
    
    # Generate missing edges in B
    missing_edges = []
    for i in range(3, 16):
        for j in range(i+1, 16):
            if not (adj_mask[i] & (1 << j)):
                missing_edges.append((i, j))
                
    # Heuristic: Sort missing edges by some metric?
    # Maybe prioritize edges that connect high-degree vertices in B?
    # Initially all degrees in B are 1 (except 15 is 0).
    # As we add edges, degrees increase.
    # We can just iterate.
    
    def check_q3_free_with_new_edge(u, v):
        # Check if adding u-v creates a Q3.
        # u, v are in B.
        # A Q3 involving u-v must map some Q3 edge (a,b) to (u,v).
        # Since u,v are in B, and Q3 is bipartite, u and v are in different partitions of Q3.
        # The other 6 vertices of Q3 must be mapped to other graph vertices.
        # Some of those 6 might be in A, some in B.
        
        # We iterate over all Q3 edges (a,b).
        for a, b in q3_edges:
            # Map a->u, b->v
            if _can_extend_q3(u, v, a, b):
                return False
            # Map a->v, b->u
            if _can_extend_q3(v, u, a, b):
                return False
        return True

    def _can_extend_q3(gu, gv, qa, qb):
        # gu, gv are graph vertices. qa, qb are Q3 vertices.
        # qa is mapped to gu, qb to gv.
        # We need to map remaining 6 Q3 vertices to distinct graph vertices.
        # Constraints:
        # 1. If q1, q2 are adjacent in Q3, their images must be adjacent in G.
        # 2. Images must be distinct.
        # 3. If q is mapped to g, and q has neighbor qn in Q3 which is already mapped to gn, then g must be adjacent to gn.
        
        remaining_q = [x for x in range(8) if x != qa and x != qb]
        
        # Order remaining_q by degree in Q3? All have degree 3.
        # But we can order by how many neighbors are already assigned.
        # Initially, qa, qb are assigned.
        # Neighbors of qa: 3. One is qb. Two others are in remaining_q.
        # Neighbors of qb: 3. One is qa. Two others are in remaining_q.
        
        # Let's use backtracking.
        assignment = {qa: gu, qb: gv}
        used = {gu, gv}
        
        # Pre-compute neighbors in Q3 for fast lookup
        # q3_adj is bitmask
        
        def backtrack(idx):
            if idx == len(remaining_q):
                return True
            
            q = remaining_q[idx]
            
            # Candidates for q
            # Must be distinct from used
            # Must be adjacent to all assigned neighbors of q
            
            candidates = []
            for g in range(n):
                if g in used:
                    continue
                # Check adjacency to assigned neighbors
                valid = True
                # Neighbors of q in Q3
                neigh_mask = q3_adj[q]
                # Iterate through bits of neigh_mask
                # But we only care about assigned ones.
                # We can check: for each assigned qn, is g adjacent to assignment[qn]?
                for qn in assignment:
                    if (neigh_mask >> qn) & 1:
                        gn = assignment[qn]
                        if not (adj_mask[g] & (1 << gn)):
                            valid = False
                            break
                if valid:
                    candidates.append(g)
                    
            if not candidates:
                return False
                
            for g in candidates:
                used.add(g)
                assignment[q] = g
                if backtrack(idx + 1):
                    return True
                used.remove(g)
                del assignment[q]
                
            return False
            
        return backtrack(0)

    for u, v in missing_edges:
        if time.time() > deadline:
            break
            
        # Tentatively add
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        if check_q3_free_with_new_edge(u, v):
            edges.append([u, v])
            best_edges = list(edges)
            best_count = len(edges)
        else:
            # Revert
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            
    return best_edges
