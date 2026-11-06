import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Q3 vertices: 0..7
    # Standard labeling: binary representation
    # Edges connect vertices differing by 1 bit
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            # Check if i and j differ by exactly 1 bit
            diff = i ^ j
            if diff in [1, 2, 4]:
                q3_edges.append((i, j))
    
    # Initial graph: K3,13
    # A = {0, 1, 2}, B = {3..15}
    G = set()
    for a in range(3):
        for b in range(3, 16):
            G.add((a, b))
            G.add((b, a)) # Undirected, store both or just sorted? 
            # Let's store sorted tuples (u,v) with u<v for uniqueness
            # But for lookup, set of frozensets or sorted tuples is good.
            # Let's use set of (min, max)
    
    # Re-initialize G with sorted tuples
    G = set()
    for a in range(3):
        for b in range(3, 16):
            G.add((min(a,b), max(a,b)))
            
    best_edges = list(G)
    
    # All possible edges
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    missing_edges = [e for e in all_edges if e not in G]
    random.shuffle(missing_edges)
    
    # Helper to check if Q3 is subgraph of G on vertex set S (size 8)
    def has_q3_in_subset(S, G):
        # S is a set of 8 vertices
        # Check if there is a bijection from Q3 vertices to S such that all Q3 edges are in G
        # Simple backtracking
        # Map Q3 vertex indices 0..7 to vertices in S
        # To speed up, we can try to match vertices with high degree constraints?
        # All degrees are 3.
        
        # Precompute adjacency in G for vertices in S
        # adj[u] = set of neighbors in S
        
        s_list = list(S)
        # Mapping from q3 vertex index to actual vertex
        # We will assign q3 vertices 0..7 to s_list[0..7]
        
        # To make it faster, we can fix the mapping of one vertex?
        # No, we need to check all.
        
        # Pruning: Count edges in G[S]
        # If < 12, return False
        edge_count = 0
        for u in S:
            for v in S:
                if u < v and (u,v) in G:
                    edge_count += 1
        if edge_count < 12:
            return False
            
        # Backtracking
        # Order of Q3 vertices to assign
        # We can use a static order 0..7
        # Or dynamic based on constraints.
        
        def backtrack(assign, next_q3_vertex):
            if next_q3_vertex == 8:
                return True
            
            qv = next_q3_vertex
            # Try all unassigned vertices in S for qv
            for sv in s_list:
                if sv in assign.values():
                    continue
                
                # Check if sv can be assigned to qv
                # Check edges from qv to previously assigned q3 vertices
                ok = True
                for qv2 in range(next_q3_vertex):
                    if (qv, qv2) in q3_edges_set or (qv2, qv) in q3_edges_set:
                        # Edge must exist in G
                        sv2 = assign[qv2]
                        if (min(sv, sv2), max(sv, sv2)) not in G:
                            ok = False
                            break
                if ok:
                    assign[qv] = sv
                    if backtrack(assign, next_q3_vertex + 1):
                        return True
                    del assign[qv]
            return False
            
        return backtrack({}, 0)
        
    q3_edges_set = set(q3_edges)
    # Add reverse for quick lookup
    for u, v in q3_edges:
        q3_edges_set.add((v, u))

    for e in missing_edges:
        if time.time() > deadline:
            break
            
        u, v = e
        G.add(e)
        
        # Check if adding e created a Q3
        # Only check subsets containing u and v
        found_q3 = False
        # Other 6 vertices from the remaining 14
        other_vertices = [i for i in range(16) if i != u and i != v]
        
        # Iterate over combinations of 6 from 14
        # 3003 combinations
        # This might be slow if done in pure python for every edge.
        # 80 edges * 3003 combos * backtracking cost.
        
        # Optimization: 
        # If we just check all 8-subsets of G?
        # No, we only care about subsets containing e.
        
        # Let's optimize the subset generation
        import itertools
        for combo in itertools.combinations(other_vertices, 6):
            S = frozenset([u, v] + list(combo))
            if has_q3_in_subset(S, G):
                found_q3 = True
                break
                
        if found_q3:
            G.remove(e)
        else:
            best_edges = list(G)
            
    return [[u, v] for u, v in best_edges]
