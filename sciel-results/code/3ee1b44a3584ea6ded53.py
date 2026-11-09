import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    
    # Q3 structure: 8 vertices, 12 edges
    q3_adj = [0] * 8
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_adj[i] |= (1 << j)
                q3_adj[j] |= (1 << i)
                q3_edges.append((i, j))
    
    # Precompute Q3 neighbor lists as lists for iteration
    q3_neigh = []
    for i in range(8):
        neigh = []
        for j in range(8):
            if (q3_adj[i] >> j) & 1:
                neigh.append(j)
        q3_neigh.append(neigh)
    
    # Initial graph: K_{3,13} + triangle in A + matching in B
    adj_mask = [0] * n
    edges = []
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
    
    # A = {0, 1, 2}, B = {3..15}
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(0, 2)
    
    for i in range(3, 15, 2):
        add_edge(i, i + 1)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Generate missing edges in B
    missing_edges = []
    for i in range(3, 16):
        for j in range(i + 1, 16):
            if not (adj_mask[i] & (1 << j)):
                missing_edges.append((i, j))
    
    # Also include within-A edges (already added) - skip those
    # Sort: prioritize edges that connect vertices with lower current degree in B
    # (less likely to create Q3)
    def edge_priority(e):
        u, v = e
        du = bin(adj_mask[u]).count('1')
        dv = bin(adj_mask[v]).count('1')
        return du + dv
    
    missing_edges.sort(key=edge_priority)
    
    # Fast Q3 check: does adding edge (u,v) create a Q3?
    def check_q3_free_with_new_edge(u, v):
        # A new Q3 must include edge (u,v)
        for a, b in q3_edges:
            if _can_extend_q3(u, v, a, b):
                return False
            if _can_extend_q3(v, u, a, b):
                return False
        return True
    
    def _can_extend_q3(gu, gv, qa, qb):
        # gu maps to qa, gv maps to qb
        remaining_q = [x for x in range(8) if x != qa and x != qb]
        
        # Sort remaining by number of already-assigned neighbors (most constrained first)
        def constraint_count(q):
            count = 0
            if (q3_adj[q] >> qa) & 1:
                count += 1
            if (q3_adj[q] >> qb) & 1:
                count += 1
            return count
        
        remaining_q.sort(key=constraint_count, reverse=True)
        
        assignment = {qa: gu, qb: gv}
        used = {gu, gv}
        
        def backtrack(idx):
            if idx == len(remaining_q):
                return True
            
            q = remaining_q[idx]
            
            # Candidates: vertices not used, adjacent to all assigned neighbors of q
            neigh_mask = q3_adj[q]
            candidates = []
            
            for g in range(n):
                if g in used:
                    continue
                valid = True
                for qn in (qa, qb):
                    if (neigh_mask >> qn) & 1:
                        gn = assignment[qn]
                        if not (adj_mask[g] & (1 << gn)):
                            valid = False
                            break
                if not valid:
                    continue
                # Also check against any other assigned vertices
                for qn2 in remaining_q[:idx]:
                    if (neigh_mask >> qn2) & 1:
                        gn2 = assignment[qn2]
                        if not (adj_mask[g] & (1 << gn2)):
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
    
    # Greedy edge addition
    for u, v in missing_edges:
        if time.time() > deadline:
            break
        
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        if check_q3_free_with_new_edge(u, v):
            edges.append([u, v])
            best_edges = list(edges)
            best_count = len(edges)
        else:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
    
    # Polish pass: try edges that were rejected (graph may have changed)
    # Collect rejected edges
    rejected = []
    for u, v in missing_edges:
        if not (adj_mask[u] & (1 << v)):
            rejected.append((u, v))
    
    if rejected:
        rejected.sort(key=edge_priority)
        for u, v in rejected:
            if time.time() > deadline - 5:
                break
            
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if check_q3_free_with_new_edge(u, v):
                edges.append([u, v])
                best_edges = list(edges)
                best_count = len(edges)
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
    
    return best_edges
