import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Q3 structure: 3-cube with 8 vertices, 12 edges
    # Vertices are 3-bit numbers 0..7
    # Edge if Hamming distance 1
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))
    
    # Precompute Q3 adjacency as bitmasks for fast checking
    q3_adj_mask = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj_mask[u] |= (1 << v)
        q3_adj_mask[v] |= (1 << u)
        
    # Precompute neighbors in Q3 for each vertex
    q3_neighbors = [[] for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)

    def creates_q3(adj_mask, u, v):
        """
        Check if adding edge (u, v) to a Q3-free graph creates a Q3.
        Uses bitmask intersection for fast candidate filtering.
        """
        # Precompute common neighbors of u and v in adj_mask
        common_uv = adj_mask[u] & adj_mask[v]
        # Remove u and v themselves from consideration
        common_uv &= ~(1 << u)
        common_uv &= ~(1 << v)
        
        # For each Q3 edge (qa, qb), try both orientations
        for qa, qb in Q3_EDGES:
            for orient in [(u, v), (v, u)]:
                g_qa, g_qb = orient
                
                # The other 6 vertices of Q3 must map to vertices that are
                # adjacent to the appropriate mapped vertices.
                # 
                # In Q3, vertices adjacent to qa (excluding qb) are qa's other 2 neighbors
                # Vertices adjacent to qb (excluding qa) are qb's other 2 neighbors
                # The remaining 2 vertices are adjacent to neither qa nor qb directly
                # but are adjacent to 2 of the other vertices.
                
                # Let's use a more structured approach:
                # After fixing qa->g_qa and qb->g_qb, we need to find a bijection
                # from the remaining 6 Q3 vertices to 6 of the 14 remaining G vertices
                # such that all Q3 edges are preserved.
                
                # Key insight: In Q3, the 6 remaining vertices form a specific structure.
                # Let's identify them by their adjacency to qa and qb.
                
                # Q3 vertices: 0,1,2,3,4,5,6,7
                # Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                
                # For given qa, qb (adjacent in Q3):
                # - qa's other neighbors (not qb): 2 vertices
                # - qb's other neighbors (not qa): 2 vertices  
                # - Vertices adjacent to neither qa nor qb: 2 vertices
                #   These 2 are adjacent to each other and to one vertex from each of the above groups.
                
                qa_q3_nbs = q3_neighbors[qa]
                qb_q3_nbs = q3_neighbors[qb]
                
                # qa's neighbors excluding qb
                qa_other = [x for x in qa_q3_nbs if x != qb]
                # qb's neighbors excluding qa
                qb_other = [x for x in qb_q3_nbs if x != qa]
                
                # Remaining 2 vertices
                remaining2 = [x for x in range(8) if x != qa and x != qb and x not in qa_other and x not in qb_other]
                
                # In Q3, each vertex in qa_other is adjacent to exactly one vertex in qb_other
                # and one vertex in remaining2.
                # Each vertex in remaining2 is adjacent to one vertex in qa_other and one in qb_other.
                
                # Let's try all assignments:
                # qa_other[0] -> g1, qa_other[1] -> g2
                # qb_other[0] -> g3, qb_other[1] -> g4
                # remaining2[0] -> g5, remaining2[1] -> g6
                
                # Constraints:
                # g1 must be adjacent to g_qa (already fixed) and to g3 or g4 (whichever is adjacent to qa_other[0] in Q3)
                # and to g5 or g6 (whichever is adjacent to qa_other[0] in Q3)
                
                # Let's be more systematic. For each of the 6 remaining Q3 vertices,
                # determine which already-mapped Q3 vertices it's adjacent to.
                
                # Build the constraint structure
                # remaining_q3 = qa_other + qb_other + remaining2 (6 vertices)
                remaining_q3_list = qa_other + qb_other + remaining2
                
                # For each remaining Q3 vertex, find which of {qa, qb} it's adjacent to
                constraints = {}
                for rv in remaining_q3_list:
                    adj_to_mapped = []
                    if rv in qa_q3_nbs:
                        adj_to_mapped.append(qa)
                    if rv in qb_q3_nbs:
                        adj_to_mapped.append(qb)
                    constraints[rv] = adj_to_mapped
                
                # Also, among the remaining 6, there are edges between them.
                # Let's precompute which pairs among remaining_q3_list are adjacent in Q3.
                rem_edges = []
                for i in range(len(remaining_q3_list)):
                    for j in range(i+1, len(remaining_q3_list)):
                        a, b = remaining_q3_list[i], remaining_q3_list[j]
                        if (q3_adj_mask[a] >> b) & 1:
                            rem_edges.append((i, j))
                
                # Now do backtracking with bitmask filtering
                mapping = [-1] * 8
                mapping[qa] = g_qa
                mapping[qb] = g_qb
                
                # Candidate pool: vertices adjacent to all their mapped Q3 neighbors
                # Start with all 14 unused vertices
                used_mask = (1 << g_qa) | (1 << g_qb)
                available = common_uv  # These are common neighbors, but we need all unused vertices
                
                # Actually, let's just use all 16 vertices minus the 2 used
                all_mask = (1 << 16) - 1
                available = all_mask & ~used_mask
                
                def dfs(idx):
                    if time.time() > deadline - 0.3:
                        return False
                    if idx == len(remaining_q3_list):
                        return True
                    
                    rv = remaining_q3_list[idx]
                    # Get the G-vertices that rv must be adjacent to (mapped neighbors)
                    must_be_adj = []
                    for mq in constraints[rv]:
                        if mapping[mq] != -1:
                            must_be_adj.append(mapping[mq])
                    
                    # Also check edges to already-assigned remaining Q3 vertices
                    for (ei, ej) in rem_edges:
                        if ei == idx:
                            # rv is adjacent to remaining_q3_list[ej], which is assigned
                            if ej < idx:
                                must_be_adj.append(mapping[remaining_q3_list[ej]])
                        elif ej == idx:
                            if ei < idx:
                                must_be_adj.append(mapping[remaining_q3_list[ei]])
                    
                    # Filter available vertices that are adjacent to all must_be_adj
                    candidates = available
                    for g in must_be_adj:
                        candidates &= adj_mask[g]
                    
                    # Try each candidate
                    cands_list = []
                    cm = candidates
                    while cm:
                        lsb = cm & (-cm)
                        g_vertex = lsb.bit_length() - 1
                        cands_list.append(g_vertex)
                        cm ^= lsb
                    
                    for g_vertex in cands_list:
                        mapping[rv] = g_vertex
                        new_used = used_mask | (1 << g_vertex)
                        new_avail = available & ~(1 << g_vertex)
                        
                        if dfs(idx + 1):
                            return True
                        
                        mapping[rv] = -1
                    
                    return False
                
                if dfs(0):
                    return True
        
        return False

    def find_one_q3_fast(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        adj_mask: list of 16 integers, each a bitmask of neighbors.
        """
        # Iterate over all 8-element subsets
        all_8_subsets = list(itertools.combinations(range(n), 8))
        for S in all_8_subsets:
            S_set = set(S)
            
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                mask = adj_mask[v] & ((1 << 16) - 1)
                deg = 0
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Backtracking with bitmask pruning
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S
            
            def dfs(idx):
                nonlocal used
                if time.time() > deadline - 0.5:
                    return False  # Abort if too slow
                
                if idx == 8:
                    return True
                
                # Neighbors of idx in Q3 that are already assigned (index < idx)
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                # Try each unused vertex in S
                for v in S:
                    if (used >> v) & 1:
                        continue
                    
                    # Check if v is adjacent to all already-assigned neighbors of idx
                    valid = True
                    for nb in assigned_nbs:
                        if not (adj_mask[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                
                return False
            
            if dfs(0):
                # Extract the 12 edges of the found Q3
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return (True, edges)
        
        return (False, None)
    
    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    # Strategy: Greedy + Local Search + Simulated Annealing swaps
    
    best_m = 0
    best_adj = [0] * n
    best_edges = []
    
    # Initial solution: K_16
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    current_edges = _count_edges(adj)
    
    # Phase 1: Remove edges to eliminate Q3
    while time.time() < deadline - 15.0:
        found, edges = find_one_q3_fast(adj)
        if not found:
            break
        
        # Remove a random edge from the found Q3
        random.shuffle(edges)
        u, v = edges[0]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_edges -= 1
    
    # Phase 2: Greedy add edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                candidates.append((u, v))
    
    # Multiple passes of greedy addition
    for _pass in range(50):
        if time.time() > deadline - 10.0:
            break
        random.shuffle(candidates)
        improved = False
        for u, v in candidates:
            if time.time() > deadline - 8.0:
                break
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            if creates_q3(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                improved = True
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                    best_edges = get_edge_list(adj)
        if not improved:
            break
    
    # Phase 3: Simulated Annealing with edge swaps
    # Swap: remove one existing edge, try to add two new edges
    # This allows escaping local optima where no single edge can be added.
    
    # Build lists for fast access
    def get_existing_edges(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    def get_missing_edges(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    # SA parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    sa_iter = 0
    max_sa_iters = 20000
    
    while time.time() < deadline - 3.0 and sa_iter < max_sa_iters:
        sa_iter += 1
        T = T_start * (cooling_rate ** sa_iter)
        if T < T_end:
            T = T_end
        
        # Get current edge lists
        existing = get_existing_edges(adj)
        missing = get_missing_edges(adj)
        
        if not existing or len(missing) < 2:
            break
        
        # Choose a random existing edge to potentially remove
        rem_u, rem_v = random.choice(existing)
        
        # Choose two random missing edges to potentially add
        add1_u, add1_v = random.choice(missing)
        add2_u, add2_v = random.choice(missing)
        
        # Ensure the edges are distinct
        if (add1_u, add1_v) == (rem_u, rem_v) or (add2_u, add2_v) == (rem_u, rem_v):
            continue
        if (add1_u, add1_v) == (add2_u, add2_v):
            continue
        
        # Apply the swap: remove (rem_u, rem_v), add (add1_u, add1_v) and (add2_u, add2_v)
        # First remove
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        current_edges -= 1
        
        # Try to add first edge
        adj[add1_u] |= (1 << add1_v)
        adj[add1_v] |= (1 << add1_u)
        current_edges += 1
        
        check1 = creates_q3(adj, add1_u, add1_v)
        
        if check1:
            # Revert first add
            adj[add1_u] &= ~(1 << add1_v)
            adj[add1_v] &= ~(1 << add1_u)
            current_edges -= 1
            # Revert remove
            adj[rem_u] |= (1 << rem_v)
            adj[rem_v] |= (1 << rem_u)
            current_edges += 1
            continue
        
        # Try to add second edge
        adj[add2_u] |= (1 << add2_v)
        adj[add2_v] |= (1 << add2_u)
        current_edges += 1
        
        check2 = creates_q3(adj, add2_u, add2_v)
        
        if check2:
            # Revert second add
            adj[add2_u] &= ~(1 << add2_v)
            adj[add2_v] &= ~(1 << add2_u)
            current_edges -= 1
            # Revert first add
            adj[add1_u] &= ~(1 << add1_v)
            adj[add1_v] &= ~(1 << add1_u)
            current_edges -= 1
            # Revert remove
            adj[rem_u] |= (1 << rem_v)
            adj[rem_v] |= (1 << rem_u)
            current_edges += 1
            continue
        
        # Swap is valid. Accept or reject based on SA criterion
        # Delta = new_edges - old_edges = +1 (removed 1, added 2)
        delta = 1  # Always +1 for a valid 1-out-2-in swap
        
        if delta >= 0:
            # Always accept improving or equal moves
            if current_edges > best_m:
                best_m = current_edges
                best_adj = adj[:]
                best_edges = get_edge_list(adj)
        else:
            # Accept with probability exp(delta / T)
            if random.random() < (2.718281828 ** (delta / T)):
                pass  # Accept
            else:
                # Revert
                adj[add2_u] &= ~(1 << add2_v)
                adj[add2_v] &= ~(1 << add2_u)
                current_edges -= 1
                adj[add1_u] &= ~(1 << add1_v)
                adj[add1_v] &= ~(1 << add1_u)
                current_edges -= 1
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                current_edges += 1
    
    # Phase 4: Final greedy pass
    if time.time() < deadline - 1.0:
        missing_final = get_missing_edges(adj)
        random.shuffle(missing_final)
        for u, v in missing_final:
            if time.time() > deadline - 0.5:
                break
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            if creates_q3(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                    best_edges = get_edge_list(adj)
    
    # Final update
    if current_edges > best_m:
        found, _ = find_one_q3_fast(adj)
        if not found:
            best_m = current_edges
            best_adj = adj[:]
            best_edges = get_edge_list(adj)
            
    return best_edges
