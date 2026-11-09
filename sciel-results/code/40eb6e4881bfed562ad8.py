import time
import random
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)  # 120 edges
    
    # Edge index mapping
    edge_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    def edge_to_idx(u, v):
        return edge_idx[(min(u,v), max(u,v))]
    
    # Precompute compatibility: two edges are compatible if they don't form a triangle
    # Edge e1=(a,b) and e2=(c,d) are incompatible if:
    #   - they share a vertex AND the third edge exists in the graph (but for pairwise, we check if adding both would create a triangle with some other edge)
    # Actually, for triangle-free: a set of edges is triangle-free iff no three vertices have all three edges present.
    # Pairwise: edges (a,b) and (c,d) are "potentially incompatible" if {a,b,c,d} could form a triangle.
    # But any two edges can coexist in a triangle-free graph (just don't add the third).
    # So the constraint is 3-way, not pairwise.
    
    # New approach: Use branch-and-bound on vertex subsets
    # For each vertex, decide which neighbors to connect to
    # Key insight: In a triangle-free graph, the neighbors of any vertex form an independent set
    
    # Precompute for each vertex v, the set of vertices that can be neighbors
    # v's neighbors must form an independent set in the remaining graph
    
    # Better: Use the fact that in a triangle-free graph, for each vertex v,
    # N(v) is an independent set. So we can model this as:
    # For each vertex v, choose an independent set S_v from V\{v} as its neighbors
    # such that for all u,v: u in S_v iff v in S_u
    
    # Even better: Maximum clique in a compatibility graph
    # Nodes: all 120 edges of K16
    # Two edges e1, e2 are adjacent in compatibility graph if they can coexist in a triangle-free graph
    # e1=(a,b), e2=(c,d) are compatible if:
    #   - They don't share a vertex, OR
    #   - They share a vertex (then they're just two edges from the same vertex, always OK)
    #   Wait, that's not right either.
    
    # Let me think differently. The triangle-free constraint means:
    # For any three vertices {u,v,w}, at most 2 of the edges (u,v), (v,w), (u,w) can be present.
    # This is a 3-uniform hypergraph constraint.
    
    # Maximum independent set in a 3-uniform hypergraph is NP-hard, but for 16 vertices we can use branch-and-bound.
    
    # Strategy: Branch on vertices. For each vertex v, decide its neighborhood N(v).
    # N(v) must be an independent set (no two vertices in N(v) are adjacent).
    # Also, for each u in N(v), v must be in N(u).
    
    # Even better: Use the following formulation
    # Let x_{uv} be 1 if edge (u,v) is present
    # Constraints: For all u,v,w: x_{uv} + x_{vw} + x_{uw} <= 2
    # Maximize: sum x_{uv}
    # Subject to: graph is not 3-colorable
    
    # For 16 vertices, we can use branch-and-bound with the following:
    # Order vertices 0,1,...,15
    # When processing vertex v, we decide which of v+1,...,15 are neighbors of v
    # The neighbors of v among {v+1,...,15} must form an independent set in the subgraph induced by {v+1,...,15}
    # (because if two neighbors of v were adjacent, we'd have a triangle)
    
    # Let me implement a clean branch-and-bound:
    
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check using backtracking with MRV heuristic and bitmasks."""
        color = [-1] * n
        
        def backtrack():
            # Find uncolored vertex with most colored neighbors (MRV)
            best_v = -1
            best_cnt = -1
            for v in range(n):
                if color[v] == -1:
                    cnt = bin(adj_mask[v]).count('1')
                    # Count only colored neighbors
                    cnt = 0
                    m = adj_mask[v]
                    while m:
                        lsb = m & (-m)
                        u = lsb.bit_length() - 1
                        if color[u] >= 0:
                            cnt += 1
                        m ^= lsb
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
            
            if best_v == -1:
                return True
            
            v = best_v
            used = 0
            m = adj_mask[v]
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if color[u] >= 0:
                    used |= (1 << color[u])
                m ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1
            return False
        
        return backtrack()
    
    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    # Branch and bound: process vertices in order
    # State: current adjacency (bitmasks), current vertex being processed, best count so far
    # Key: when processing vertex v, we decide which of v+1,...,15 are neighbors
    # The chosen neighbors must form an independent set in the "future" subgraph
    
    best_adj = [0] * n
    best_count = 0
    
    def branch_and_bound(adj_mask, v, current_count, time_limit):
        """Branch and bound for maximum triangle-free non-3-colorable graph."""
        nonlocal best_adj, best_count
        
        if time.time() > time_limit:
            return
        
        if v >= n:
            # All vertices processed
            if current_count > best_count:
                if not is_3_colorable(adj_mask):
                    best_count = current_count
                    best_adj = list(adj_mask)
            return
        
        # Upper bound: current_count + max possible edges from v to v+1,...,15
        # Max is (n-1-v) edges from v, plus edges among v+1,...,15
        # But we need to be tighter. For now, use a simple bound.
        remaining_vertices = n - 1 - v
        upper_bound = current_count + remaining_vertices * (n - 1 - v) // 2 + (n - 1 - v)
        # Actually, max edges from v to later vertices is (n-1-v)
        # Max edges among later vertices is (n-1-v)*(n-2-v)//2
        upper_bound = current_count + (n-1-v) + (n-1-v)*(n-2-v)//2
        if upper_bound <= best_count:
            return
        
        # Try all subsets of {v+1, ..., 15} as neighbors of v
        # The subset must be an independent set in the induced subgraph on {v+1,...,15}
        # (i.e., no two chosen vertices are adjacent to each other in the current partial graph)
        
        # Get the set of vertices after v
        later_mask = 0
        for u in range(v+1, n):
            later_mask |= (1 << u)
        
        # Generate all independent sets in the subgraph induced by later_mask
        # This is expensive, so use a smarter approach:
        # Try adding vertices one by one, maintaining independence
        
        # For efficiency, try a few promising subsets:
        # 1. Empty set (v has no neighbors among later vertices)
        # 2. Maximum independent set in the later subgraph
        # 3. Random subsets
        
        # First, try the empty set
        new_adj = list(adj_mask)
        branch_and_bound(new_adj, v+1, current_count, time_limit)
        
        # Now try non-empty sets
        # Use a greedy approach to find good independent sets
        later_vertices = list(range(v+1, n))
        
        # Try each single vertex
        for u in later_vertices:
            new_adj = list(adj_mask)
            add_edge(new_adj, v, u)
            branch_and_bound(new_adj, v+1, current_count + 1, time_limit)
            remove_edge(new_adj, v, u)
        
        # Try pairs that form an independent set
        for i in range(len(later_vertices)):
            for j in range(i+1, len(later_vertices)):
                u, w = later_vertices[i], later_vertices[j]
                # Check if u and w are not adjacent
                if adj_mask[u] & (1 << w):
                    continue
                new_adj = list(adj_mask)
                add_edge(new_adj, v, u)
                add_edge(new_adj, v, w)
                branch_and_bound(new_adj, v+1, current_count + 2, time_limit)
                remove_edge(new_adj, v, u)
                remove_edge(new_adj, v, w)
        
        # Try triples that form an independent set
        if len(later_vertices) >= 3:
            for i in range(len(later_vertices)):
                for j in range(i+1, len(later_vertices)):
                    for k in range(j+1, len(later_vertices)):
                        if time.time() > time_limit:
                            return
                        u, w, x = later_vertices[i], later_vertices[j], later_vertices[k]
                        # Check if u,w,x form an independent set
                        if (adj_mask[u] & (1 << w)) or (adj_mask[u] & (1 << x)) or (adj_mask[w] & (1 << x)):
                            continue
                        new_adj = list(adj_mask)
                        add_edge(new_adj, v, u)
                        add_edge(new_adj, v, w)
                        add_edge(new_adj, v, x)
                        branch_and_bound(new_adj, v+1, current_count + 3, time_limit)
                        remove_edge(new_adj, v, u)
                        remove_edge(new_adj, v, w)
                        remove_edge(new_adj, v, x)
        
        # Try larger sets using greedy independent set
        if len(later_vertices) >= 4:
            # Greedy: pick vertices that are not adjacent to each other
            greedy_set = []
            remaining = set(later_vertices)
            for u in later_vertices:
                if time.time() > time_limit:
                    return
                if u not in remaining:
                    continue
                # Check if u is not adjacent to any in greedy_set
                compatible = True
                for w in greedy_set:
                    if adj_mask[u] & (1 << w):
                        compatible = False
                        break
                if compatible:
                    greedy_set.append(u)
                    remaining.discard(u)
            
            if len(greedy_set) >= 4:
                new_adj = list(adj_mask)
                for u in greedy_set:
                    add_edge(new_adj, v, u)
                branch_and_bound(new_adj, v+1, current_count + len(greedy_set), time_limit)
                for u in greedy_set:
                    remove_edge(new_adj, v, u)
    
    # Start with the Grötzsch graph (Mycielski of C5) on 11 vertices
    adj_mask = [0] * n
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(adj_mask, i, (i+1) % 5)
    
    # Mycielski construction: add vertices 5-9 as copies of 0-4, vertex 10 as apex
    for i in range(5):
        u, v = i, (i+1) % 5
        add_edge(adj_mask, u, 5+v)
        add_edge(adj_mask, v, 5+u)
    
    for i in range(5):
        add_edge(adj_mask, 5+i, 10)
    
    initial_count = edge_count(adj_mask)
    best_count = initial_count
    best_adj = list(adj_mask)
    
    # Phase 1: Greedy edge addition
    def greedy_add(adj_mask, max_time):
        """Try to add edges greedily."""
        changed = True
        while changed and time.time() < max_time:
            changed = False
            edges = list(all_edges)
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > max_time:
                    break
                if adj_mask[u] & (1 << v):
                    continue
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(adj_mask, u, v)
                changed = True
        return adj_mask
    
    adj_mask = greedy_add(adj_mask, deadline - 60)
    current_count = edge_count(adj_mask)
    if current_count > best_count:
        best_count = current_count
        best_adj = list(adj_mask)
    
    # Phase 2: Local search with edge swaps
    def local_search(adj_mask, max_time):
        """Local search: try edge swaps (remove one, add another)."""
        current = list(adj_mask)
        current_count = edge_count(current)
        best_local = list(current)
        best_local_count = current_count
        
        no_improve = 0
        while time.time() < max_time and no_improve < 300:
            existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if current[i] & (1 << j)]
            if not existing:
                break
            
            ui, vi = random.choice(existing)
            remove_edge(current, ui, vi)
            
            added = False
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (current[i] & (1 << j)) and (i,j) != (ui, vi)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > max_time:
                    break
                common = current[u] & current[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                new_adj = list(current)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(current, u, v)
                added = True
                break
            
            if not added:
                add_edge(current, ui, vi)
                no_improve += 1
                continue
            
            new_count = edge_count(current)
            if new_count > best_local_count:
                best_local_count = new_count
                best_local = list(current)
                no_improve = 0
            else:
                no_improve += 1
        
        return best_local, best_local_count
    
    adj_mask, local_count = local_search(adj_mask, deadline - 30)
    if local_count > best_count:
        best_count = local_count
        best_adj = list(adj_mask)
    
    # Phase 3: Multi-start with perturbation
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        current = list(best_adj)
        existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if current[i] & (1 << j)]
        random.shuffle(existing)
        num_remove = min(4, len(existing))
        for edge in existing[:num_remove]:
            remove_edge(current, edge[0], edge[1])
        
        current = greedy_add(current, deadline - 20 - restart * 2)
        current_count = edge_count(current)
        
        current, ls_count = local_search(current, deadline - 15 - restart * 2)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = list(current)
    
    # Final greedy cleanup
    final_adj = list(best_adj)
    edges = list(all_edges)
    random.shuffle(edges)
    for u, v in edges:
        if time.time() > deadline - 2:
            break
        if final_adj[u] & (1 << v):
            continue
        common = final_adj[u] & final_adj[v] & ~(1 << u) & ~(1 << v)
        if common:
            continue
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if is_3_colorable(new_adj):
            continue
        add_edge(final_adj, u, v)
    
    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj
    
    # Return result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
