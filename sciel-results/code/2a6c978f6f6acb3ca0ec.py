"""K4-free graphs on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    n = 18
    random.seed(42)
    full_mask = (1 << n) - 1
    
    # Precompute popcount for 18-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def is_k4_free(masks):
        """Check if graph with adjacency masks is K4-free."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (masks[a] & (1 << b)):
                    continue
                common_ab = masks[a] & masks[b]
                # For each pair (c, d) in common_ab with c < d, check if c-d is an edge
                m = common_ab
                while m:
                    c_bit = m & -m
                    c = c_bit.bit_length() - 1
                    rest = m ^ c_bit
                    if rest & masks[c]:
                        return False
                    m = rest
        return True
    
    def max_triangle_free_induced(masks):
        """
        Find the largest subset of vertices inducing a triangle-free subgraph.
        Uses branch and bound with bitmask representation.
        """
        best = [0]
        
        # Precompute: for each vertex v, which other vertices form a triangle with v
        # tri_conflict[v] = bitmask of vertices w such that v, w, and some common neighbor form a triangle
        # Actually: v and w are in conflict if they share a common neighbor (i.e., there exists x adjacent to both)
        # In a triangle-free induced subgraph, no two vertices in the set can share a common neighbor within the set.
        # So: a set S is triangle-free iff for every pair u,v in S, (masks[u] & masks[v] & S) == 0
        
        # Better: for each pair (u,v), precompute their common neighbors
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                common[u][v] = masks[u] & masks[v]
                common[v][u] = common[u][v]
        
        def can_add(S, v):
            """Check if v can be added to S while keeping S triangle-free."""
            # v can be added iff for all u in S, u and v have no common neighbor in S
            # i.e., for all u in S: common[u][v] & S == 0
            m = S
            while m:
                u_bit = m & -m
                u = u_bit.bit_length() - 1
                if common[u][v] & S:
                    return False
                m ^= u_bit
            return True
        
        def search(S, candidates):
            if time.time() > deadline - 5:
                return
            sz = popcount[S]
            if sz + popcount[candidates] <= best[0]:
                return
            if candidates == 0:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Branch and bound: pick a vertex from candidates
            # Use a heuristic: pick the vertex with most connections in candidates
            # For speed, just pick the lowest set bit
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            remaining = candidates ^ v_bit
            
            # Try including v
            if can_add(S, v):
                new_S = S | v_bit
                # Filter candidates: remove vertices that conflict with new_S
                new_cands = 0
                m = remaining
                while m:
                    w_bit = m & -m
                    w = w_bit.bit_length() - 1
                    if can_add(new_S, w):
                        new_cands |= w_bit
                    m ^= w_bit
                search(new_S, new_cands)
            
            # Try excluding v
            search(S, remaining)
        
        search(0, full_mask)
        return best[0]
    
    def edge_count(masks):
        """Count edges in the graph."""
        count = 0
        for u in range(n):
            count += popcount[masks[u]]
        return count // 2
    
    def add_edge_safe(masks, u, v):
        """Add edge (u,v) if it doesn't create a K4. Returns True if added."""
        if masks[u] & (1 << v):
            return False
        # Check if adding (u,v) creates a K4
        # A K4 containing (u,v) needs two common neighbors w,x that are adjacent
        common = masks[u] & masks[v]
        m = common
        while m:
            w_bit = m & -m
            w = w_bit.bit_length() - 1
            if masks[w] & (m ^ w_bit):
                return False  # Would create K4
            m ^= w_bit
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        return True
    
    def remove_edge(masks, u, v):
        """Remove edge (u,v)."""
        masks[u] ^= (1 << v)
        masks[v] ^= (1 << u)
    
    def random_k4_free_graph():
        """Generate a random K4-free graph by randomly adding edges."""
        masks = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            add_edge_safe(masks, u, v)
        return masks
    
    def tripartite_graph():
        """Complete tripartite graph K6,6,6."""
        masks = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        masks[a] |= (1 << b)
                        masks[b] |= (1 << a)
        return masks
    
    def local_search(masks, time_limit=30.0):
        """Local search: try adding/removing edges to improve score."""
        best_masks = masks[:]
        best_score = max_triangle_free_induced(masks)
        t_start = time.time()
        
        while time.time() < t_start + time_limit:
            improved = False
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > t_start + time_limit:
                    break
                
                if masks[u] & (1 << v):
                    # Try removing edge
                    remove_edge(masks, u, v)
                    new_score = max_triangle_free_induced(masks)
                    if new_score < best_score:
                        best_score = new_score
                        best_masks = masks[:]
                        improved = True
                    else:
                        add_edge_safe(masks, u, v)  # restore
                else:
                    # Try adding edge
                    if add_edge_safe(masks, u, v):
                        new_score = max_triangle_free_induced(masks)
                        if new_score < best_score:
                            best_score = new_score
                            best_masks = masks[:]
                            improved = True
                        else:
                            remove_edge(masks, u, v)
            
            if not improved:
                # Try a random restart of local search
                if time.time() > t_start + time_limit - 10:
                    break
                # Perturb: randomly flip a few edges
                for _ in range(3):
                    u, v = random.choice(all_edges)
                    if masks[u] & (1 << v):
                        remove_edge(masks, u, v)
                    else:
                        add_edge_safe(masks, u, v)
        
        return best_masks, best_score
    
    # Start with tripartite graph
    best_masks = tripartite_graph()
    best_score = max_triangle_free_induced(best_masks)
    
    # Try multiple random starts with local search
    attempts = 0
    while time.time() < deadline - 40:
        attempts += 1
        if attempts % 5 == 0:
            # Periodically try from a random graph
            masks = random_k4_free_graph()
        else:
            # Try from perturbed version of current best
            masks = best_masks[:]
            # Randomly flip 2-5 edges
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            for _ in range(random.randint(2, 5)):
                u, v = random.choice(all_edges)
                if masks[u] & (1 << v):
                    remove_edge(masks, u, v)
                else:
                    add_edge_safe(masks, u, v)
        
        if not is_k4_free(masks):
            continue
        
        score = max_triangle_free_induced(masks)
        if score < best_score:
            best_score = score
            best_masks = masks[:]
        
        # Local search
        if time.time() < deadline - 40:
            masks, score = local_search(masks, time_limit=15.0)
            if score < best_score:
                best_score = score
                best_masks = masks[:]
    
    # Final local search on best
    if time.time() < deadline - 20:
        best_masks, best_score = local_search(best_masks, time_limit=20.0)
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_masks[u] & (1 << v):
                result.append([u, v])
    return result
