import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    N = 16
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    sys.setrecursionlimit(5000)

    # Precompute: for each edge (u,v), which other edges become forbidden if we add (u,v)?
    # An edge (u,v) forbids (u,w) for all w in N(v) and (v,w) for all w in N(u)
    # We'll compute this on the fly but cache the neighbor lists

    # 3-colorability check with greedy pre-filter
    def is_3_colorable(adj_mask):
        # Fast greedy check first
        order = list(range(N))
        order.sort(key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        # Try greedy coloring
        colors = [-1] * N
        for v in order:
            used = 0
            m = adj_mask[v]
            for u in range(N):
                if m & (1 << u) and colors[u] != -1:
                    used |= (1 << colors[u])
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    break
            if colors[v] == -1:
                return False  # Not 3-colorable
        
        # Verify with backtracking (greedy might fail on some graphs)
        # Actually, if greedy succeeds, it's definitely 3-colorable
        # If greedy fails, we need backtracking to confirm
        pos = {v: i for i, v in enumerate(order)}
        later_neighbors = [[] for _ in range(N)]
        for idx in range(N):
            v = order[idx]
            m = adj_mask[v]
            for j in range(idx + 1, N):
                u = order[j]
                if m & (1 << u):
                    later_neighbors[idx].append(j)
        
        forb = [0] * N

        def backtrack(idx):
            if idx == N:
                return True
            avail = 7 & ~forb[idx]
            if avail == 0:
                return False
            c = avail
            while c:
                bit = c & (-c)
                c -= bit
                col = bit.bit_length() - 1
                touched = []
                for j in later_neighbors[idx]:
                    old = forb[j]
                    new = old | bit
                    forb[j] = new
                    touched.append((j, old))
                if backtrack(idx + 1):
                    return True
                for j, old in touched:
                    forb[j] = old
            return False

        return backtrack(0)

    # Local improvement: try to add more edges to a 4-chromatic triangle-free graph
    def try_add_edges(adj_mask, edge_list):
        """Try to add more edges while maintaining triangle-free and 4-chromatic properties."""
        improved = True
        while improved:
            improved = False
            # Try each non-edge
            for u in range(N):
                for v in range(u + 1, N):
                    if adj_mask[u] & (1 << v):
                        continue  # Already an edge
                    # Check if adding (u,v) creates a triangle
                    if adj_mask[u] & adj_mask[v]:
                        continue
                    # Tentatively add
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edge_list.append([u, v])
                    
                    # Check if still 4-chromatic
                    if not is_3_colorable(adj_mask):
                        improved = True
                    else:
                        # Revert
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        edge_list.pop()
        return edge_list

    best_edges = []
    best_count = 0

    # Branch and bound search
    def search_branch(idx, adj_mask, forbidden_mask, current_edges, edge_count):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if idx == len(all_pairs):
            if not is_3_colorable(adj_mask):
                if edge_count > best_count:
                    best_count = edge_count
                    best_edges = [list(e) for e in current_edges]
            return
            
        # Pruning: max possible edges
        if edge_count + (len(all_pairs) - idx) <= best_count:
            return
            
        u, v = all_pairs[idx]
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Check if (u,v) is forbidden
        if forbidden_mask[u] & bit_v:
            search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)
            return
            
        # Check if adding (u,v) creates a triangle
        if adj_mask[u] & adj_mask[v]:
            search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)
            return
            
        # Branch 1: Add (u,v)
        old_adj_u = adj_mask[u]
        old_adj_v = adj_mask[v]
        
        new_forb = forbidden_mask[:]
        
        # Add edge
        adj_mask[u] |= bit_v
        adj_mask[v] |= bit_u
        current_edges.append((u, v))
        
        # Update forbidden mask
        new_forb[v] |= adj_mask[u]
        new_forb[u] |= adj_mask[v]
        
        search_branch(idx + 1, adj_mask, new_forb, current_edges, edge_count + 1)
        
        # Undo
        adj_mask[u] = old_adj_u
        adj_mask[v] = old_adj_v
        current_edges.pop()
        
        # Branch 2: Skip (u,v)
        search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)

    # Run multiple searches with different orderings
    iterations = 0
    while time.time() < deadline - 10.0:
        iterations += 1
        random.shuffle(all_pairs)
        
        adj_mask = [0] * N
        forbidden_mask = [0] * N
        current_edges = []
        
        search_branch(0, adj_mask, forbidden_mask, current_edges, 0)
        
        if best_count >= 64:
            break

    # Local improvement on the best solution found
    if best_edges and time.time() < deadline - 5.0:
        adj_mask = [0] * N
        for u, v in best_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        improved_edges = try_add_edges(adj_mask, [list(e) for e in best_edges])
        if len(improved_edges) > best_count:
            best_edges = improved_edges
            best_count = len(improved_edges)

    return best_edges
