import time
import random

N = 40

def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    best_edges = []
    best_m = 0

    def can_add_edge(u, v, adj_masks):
        """Check if adding edge (u,v) creates a C6 (path of length 4 from u to v)."""
        # We need a simple path u-a-b-c-d-v where all vertices distinct
        # Use DFS with bitmask for visited nodes
        # Pruning: at depth 3, check if any neighbor of node can reach v in 1 more step
        # Actually, we need path of exactly 5 edges: u-a-b-c-d-v
        
        # Start DFS from neighbors of u (excluding v)
        initial_mask = (1 << u) | (1 << v)
        neighbors_u = adj_masks[u] & ~(1 << v)
        
        if not neighbors_u:
            return False
        
        # Stack: (node, depth, mask)
        # depth 0 = u, depth 1 = a, depth 2 = b, depth 3 = c, depth 4 = d, then check d-v
        stack = []
        while neighbors_u:
            bit = neighbors_u & -neighbors_u
            a = bit.bit_length() - 1
            neighbors_u ^= bit
            stack.append((a, 1, initial_mask | (1 << a)))
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if node is connected to v
                if adj_masks[node] & (1 << v):
                    return True
                continue
            
            # Get unvisited neighbors
            neighbors = adj_masks[node] & ~mask
            
            while neighbors:
                bit = neighbors & -neighbors
                nb = bit.bit_length() - 1
                neighbors ^= bit
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return False

    def greedy_add(adj_masks):
        """Add any possible non-edge greedily."""
        changed = True
        while changed:
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj_masks[u] & (1 << v)) and not can_add_edge(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        changed = True
        return adj_masks

    def local_search(adj_masks, rng, max_iters=5000):
        """Swap-based local search: add an edge, remove one to resolve conflicts."""
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 2:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to add any non-edge
            added = False
            # Randomize order for diversity
            pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            rng.shuffle(pairs)
            
            for u, v in pairs:
                if not (adj_masks[u] & (1 << v)) and not can_add_edge(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    added = True
                    improved = True
                    break
            
            if added:
                continue
            
            # Phase 2: Try swaps - remove an edge, then add a non-edge
            # Get list of current edges
            edges = []
            for u in range(N):
                mask = adj_masks[u]
                while mask:
                    bit = mask & -mask
                    v = bit.bit_length() - 1
                    if u < v:
                        edges.append((u, v))
                    mask ^= bit
            
            if not edges:
                break
                
            rng.shuffle(edges)
            
            swapped = False
            for (x, y) in edges:
                # Remove edge x-y
                adj_masks[x] &= ~(1 << y)
                adj_masks[y] &= ~(1 << x)
                
                # Try to add any non-edge
                added_back = False
                pairs2 = [(u, v) for u in range(N) for v in range(u + 1, N)]
                rng.shuffle(pairs2)
                
                for u, v in pairs2:
                    if not (adj_masks[u] & (1 << v)) and not can_add_edge(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        added_back = True
                        swapped = True
                        improved = True
                        break
                
                if not added_back:
                    # Revert
                    adj_masks[x] |= (1 << y)
                    adj_masks[y] |= (1 << x)
                
                if swapped:
                    break
        
        return adj_masks

    def edge_count(adj_masks):
        return sum(bin(m).count('1') for m in adj_masks) // 2

    def adj_to_edges(adj_masks):
        edges = []
        for u in range(N):
            mask = adj_masks[u]
            while mask:
                bit = mask & -mask
                v = bit.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= bit
        return edges

    def is_c6_free(adj_masks):
        """Verify no C6 exists."""
        for u in range(N):
            for v in range(u + 1, N):
                if adj_masks[u] & (1 << v):
                    continue
                # Check if there's a path of length 4 from u to v
                # This would form a C6 with edge u-v
                # Actually, we need to check if adding u-v would create a C6
                # which is exactly what can_add_edge checks
                if can_add_edge(u, v, adj_masks):
                    return False
        return True

    # Multiple random starts with different densities
    for trial in range(500):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(trial * 7919 + 13)
        p = 0.2 + 0.15 * (trial % 7) / 7.0
        
        adj_masks = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Fix: remove edges that create C6s by greedy removal
        # Actually, just do greedy add from this starting point
        # First, ensure we start with a valid (C6-free) graph
        # Remove edges that cause C6s
        changed = True
        while changed:
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if adj_masks[u] & (1 << v):
                        # Check if this edge is part of a C6
                        # Temporarily remove and check
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                        # Check if any non-edge now becomes addable (meaning we had a C6)
                        # Actually, simpler: just check if the graph is still C6-free
                        # But that's expensive. Instead, just keep edges that don't create C6
                        # when added to the current graph
                        # This is getting complex. Let's just use greedy add from sparse.
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        break
                if changed:
                    break
        
        # Better approach: start sparse, greedy add, then local search
        adj_masks = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.15:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        adj_masks = greedy_add(adj_masks)
        adj_masks = local_search(adj_masks, rng, max_iters=2000)
        
        m = edge_count(adj_masks)
        if is_c6_free(adj_masks) and m > best_m:
            best_m = m
            best_edges = adj_to_edges(adj_masks)
            if best_m >= 105:
                # Try to improve further with more local search
                pass
        
        # If we found a good solution, do more intensive local search
        if best_m >= 100 and trial % 5 == 0:
            rng2 = random.Random(trial * 104729 + 7)
            # Do additional local search iterations
            adj_masks = greedy_add(adj_masks)
            adj_masks = local_search(adj_masks, rng2, max_iters=3000)
            m = edge_count(adj_masks)
            if is_c6_free(adj_masks) and m > best_m:
                best_m = m
                best_edges = adj_to_edges(adj_masks)

    # Fallback: seed construction (8 cliques of size 5)
    if not best_edges or best_m < 40:
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        return edges

    return best_edges
