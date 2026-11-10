import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))
    
    # Precompute bit masks for vertices
    def bit(i):
        return 1 << i

    def popcount(x):
        return bin(x).count('1')

    def is_k33_free(adj_mask):
        for u, v, w in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) >= 3:
                return False
        return True

    def edge_count(adj_mask):
        return sum(popcount(m) for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges

    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & bit(j)):
                        # Try adding
                        adj_mask[i] |= bit(j)
                        adj_mask[j] |= bit(i)
                        if is_k33_free(adj_mask):
                            changed = True
                        else:
                            adj_mask[i] &= ~bit(j)
                            adj_mask[j] &= ~bit(i)
        return adj_mask

    def local_search(adj_mask, max_iters=500):
        """
        1. Repair: Remove edges to fix violations.
        2. Greedy Add.
        3. Random Walk: Try swapping edges or adding edges with removal.
        """
        # Phase 1: Repair
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            # Find a violating triple
            viol = None
            for u, v, w in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) >= 3:
                    viol = (u, v, w, common)
                    break
            if viol is None:
                break
            u, v, w, common = viol
            # Remove a random edge incident to a common neighbor
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                # Try removing (u,c), (v,c), or (w,c)
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if is_k33_free(adj_mask):
                            removed = True
                            break
                        else:
                            # Restore
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                # Force remove one
                c = common_list[0]
                a, b = u, c
                adj_mask[a] &= ~bit(b)
                adj_mask[b] &= ~bit(a)

        # Phase 2: Greedy Add
        adj_mask = greedy_add(adj_mask)

        # Phase 3: Local Search (Hill Climbing)
        # Try to add missing edges. If it fails, try to swap.
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Get missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & bit(j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                # Try adding
                adj_mask[u] |= bit(v)
                adj_mask[v] |= bit(u)
                if is_k33_free(adj_mask):
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[u] &= ~bit(v)
                    adj_mask[v] &= ~bit(u)
            
            if improved:
                continue
            
            # If no direct addition works, try a swap:
            # Remove one edge, add one edge.
            # This maintains edge count but might allow future additions.
            # Or, try to remove 1 edge and add 1 edge such that the new state is "better" in some heuristic?
            # Since we want max edges, we only care about count.
            # Swaps don't change count. But they might break a "deadlock" where no single edge can be added.
            
            # Try random swap
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & bit(j)]
            if not all_edges:
                break
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            for i in range(min(10, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove r_u-r_v
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                # Add a_u-a_v
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if is_k33_free(adj_mask):
                    # Check if this swap allows adding more edges
                    # Just accept it for now to explore
                    # Then do a quick greedy add
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not improved:
                # Try removing a random edge to see if it helps
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
                    # If edge count didn't increase or decrease (should be same or +k), keep it?
                    # If it decreases, revert? No, we want to explore.
                    # But if we lose an edge and don't gain, we are worse.
                    # So only keep if edge_count is higher?
                    # Actually, greedy_add might add multiple edges.
                    # Let's just keep the state if it's valid.
                    # The loop will continue to try to add more.

        return adj_mask

    best_adj = None
    best_edges = 0
    rng = random.Random(42)

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        density = 0.4 + rng.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and Optimize
        adj = local_search(adj, max_iters=200)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # If we reach a high number, we might stop early?
        # Max possible is 190.
        # If we get > 80, it's good.
        
    if best_adj is None:
        # Fallback: Empty graph
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
