import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))
    
    def bit(i):
        return 1 << i

    def popcount(x):
        return x.bit_count()

    def is_k33_free(adj_mask):
        for u, v, w in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        for a, b, c in triples:
            # Optimization: skip if neither u nor v is in the triple
            if u not in (a, b, c) and v not in (a, b, c):
                continue
                
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        return True

    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & bit(j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= bit(j)
                            adj_mask[j] |= bit(i)
                            changed = True
        return adj_mask

    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            # Find a violating triple
            viol = None
            for u, v, w in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
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
                if common_list:
                    c = common_list[0]
                    a, b = u, c
                    adj_mask[a] &= ~bit(b)
                    adj_mask[b] &= ~bit(a)

    def local_search(adj_mask, max_iters=500):
        # Phase 1: Repair
        repair(adj_mask)
        
        # Phase 2: Greedy Add
        adj_mask = greedy_add(adj_mask)
        
        # Phase 3: Local Search (Hill Climbing with Swaps)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & bit(j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no direct addition works, try a swap
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & bit(j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try a few swaps
            for i in range(min(20, len(all_edges), len(missing))):
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
                # Try removing a random edge to see if it helps (escape local optimum)
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
                    # We keep this state regardless of edge count change 
                    # because it might be a stepping stone.
                    # However, if edge count drops significantly, it's risky.
                    # But greedy_add usually recovers.

        return adj_mask

    best_adj = None
    best_edges = 0
    rng = random.Random(42)

    # Multiple restarts
    for restart in range(100):
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
            
    if best_adj is None:
        # Fallback: Empty graph
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
