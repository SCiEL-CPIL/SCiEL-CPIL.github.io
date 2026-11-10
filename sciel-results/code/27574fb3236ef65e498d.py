import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
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
        for _ in range(200):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            viol = None
            for u, v, w in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    viol = (u, v, w, common)
                    break
            if viol is None:
                break
            u, v, w, common = viol
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if is_k33_free(adj_mask):
                            removed = True
                            break
                        else:
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                if common_list:
                    c = common_list[0]
                    a, b = u, c
                    adj_mask[a] &= ~bit(b)
                    adj_mask[b] &= ~bit(a)

    def local_search(adj_mask, max_iters=300):
        repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
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
            
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & bit(j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            for i in range(min(30, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not improved:
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)

        return adj_mask

    best_adj = None
    best_edges = 0
    rng = random.Random(42)

    # Try different initialization strategies
    strategies = [
        lambda r: (0.3 + r.random() * 0.4, None),  # Random density
        lambda r: (0.5, 'complete_minus_random'),   # Near-complete
        lambda r: (0.0, 'empty'),                   # Empty start
        lambda r: (0.6 + r.random() * 0.2, None),  # High density
    ]
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        strat_idx = restart % len(strategies)
        density, mode = strategies[strat_idx](rng)
        
        adj = [0] * n
        if mode == 'empty':
            pass
        elif mode == 'complete_minus_random':
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() > 0.15:
                        adj[i] |= bit(j)
                        adj[j] |= bit(i)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        adj[i] |= bit(j)
                        adj[j] |= bit(i)
        
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # If we have a very good result, do extra refinement
        if best_edges >= 100 and restart % 10 == 0:
            if time.time() < deadline - 10:
                adj = list(best_adj)
                adj = local_search(adj, max_iters=200)
                cnt2 = edge_count(adj)
                if cnt2 > best_edges:
                    best_edges = cnt2
                    best_adj = list(adj)

    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
