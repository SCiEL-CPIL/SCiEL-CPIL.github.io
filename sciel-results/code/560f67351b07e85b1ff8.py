import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute vertex to triples mapping for faster can_add
    # vertex_to_triples[v] = list of indices of triples containing v
    vertex_to_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
    
    def bit(i):
        return 1 << i
    
    def popcount(x):
        return x.bit_count()
    
    def is_k33_free(adj_mask):
        """Check if graph is K3,3-free."""
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
        Uses precomputed vertex_to_triples for speed.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for t_idx in vertex_to_triples[u]:
            a, b, c = triples[t_idx]
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
        
        # Check triples containing v (but not u)
        for t_idx in vertex_to_triples[v]:
            a, b, c = triples[t_idx]
            if u in (a, b, c):
                continue # Already checked
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == v: ma = new_v
            if b == v: mb = new_v
            if c == v: mc = new_v
            
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
    
    def find_violation(adj_mask):
        """Find a K3,3 violation, return (u,v,w,common_neighbors_mask) or None."""
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return (u, v, w, common)
        return None
    
    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(100):
            if time.time() > deadline:
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, common = viol
            # Try to remove an edge (x, c) where x in {u,v,w} and c in common
            # Randomize to avoid getting stuck
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                for x in (u, v, w):
                    if adj_mask[x] & bit(c):
                        # Temporarily remove
                        adj_mask[x] &= ~bit(c)
                        adj_mask[c] &= ~bit(x)
                        # Check if fixed
                        if find_violation(adj_mask) is None:
                            removed = True
                            break
                        else:
                            # Revert
                            adj_mask[x] |= bit(c)
                            adj_mask[c] |= bit(x)
                if removed:
                    break
            if not removed:
                # If we couldn't find a single edge removal that fixes it, 
                # force remove one (might need multiple, but this handles simple cases)
                if common_list:
                    c = common_list[0]
                    x = random.choice([u, v, w])
                    if adj_mask[x] & bit(c):
                        adj_mask[x] &= ~bit(c)
                        adj_mask[c] &= ~bit(x)
        return adj_mask
    
    def local_search(adj_mask, max_iters=200):
        """
        Local search with greedy add and edge swaps.
        """
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Try to add any missing edge
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            random.shuffle(missing)
            added = False
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    added = True
                    break
            
            if added:
                # After adding, try to add more greedily
                adj_mask = greedy_add(adj_mask)
                continue
            
            # 2. If no single edge can be added, try a swap
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            swapped = False
            # Try a few random swaps
            for i in range(min(50, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove r_u-r_v
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                
                # Try to add a_u-a_v
                if can_add(a_u, a_v, adj_mask):
                    adj_mask[a_u] |= bit(a_v)
                    adj_mask[a_v] |= bit(a_u)
                    # Check if the whole graph is still valid (it should be if can_add passed, 
                    # but can_add only checks triples involving a_u or a_v. 
                    # Removing an edge cannot create a violation, so it's safe.)
                    # However, we should verify just in case, or trust the logic.
                    # Logic: Removing an edge reduces common neighbors, so no new K3,3.
                    # Adding an edge is checked by can_add. So it is safe.
                    
                    # After a successful swap, try to add more greedily
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
            
            if swapped:
                continue
            
            # 3. If no swap works, remove a random edge to escape local optimum
            if all_edges:
                r_u, r_v = random.choice(all_edges)
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Try multiple restarts with different densities
    # Densities around 0.4-0.6 seem to work well for n=20
    for restart in range(40):
        if time.time() > deadline - 5:
            break
        
        density = 0.4 + rng.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and optimize
        adj = local_search(adj, max_iters=200)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Early termination if we hit a very high count
        # For n=20, ex(20, K3,3) is likely around 100-120?
        # Let's not hardcode a limit, just use time.
        
    # Final polish: try to add any remaining edges to the best found
    if best_adj is not None and time.time() < deadline - 2:
        adj = list(best_adj)
        adj = greedy_add(adj)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_edges = 0
    
    edges = mask_to_edges(best_adj)
    return edges
