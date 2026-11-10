import time
import numpy as np
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def can_add_mask(adj_masks, u, v):
        """
        Check if adding edge (u,v) violates K3,3.
        Optimized:
        A new K3,3 must use the edge (u,v) in the sense that u and v are in different parts.
        Case 1: Part A = {u, a, b}, Part B = {v, c, d}.
        This requires v to be a common neighbor of a and b.
        And |N(u) intersect N(a) intersect N(b)| >= 2 (since v is added to N(u), and v is in N(a), N(b), 
        but v is NOT in N(u) yet, so v is a NEW common neighbor for {u,a,b} if v in N(a) and v in N(b).
        Wait, the common neighbors of {u,a,b} are N(u) intersect N(a) intersect N(b).
        Before adding (u,v), v is not in N(u).
        After adding (u,v), v IS in N(u).
        So if v is in N(a) and v is in N(b), then v becomes a common neighbor of {u,a,b}.
        So the count of common neighbors increases by 1 if v in N(a) and v in N(b).
        We need the final count to be < 3.
        So we need: (Old Count) + (1 if v in N(a) and v in N(b) else 0) < 3.
        Old Count = bit_count(N(u) & N(a) & N(b)).
        Note: v is not in N(u) yet.
        
        So condition for violation in {u, a, b}:
        (v in N(a) and v in N(b)) and (bit_count(N(u) & N(a) & N(b)) >= 2)
        
        Similarly for {v, a, b}:
        (u in N(a) and u in N(b)) and (bit_count(N(v) & N(a) & N(b)) >= 2)
        """
        bit_v = 1 << v
        bit_u = 1 << u
        
        # Check triples {u, a, b}
        # We need v in N(a) and v in N(b)
        # Iterate a in neighbors of v
        # Iterate b in neighbors of v, b > a
        # Check bit_count(N(u) & N(a) & N(b)) >= 2
        
        neighbors_v = []
        for x in range(n):
            if x != v and (adj_masks[v] >> x) & 1:
                neighbors_v.append(x)
        
        for i_a in range(len(neighbors_v)):
            a = neighbors_v[i_a]
            if a == u: continue
            ma = adj_masks[a]
            for i_b in range(i_a + 1, len(neighbors_v)):
                b = neighbors_v[i_b]
                if b == u: continue
                mb = adj_masks[b]
                
                # v is in N(a) and N(b) by construction of neighbors_v
                # Check if |N(u) & N(a) & N(b)| >= 2
                common = adj_masks[u] & ma & mb
                if common.bit_count() >= 2:
                    return False
        
        # Check triples {v, a, b}
        # We need u in N(a) and u in N(b)
        neighbors_u = []
        for x in range(n):
            if x != u and (adj_masks[u] >> x) & 1:
                neighbors_u.append(x)
        
        for i_a in range(len(neighbors_u)):
            a = neighbors_u[i_a]
            if a == v: continue
            ma = adj_masks[a]
            for i_b in range(i_a + 1, len(neighbors_u)):
                b = neighbors_u[i_b]
                if b == v: continue
                mb = adj_masks[b]
                
                # u is in N(a) and N(b) by construction
                # Check if |N(v) & N(a) & N(b)| >= 2
                common = adj_masks[v] & ma & mb
                if common.bit_count() >= 2:
                    return False
        
        return True

    best_edges = []
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Start with random graph
    # 2. Remove edges to make K3,3 free
    # 3. Greedy add
    # 4. Local search
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        p = 0.4 + 0.3 * rng.random()
        adj_masks = [0] * n
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Clean - remove violating edges
        max_clean_iters = 500
        for _ in range(max_clean_iters):
            if time.time() > deadline - 5:
                break
            found_violation = False
            for i, j, k in triples:
                common = adj_masks[i] & adj_masks[j] & adj_masks[k]
                if common.bit_count() >= 3:
                    # Find neighbors in common
                    neighbors = []
                    for x in range(n):
                        if (common >> x) & 1:
                            neighbors.append(x)
                    # Pick random edges from the K3,3 to remove
                    # Try removing an edge that breaks the violation
                    removed = False
                    for _try in range(20):
                        vi = [i, j, k][rng.integers(0, 3)]
                        vx = neighbors[rng.integers(0, len(neighbors))]
                        if (adj_masks[vi] >> vx) & 1:
                            adj_masks[vi] &= ~(1 << vx)
                            adj_masks[vx] &= ~(1 << vi)
                            removed = True
                            break
                    if not removed:
                        # Fallback: remove any edge
                        vi, vx = i, neighbors[0]
                        adj_masks[vi] &= ~(1 << vx)
                        adj_masks[vx] &= ~(1 << vi)
                    found_violation = True
                    break
            if not found_violation:
                break
        
        # Rebuild edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        
        current_count = len(edges)
        
        # Greedy Add
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            if can_add_mask(adj_masks, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append((u, v))
                current_count += 1
        
        # Local Search - improved version
        improved = True
        max_local_iters = 50
        for _ in range(max_local_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            edge_list = list(edges)
            rng.shuffle(edge_list)
            
            for u, v in edge_list:
                if time.time() > deadline - 5:
                    break
                
                # Remove (u,v)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                edges.remove((u, v))
                
                # Try to add up to 2 edges
                missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
                rng.shuffle(missing_now)
                
                added = 0
                for a, b in missing_now:
                    if added >= 2:
                        break
                    if can_add_mask(adj_masks, a, b):
                        adj_masks[a] |= (1 << b)
                        adj_masks[b] |= (1 << a)
                        edges.append((a, b))
                        added += 1
                        
                if added > 1:
                    improved = True
                elif added == 0:
                    # Restore
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append((u, v))
                    
            if not improved:
                break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            if best_count >= 60:
                if time.time() > deadline - 20:
                    break

    return [list(e) for e in best_edges]
