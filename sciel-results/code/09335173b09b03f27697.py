import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples (u, v, w) for K3,3 check
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute which triples contain each vertex for faster can_add
    triples_containing = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        triples_containing[a].append(idx)
        triples_containing[b].append(idx)
        triples_containing[c].append(idx)
    
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
        Only checks triples containing u or v using precomputed lists.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Get unique triple indices containing u or v
        checked = set(triples_containing[u])
        checked.update(triples_containing[v])
        
        for tidx in checked:
            a, b, c = triples[tidx]
            
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
    
    def find_violation(adj_mask):
        """Find a K3,3 violation, return (u,v,w,common_neighbors) or None."""
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return (u, v, w, common)
        return None
    
    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(200):
            if time.time() > deadline:
                break
            viol = find_violation(adj_mask)
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
                        if find_violation(adj_mask) is None:
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
                    adj_mask[u] &= ~bit(c)
                    adj_mask[c] &= ~bit(u)
        return adj_mask
    
    def local_search(adj_mask, max_iters=300):
        """Local search with edge additions and swaps."""
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                adj_mask = greedy_add(adj_mask)
                continue
            
            # Try swaps: remove one edge, add another
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            swapped = False
            for i in range(min(30, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove (r_u, r_v)
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                # Add (a_u, a_v)
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if find_violation(adj_mask) is None:
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not swapped:
                # Try removing a random edge to escape local optimum
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    def simulated_annealing(adj_mask, max_iters=500):
        """Simulated Annealing to escape local optima."""
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        current_count = edge_count(adj_mask)
        best_adj = list(adj_mask)
        best_count = current_count
        
        # Temperature schedule
        t0 = 5.0
        t1 = 0.1
        cooling_rate = (t1 / t0) ** (1.0 / max(max_iters, 1))
        temperature = t0
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Randomly pick an edge to flip (add or remove)
            u, v = random.choice(all_pairs)
            has_edge = adj_mask[u] & bit(v)
            
            # Apply flip
            if has_edge:
                adj_mask[u] &= ~bit(v)
                adj_mask[v] &= ~bit(u)
            else:
                adj_mask[u] |= bit(v)
                adj_mask[v] |= bit(u)
            
            # Check validity
            if find_violation(adj_mask) is not None:
                # Revert invalid move
                if has_edge:
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                else:
                    adj_mask[u] &= ~bit(v)
                    adj_mask[v] &= ~bit(u)
                temperature *= cooling_rate
                continue
            
            new_count = edge_count(adj_mask)
            delta = new_count - current_count
            
            if delta > 0:
                # Accept improvement
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_adj = list(adj_mask)
            else:
                # Accept worse with probability exp(delta / temperature)
                if temperature > 0.01 and random.random() < np.exp(delta / temperature):
                    current_count = new_count
            
            temperature *= cooling_rate
        
        return best_adj
    
    # Use numpy for vectorized operations where possible
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Direct construction approach: start with known good structure
    # Try random restarts with different densities
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization with varying density
        density = 0.3 + rng.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and optimize with local search
        adj = local_search(adj, max_iters=100)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Early termination if we find a very good solution
        if best_edges > 145:
            break
    
    # Run simulated annealing on the best solution found
    if best_adj is not None and time.time() < deadline - 30:
        adj = list(best_adj)
        adj = simulated_annealing(adj, max_iters=800)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    # Final local search polish if time allows
    if best_adj is not None and time.time() < deadline - 10:
        adj = list(best_adj)
        adj = local_search(adj, max_iters=300)
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
