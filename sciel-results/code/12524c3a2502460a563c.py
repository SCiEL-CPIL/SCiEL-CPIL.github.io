import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 18
    all_vertices = list(range(n))
    
    # Precompute all 12-subsets
    subsets_12 = list(combinations(range(n), 12))
    subset_masks = []
    for combo in subsets_12:
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subset_masks.append(mask)
    
    def is_triangle_free_mask(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True
    
    def count_triangles_in_mask(mask, adj):
        """Count triangles in subgraph induced by mask."""
        count = 0
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    count += 1
        return count // 3
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True
    
    def find_conflict_subsets(adj, limit=200):
        """Find triangle-free 12-subsets, return list of (mask, non_edges_in_mask)."""
        conflicts = []
        for mask in subset_masks:
            if is_triangle_free_mask(mask, adj):
                non_edges = []
                for u in range(n):
                    if not (mask & (1 << u)):
                        continue
                    for v in range(u+1, n):
                        if (mask & (1 << v)) and not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                conflicts.append((mask, non_edges))
                if len(conflicts) >= limit:
                    break
        return conflicts
    
    def score_adj(adj):
        """Count triangle-free 12-subsets."""
        count = 0
        for mask in subset_masks:
            if is_triangle_free_mask(mask, adj):
                count += 1
        return count
    
    def dpll_solve(adj, max_time=None):
        """
        DPLL-style: repeatedly pick a conflict, try to fix it by adding an edge.
        Returns improved adj or None if no improvement found.
        """
        if max_time is None:
            max_time = deadline
        adj = list(adj)
        
        # First, greedily add all possible edges that don't create K4
        # and help reduce triangle-free subsets
        improved = True
        while improved:
            improved = False
            conflicts = find_conflict_subsets(adj, limit=50)
            if not conflicts:
                break
            
            # Count how often each non-edge appears in conflicts
            edge_freq = {}
            for mask, non_edges in conflicts:
                for e in non_edges:
                    edge_freq[e] = edge_freq.get(e, 0) + 1
            
            # Try adding the most frequent non-edge that's K4-safe
            candidates = sorted(edge_freq.items(), key=lambda x: -x[1])
            for (u, v), freq in candidates:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    break
        
        return adj
    
    def random_maximal_k4_free():
        """Generate a random maximal K4-free graph."""
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    # Best solution found so far
    best_adj = None
    best_score = 999
    
    def update_best(adj):
        nonlocal best_adj, best_score
        s = score_adj(adj)
        if s < best_score:
            best_score = s
            best_adj = list(adj)
            if s == 0:
                return True
        return False
    
    # Phase 1: Try structured seeds
    # Seed 1: 3 disjoint K6's (score 12)
    seed1 = [0] * n
    for start in [0, 6, 12]:
        for i in range(start, start+6):
            for j in range(i+1, start+6):
                seed1[i] |= (1 << j)
                seed1[j] |= (1 << i)
    if not update_best(seed1):
        pass
    
    # Seed 2: Complete bipartite K9,9
    seed2 = [0] * n
    for i in range(9):
        for j in range(9, 18):
            seed2[i] |= (1 << j)
            seed2[j] |= (1 << i)
    if not update_best(seed2):
        pass
    
    # Seed 3: Turan graph T(18,3) = K6,6,6 (complement of 3 disjoint K6's)
    seed3 = [0] * n
    for i in range(18):
        for j in range(i+1, 18):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                seed3[i] |= (1 << j)
                seed3[j] |= (1 << i)
    if not update_best(seed3):
        pass
    
    # Phase 2: Random restarts with DPLL improvement
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        
        # Generate random maximal K4-free graph
        adj = random_maximal_k4_free()
        
        # Quick pre-filter: count conflicts
        conflicts = find_conflict_subsets(adj, limit=30)
        if len(conflicts) >= 30:
            continue
        
        # Apply DPLL improvement
        adj_improved = dpll_solve(adj)
        
        if update_best(adj_improved):
            if best_score == 0:
                break
            # Try more aggressive improvement from this solution
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                adj2 = list(adj_improved)
                # Random perturbation: remove a random edge, re-maximize
                edges_in_adj = [(u,v) for u in range(n) for v in range(u+1,n) if adj2[u] & (1<<v)]
                if edges_in_adj:
                    u, v = random.choice(edges_in_adj)
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)
                    # Re-maximize
                    all_edges = [(a,b) for a in range(n) for b in range(a+1,n) if not (adj2[a] & (1<<b))]
                    random.shuffle(all_edges)
                    for a, b in all_edges:
                        if can_add_edge(adj2, a, b):
                            adj2[a] |= (1 << b)
                            adj2[b] |= (1 << a)
                    adj2 = dpll_solve(adj2)
                    if update_best(adj2):
                        if best_score == 0:
                            break
    
    # Phase 3: If we have a good solution, try more targeted search
    if best_adj is not None and best_score > 0 and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            adj = list(best_adj)
            # Try removing random edges and re-optimizing
            edges_in_adj = [(u,v) for u in range(n) for v in range(u+1,n) if adj[u] & (1<<v)]
            if not edges_in_adj:
                break
            for _ in range(random.randint(1, 3)):
                u, v = random.choice(edges_in_adj)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges_in_adj = [(u2,v2) for u2 in range(n) for v2 in range(u2+1,n) if adj[u2] & (1<<v2)]
                if not edges_in_adj:
                    break
            # Re-maximize and improve
            all_edges = [(a,b) for a in range(n) for b in range(a+1,n) if not (adj[a] & (1<<b))]
            random.shuffle(all_edges)
            for a, b in all_edges:
                if can_add_edge(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            adj = dpll_solve(adj)
            if update_best(adj):
                if best_score == 0:
                    break
    
    if best_adj is not None:
        return adj_to_edges(best_adj)
    
    # Fallback: return seed1
    return adj_to_edges(seed1)

# EVOLVE-BLOCK-END
