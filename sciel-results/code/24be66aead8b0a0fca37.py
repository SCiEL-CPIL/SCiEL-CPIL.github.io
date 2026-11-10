import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Multi-restart Simulated Annealing to minimize regular induced
    subgraphs of size 6, then verify all sizes >= 6.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit, leave buffer)

    def get_subsets(n, size):
        """Returns list of (mask, vertices) for all subsets of given size."""
        result = []
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            result.append((mask, comb))
        return result

    def build_edge_subset_index(n, size):
        """
        Returns (subsets, edge_idx) where edge_idx[u][v] is a list of indices
        into subsets for subsets of 'size' that contain edge (u, v).
        """
        subsets = get_subsets(n, size)
        edge_idx = [[[] for _ in range(n)] for _ in range(n)]
        
        for i, (mask, verts) in enumerate(subsets):
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    edge_idx[u][v].append(i)
                    edge_idx[v][u].append(i)
                    
        return subsets, edge_idx

    def is_regular_mask(adj, mask, verts):
        """Check if induced subgraph on verts is regular."""
        if not verts:
            return False
        first_deg = (adj[verts[0]] & mask).bit_count()
        for u in verts[1:]:
            if (adj[u] & mask).bit_count() != first_deg:
                return False
        return True

    def fast_verify(adj, n, subsets_by_size):
        """
        Fast verification: check all subsets of size >= 6 for regularity.
        Uses precomputed subsets. Returns True if valid.
        """
        for k in range(6, n + 1):
            if k not in subsets_by_size:
                continue
            for mask, verts in subsets_by_size[k]:
                if is_regular_mask(adj, mask, verts):
                    return False
        return True

    def try_find_graph(n, time_budget, max_restarts=10):
        """
        Attempts to find a valid graph on n vertices using multi-restart SA.
        Returns edges list or None.
        """
        if time.time() > time_budget:
            return None
            
        # Precompute for this n
        subsets6, edge_idx6 = build_edge_subset_index(n, 6)
        num_subsets6 = len(subsets6)
        
        # Precompute subsets for verification (only if we might need them)
        # We'll build them lazily
        
        best_adj_overall = None
        best_cost_overall = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Initialize random graph with edge probability ~0.5
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Initial cost: count of regular size-6 subgraphs
            is_reg6 = [False] * num_subsets6
            cost = 0
            for i, (mask, verts) in enumerate(subsets6):
                if is_regular_mask(adj, mask, verts):
                    is_reg6[i] = True
                    cost += 1
            
            best_cost = cost
            best_adj = adj[:]
            
            # If already zero, verify immediately
            if best_cost == 0:
                # Build subsets for verification
                subsets_by_size = {}
                for k in range(6, n + 1):
                    if k != 6:
                        subsets_by_size[k] = get_subsets(n, k)
                subsets_by_size[6] = subsets6
                
                if fast_verify(best_adj, n, subsets_by_size):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if best_adj[u] & (1 << v):
                                edges.append([u, v])
                    return edges
            
            # SA parameters
            T = 1.0
            cooling = 0.9995
            min_T = 0.005
            steps = 0
            max_steps = 30000
            
            while time.time() < time_budget and T > min_T and steps < max_steps:
                steps += 1
                # Pick random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    delta_flip = -1
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    delta_flip = 1
                
                # Update cost
                old_cost = cost
                affected_indices = edge_idx6[u][v]
                
                for idx in affected_indices:
                    mask, verts = subsets6[idx]
                    new_reg = is_regular_mask(adj, mask, verts)
                    old_reg = is_reg6[idx]
                    
                    if new_reg != old_reg:
                        is_reg6[idx] = new_reg
                        if new_reg:
                            cost += 1
                        else:
                            cost -= 1
                
                delta = cost - old_cost
                
                # Acceptance
                if delta <= 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)
                
                if not accept:
                    # Revert
                    if delta_flip == -1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    # Revert cost changes
                    for idx in affected_indices:
                        mask, verts = subsets6[idx]
                        new_reg = is_regular_mask(adj, mask, verts)
                        old_reg = is_reg6[idx]
                        
                        if new_reg != old_reg:
                            is_reg6[idx] = new_reg
                            if new_reg:
                                cost += 1
                            else:
                                cost -= 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        break
                
                T *= cooling
            
            # Track best across restarts
            if best_cost < best_cost_overall:
                best_cost_overall = best_cost
                best_adj_overall = best_adj
            
            # If we found zero cost, verify
            if best_cost == 0:
                subsets_by_size = {}
                for k in range(6, n + 1):
                    if k != 6:
                        subsets_by_size[k] = get_subsets(n, k)
                subsets_by_size[6] = subsets6
                
                if fast_verify(best_adj, n, subsets_by_size):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if best_adj[u] & (1 << v):
                                edges.append([u, v])
                    return edges
        
        # If we have a best_adj with low cost but not zero, return None
        # (we need exactly zero regular subgraphs)
        return None

    # Try n=20, then 19, etc.
    # Allocate time budget: more time for larger n
    for n in range(20, 4, -1):
        # Give more time to n=20
        if n == 20:
            time_budget = deadline
        else:
            time_budget = deadline - 5
        
        edges = try_find_graph(n, time_budget)
        if edges is not None:
            return {'n': n, 'edges': edges}
        
        # If we spent too much time, break
        if time.time() > deadline - 8:
            break
            
    # Fallback: Return a trivial graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
