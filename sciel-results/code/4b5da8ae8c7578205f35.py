"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_graph_bitmask(n, adj_masks, deg):
    """Compute the maximum degree sum over all triangles using bitmasks."""
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                common = adj_masks[u] & adj_masks[v]
                w_mask = ((1 << n) - 1) ^ ((1 << (v + 1)) - 1)
                common = common & w_mask
                while common:
                    w = common & (-common)
                    w_idx = w.bit_length() - 1
                    common ^= w
                    s = deg[u] + deg[v] + deg[w_idx]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _score_sq_sum(n, adj_masks, deg):
    """Compute sum of squared triangle degree sums (smooth surrogate)."""
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                common = adj_masks[u] & adj_masks[v]
                w_mask = ((1 << n) - 1) ^ ((1 << (v + 1)) - 1)
                common = common & w_mask
                while common:
                    w = common & (-common)
                    w_idx = w.bit_length() - 1
                    common ^= w
                    s = deg[u] + deg[v] + deg[w_idx]
                    total += s * s
    return total

def _build_adj_masks_deg(n, edges):
    """Build adjacency bitmasks and degree list from edge list."""
    adj_masks = [0] * n
    deg = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    return adj_masks, deg

def _simulated_annealing_qubo(n, initial_edges, deadline, seed=42):
    """SA minimizing sum of squared triangle degree sums (QUBO surrogate)."""
    rng = random.Random(seed)
    
    adj_masks, deg = _build_adj_masks_deg(n, initial_edges)
    current_sq = _score_sq_sum(n, adj_masks, deg)
    current_max = _score_graph_bitmask(n, adj_masks, deg)
    
    best_edges = list(initial_edges)
    best_sq = current_sq
    best_max = current_max
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edge_set = set((min(u,v), max(u,v)) for u, v in initial_edges)
    
    T_start = 8.0
    T_end = 0.05
    n_iter_total = 80000
    i = 0
    
    while time.time() < deadline:
        i += 1
        T = T_start * (T_end / T_start) ** (i / n_iter_total)
        
        # Pick random edge to remove
        e_remove = rng.choice(initial_edges)
        u1, v1 = e_remove
        e_remove_key = (min(u1, v1), max(u1, v1))
        
        # Pick random non-edge to add
        e_add = rng.choice(all_edges)
        e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
        if e_add_key in current_edge_set:
            continue
        
        u2, v2 = e_add
        
        # Perform swap
        adj_masks[u1] &= ~(1 << v1)
        adj_masks[v1] &= ~(1 << u1)
        adj_masks[u2] |= (1 << v2)
        adj_masks[v2] |= (1 << u2)
        deg[u1] -= 1
        deg[v1] -= 1
        deg[u2] += 1
        deg[v2] += 1
        
        current_edge_set.remove(e_remove_key)
        current_edge_set.add(e_add_key)
        initial_edges.remove(e_remove)
        initial_edges.append(e_add)
        
        # Compute new scores
        new_sq = _score_sq_sum(n, adj_masks, deg)
        new_max = _score_graph_bitmask(n, adj_masks, deg)
        
        # Acceptance: primarily on sq-sum, with tiebreak on max
        delta_sq = new_sq - current_sq
        if delta_sq <= 0 or rng.random() < math.exp(-delta_sq / max(T, 1e-10)):
            current_sq = new_sq
            # Track best by true max, tiebreak by sq
            if new_max < best_max or (new_max == best_max and new_sq < best_sq):
                best_max = new_max
                best_sq = new_sq
                best_edges = list(initial_edges)
        else:
            # Revert
            adj_masks[u1] |= (1 << v1)
            adj_masks[v1] |= (1 << u1)
            adj_masks[u2] &= ~(1 << v2)
            adj_masks[v2] &= ~(1 << u2)
            deg[u1] += 1
            deg[v1] += 1
            deg[u2] -= 1
            deg[v2] -= 1
            
            current_edge_set.remove(e_add_key)
            current_edge_set.add(e_remove_key)
            initial_edges.remove(e_add)
            initial_edges.append(e_remove)
    
    return best_edges, best_max

def _hill_climb_max(n, edges, deadline):
    """Hill climbing on true max objective for final polish."""
    adj_masks, deg = _build_adj_masks_deg(n, edges)
    current_score = _score_graph_bitmask(n, adj_masks, deg)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for e_remove in list(edges):
            if time.time() > deadline - 1:
                break
            u1, v1 = e_remove
            e_remove_key = (min(u1, v1), max(u1, v1))
            
            for e_add in all_edges:
                e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
                if e_add_key in current_edge_set:
                    continue
                
                u2, v2 = e_add
                
                adj_masks[u1] &= ~(1 << v1)
                adj_masks[v1] &= ~(1 << u1)
                adj_masks[u2] |= (1 << v2)
                adj_masks[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _score_graph_bitmask(n, adj_masks, deg)
                
                if new_score < current_score:
                    current_edge_set.remove(e_remove_key)
                    current_edge_set.add(e_add_key)
                    edges.remove(e_remove)
                    edges.append(e_add)
                    current_score = new_score
                    improved = True
                else:
                    adj_masks[u1] |= (1 << v1)
                    adj_masks[v1] |= (1 << u1)
                    adj_masks[u2] &= ~(1 << v2)
                    adj_masks[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
    
    return edges, current_score

def _generate_initial_configs(n, target_edges, rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: K_{8,8} + one edge inside part A
    edges1 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    configs.append(edges1)
    
    # Config 2: K_{8,8} + one edge inside part B
    edges2 = [[u, v] for u in range(8) for v in range(8, 16)] + [[8, 9]]
    configs.append(edges2)
    
    # Config 3: K_{8,8} + two edges inside part A
    edges3 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1], [2, 3]]
    configs.append(edges3)
    
    # Config 4: K_{8,8} + two edges inside part B
    edges4 = [[u, v] for u in range(8) for v in range(8, 16)] + [[8, 9], [10, 11]]
    configs.append(edges4)
    
    # Config 5: K_{8,8} + one cross-part removal + two intra-part additions
    edges5 = [[u, v] for u in range(8) for v in range(8, 16) if not (u == 0 and v == 8)]
    edges5.append([0, 1])
    edges5.append([8, 9])
    configs.append(edges5)
    
    # Random near-balanced configs
    for trial in range(4):
        edges = []
        left = list(range(8))
        right = list(range(8, 16))
        rng.shuffle(left)
        rng.shuffle(right)
        
        for u in left:
            for v in right:
                if rng.random() < 0.92:
                    edges.append([u, v])
        
        while len(edges) < target_edges:
            if rng.random() < 0.5:
                u, v = rng.sample(left, 2)
            else:
                u, v = rng.sample(right, 2)
            ek = (min(u,v), max(u,v))
            if ek not in set((min(a,b),max(a,b)) for a,b in edges):
                edges.append([u, v])
        
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        
        if len(edges) == target_edges:
            configs.append(edges)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    rng = random.Random(42)
    
    initial_configs = _generate_initial_configs(n, target_edges, rng)
    
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: QUBO-style SA from each config (minimize sum of squares)
    n_configs = len(initial_configs)
    for idx, init_edges in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        remaining_configs = n_configs - idx
        per_config_time = (deadline - 10 - time.time()) / max(remaining_configs, 1)
        config_deadline = time.time() + per_config_time
        
        edges, score = _simulated_annealing_qubo(n, list(init_edges), config_deadline, seed=100 + idx)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Hill climbing polish on true max objective
    if best_edges is not None and time.time() < deadline - 5:
        polish_deadline = deadline - 2
        best_edges, best_score = _hill_climb_max(n, best_edges, polish_deadline)
    
    # Phase 3: Additional SA restarts from best if time remains
    if time.time() < deadline - 8:
        extra_deadline = deadline - 3
        edges, score = _simulated_annealing_qubo(n, list(best_edges), extra_deadline, seed=999)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Final hill climb
    if time.time() < deadline - 2:
        best_edges, best_score = _hill_climb_max(n, best_edges, deadline - 1)
    
    if best_edges is None:
        best_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
