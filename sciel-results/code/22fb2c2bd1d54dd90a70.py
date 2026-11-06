"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    Uses bitmask-based scoring, multiple seeds, local search + simulated
    annealing with soft-max acceptance, and a 2-edge swap polish phase.
    """
    n = 16
    target_edges = 65
    
    start_time = time.time()
    deadline = start_time + 147.0  # 3s safety margin
    
    def score_from_adj(adj_masks, deg):
        """Compute max triangle degree sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            mu = adj_masks[u]
            du = deg[u]
            for v in range(u + 1, n):
                if mu & (1 << v):
                    common = mu & adj_masks[v]
                    dv = deg[v]
                    base = du + dv
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        w = lsb.bit_length() - 1
                        s = base + deg[w]
                        if s > max_sum:
                            max_sum = s
                        cm ^= lsb
        return max_sum
    
    def build_adj(edge_set):
        """Build bitmask adjacency and degree array from edge set."""
        adj_masks = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_masks, deg
    
    def compute_score(edges):
        """Compute the maximum degree sum over all triangles."""
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        adj_masks, deg = build_adj(es)
        return score_from_adj(adj_masks, deg)
    
    def make_seed_edges():
        """K_{6,10} + 5 internal edges in the 10-vertex part, spread out."""
        edges = []
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        # Use a matching-like structure to minimize triangle degree sums
        # Vertices 6-15 are the larger partition
        # Choose edges that spread degrees: (6,10), (7,11), (8,12), (9,13), (14,15)
        internal_edges = [(6,10), (7,11), (8,12), (9,13), (14,15)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    def make_seed_edges_v2():
        """K_{8,8} + one internal edge."""
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    def make_seed_edges_v3():
        """K_{7,9} + 2 internal edges (one in each part)."""
        edges = []
        for u in range(7):
            for v in range(7, 16):
                edges.append([u, v])
        edges.append([0, 1])
        edges.append([10, 11])
        return edges
    
    def edges_to_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        return es
    
    def edges_to_list(es):
        return [[u, v] for u, v in es]
    
    # Try multiple seeds
    best_edges = None
    best_score = float('inf')
    
    for seed_fn in [make_seed_edges, make_seed_edges_v2, make_seed_edges_v3]:
        seed_edges = seed_fn()
        s = compute_score(seed_edges)
        if s < best_score:
            best_score = s
            best_edges = seed_edges
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)
    
    # Local search with edge swaps - improved with targeted moves
    current_es = edges_to_set(best_edges)
    current_adj, current_deg = build_adj(current_es)
    current_score = best_score
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute non-edges list for faster access
    non_edges_list = [p for p in all_pairs if p not in current_es]
    
    while time.time() < deadline - 30 and no_improve < max_no_improve:
        # Try both random and targeted moves
        if random.random() < 0.3:
            # Targeted: remove an edge incident to a high-degree vertex in a triangle
            # Find the triangle with max degree sum
            max_tri_sum = 0
            max_tri = None
            for u in range(n):
                mu = current_adj[u]
                for v in range(u + 1, n):
                    if mu & (1 << v):
                        common = mu & current_adj[v]
                        cm = common
                        while cm:
                            lsb = cm & (-cm)
                            w = lsb.bit_length() - 1
                            s = current_deg[u] + current_deg[v] + current_deg[w]
                            if s > max_tri_sum:
                                max_tri_sum = s
                                max_tri = (u, v, w)
                            cm ^= lsb
            
            if max_tri:
                u, v, w = max_tri
                # Remove one of the edges in this triangle
                candidates = [(u,v), (u,w), (v,w)]
                e_remove = random.choice([c for c in candidates if c in current_es])
            else:
                e_remove = random.choice(list(current_es))
        else:
            e_remove = random.choice(list(current_es))
        
        # Update non_edges_list if needed
        if not non_edges_list:
            non_edges_list = [p for p in all_pairs if p not in current_es]
        if not non_edges_list:
            break
        e_add = random.choice(non_edges_list)
        
        new_es = set(current_es)
        new_es.remove(e_remove)
        new_es.add(e_add)
        
        new_adj, new_deg = build_adj(new_es)
        new_score = score_from_adj(new_adj, new_deg)
        
        if new_score < current_score:
            current_es = new_es
            current_adj, current_deg = new_adj, new_deg
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = edges_to_list(current_es)
            # Update non_edges_list
            non_edges_list = [p for p in all_pairs if p not in current_es]
        else:
            no_improve += 1
    
    # Simulated annealing with soft-max acceptance - improved
    if best_score >= 23 and time.time() < deadline - 15:
        random.seed(123)
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(current_es)
        current_score = best_score
        
        T = 3.0
        T_min = 0.05
        cooling = 0.9995
        max_iter = 80000
        
        non_edges_sa = [p for p in all_pairs if p not in current_es]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            if T < T_min:
                break
            
            # Mix of random and targeted moves
            if random.random() < 0.25:
                # Targeted move: remove edge from worst triangle
                max_tri_sum = 0
                max_tri = None
                for u in range(n):
                    mu = current_adj[u]
                    for v in range(u + 1, n):
                        if mu & (1 << v):
                            common = mu & current_adj[v]
                            cm = common
                            while cm:
                                lsb = cm & (-cm)
                                w = lsb.bit_length() - 1
                                s = current_deg[u] + current_deg[v] + current_deg[w]
                                if s > max_tri_sum:
                                    max_tri_sum = s
                                    max_tri = (u, v, w)
                                cm ^= lsb
                
                if max_tri:
                    u, v, w = max_tri
                    candidates = [(u,v), (u,w), (v,w)]
                    valid_cands = [c for c in candidates if c in current_es]
                    if valid_cands:
                        e_remove = random.choice(valid_cands)
                    else:
                        e_remove = random.choice(list(current_es))
                else:
                    e_remove = random.choice(list(current_es))
            else:
                e_remove = random.choice(list(current_es))
            
            if not non_edges_sa:
                non_edges_sa = [p for p in all_pairs if p not in current_es]
            if not non_edges_sa:
                break
            e_add = random.choice(non_edges_sa)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = build_adj(new_es)
            new_score = score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            # Soft-max acceptance
            if delta < 0:
                accept = True
            else:
                if delta > 20 * T:
                    accept = False
                else:
                    accept_prob = 1.0 / (1.0 + math.exp(delta / T))
                    accept = random.random() < accept_prob
            
            if accept:
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
                non_edges_sa = [p for p in all_pairs if p not in current_es]
            
            T *= cooling
    
    # Polish phase 1: exhaustive single-edge swaps
    if time.time() < deadline - 5:
        best_es = edges_to_set(best_edges)
        best_s = best_score
        
        for e_remove in list(best_es):
            if time.time() > deadline - 5:
                break
            for e_add in all_pairs:
                if e_add in best_es:
                    continue
                new_es = set(best_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = build_adj(new_es)
                new_s = score_from_adj(new_adj, new_deg)
                if new_s < best_s:
                    best_s = new_s
                    best_es = new_es
                    best_edges = edges_to_list(best_es)
                    best_score = best_s
    
    # Polish phase 2: 2-edge swaps to escape local optima - improved
    if best_score >= 23 and time.time() < deadline - 2:
        best_es = edges_to_set(best_edges)
        best_s = best_score
        best_adj, best_deg = build_adj(best_es)
        
        edge_list = list(best_es)
        non_edges_2 = [p for p in all_pairs if p not in best_es]
        random.seed(777)
        
        num_attempts = 0
        max_attempts = 50000
        
        while num_attempts < max_attempts and time.time() < deadline - 2:
            if len(edge_list) < 2 or len(non_edges_2) < 2:
                break
            
            # Targeted 2-edge swap: find worst triangle, remove 2 of its edges,
            # add 2 edges that don't create new high-sum triangles
            if random.random() < 0.3:
                max_tri_sum = 0
                max_tri = None
                for u in range(n):
                    mu = best_adj[u]
                    for v in range(u + 1, n):
                        if mu & (1 << v):
                            common = mu & best_adj[v]
                            cm = common
                            while cm:
                                lsb = cm & (-cm)
                                w = lsb.bit_length() - 1
                                s = best_deg[u] + best_deg[v] + best_deg[w]
                                if s > max_tri_sum:
                                    max_tri_sum = s
                                    max_tri = (u, v, w)
                                cm ^= lsb
                
                if max_tri:
                    u, v, w = max_tri
                    tri_edges = [(u,v), (u,w), (v,w)]
                    valid_te = [e for e in tri_edges if e in best_es]
                    if len(valid_te) >= 2:
                        i1, i2 = random.sample(range(len(valid_te)), 2)
                        e_rem1 = valid_te[i1]
                        e_rem2 = valid_te[i2]
                    else:
                        i1, i2 = random.sample(range(len(edge_list)), 2)
                        e_rem1 = edge_list[i1]
                        e_rem2 = edge_list[i2]
                else:
                    i1, i2 = random.sample(range(len(edge_list)), 2)
                    e_rem1 = edge_list[i1]
                    e_rem2 = edge_list[i2]
            else:
                i1, i2 = random.sample(range(len(edge_list)), 2)
                e_rem1 = edge_list[i1]
                e_rem2 = edge_list[i2]
            
            j1, j2 = random.sample(range(len(non_edges_2)), 2)
            e_add1 = non_edges_2[j1]
            e_add2 = non_edges_2[j2]
            
            new_es = set(best_es)
            new_es.discard(e_rem1)
            new_es.discard(e_rem2)
            new_es.add(e_add1)
            new_es.add(e_add2)
            
            if len(new_es) != target_edges:
                num_attempts += 1
                continue
            
            new_adj, new_deg = build_adj(new_es)
            new_s = score_from_adj(new_adj, new_deg)
            
            if new_s < best_s:
                best_s = new_s
                best_es = new_es
                best_edges = edges_to_list(best_es)
                best_score = best_s
                best_adj, best_deg = new_adj, new_deg
                edge_list = list(best_es)
                non_edges_2 = [p for p in all_pairs if p not in best_es]
            
            num_attempts += 1
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
