"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    Uses bitmask-based scoring with incremental updates, multiple diverse seeds
    (K_{4,12}, K_{5,11}, K_{6,10}), simulated annealing, and systematic
    local search (1-edge and 2-edge swaps) to escape local optima.
    """
    n = 16
    target_edges = 65
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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

    def compute_score_full(adj_masks, deg):
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

    def update_score(adj_masks, deg, removed_edges, added_edges, current_score):
        """
        Efficiently update the score after a swap.
        Recalculates max sum for triangles involving vertices in the changed edges.
        """
        # Identify affected vertices
        affected_verts = set()
        for u, v in removed_edges:
            affected_verts.add(u)
            affected_verts.add(v)
        for u, v in added_edges:
            affected_verts.add(u)
            affected_verts.add(v)
        
        # We need to check all triangles that include at least one affected vertex
        # AND are still valid in the new graph.
        # Also, triangles NOT including affected vertices keep their old sums,
        # so we only need to find the max among the "new/changed" ones and compare
        # with the max of the "unchanged" ones?
        # Actually, if the global max was in an unchanged triangle, it remains.
        # If the global max was in a changed triangle, it might decrease.
        # So we must check:
        # 1. Max sum of triangles involving affected vertices.
        # 2. Max sum of triangles NOT involving affected vertices (if we don't track it, 
        #    we might miss a case where the old max was removed and the new max is lower,
        #    but we don't know the second max. 
        #    However, for small n, recalculating full score is O(T). 
        #    Let's stick to full recalc if incremental is complex, 
        #    BUT n=16 is small. 
        #    Let's try a hybrid: full recalc is ~1000 ops. 
        #    With 150s, we can do ~100k swaps. 
        #    Full recalc is sufficient.
        pass

    # Since n=16, full recalc is very fast. 
    # Let's optimize the loop structure instead.
    
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

    def edges_to_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        return es

    def edges_to_list(es):
        return [[u, v] for u, v in es]

    # Seed Generators
    def make_seed_k412():
        # K_{4,12}: 48 edges. Need 17 more.
        # Part A: 0-3, Part B: 4-15
        edges = []
        for u in range(4):
            for v in range(4, 16):
                edges.append([u, v])
        # Add 17 edges in B. 
        # Try to distribute degrees evenly.
        # B has 12 vertices. 17 edges.
        # Random matching-like structure
        b_verts = list(range(4, 16))
        random.shuffle(b_verts)
        internal_edges = []
        # Create a 2-regular structure + extra
        # Just add random edges, avoiding duplicates
        while len(internal_edges) < 17:
            u, v = random.sample(b_verts, 2)
            e = (min(u,v), max(u,v))
            if e not in internal_edges:
                internal_edges.append(e)
        for u, v in internal_edges:
            edges.append([u, v])
        return edges

    def make_seed_k511():
        # K_{5,11}: 55 edges. Need 10 more.
        edges = []
        for u in range(5):
            for v in range(5, 16):
                edges.append([u, v])
        b_verts = list(range(5, 16))
        internal_edges = []
        while len(internal_edges) < 10:
            u, v = random.sample(b_verts, 2)
            e = (min(u,v), max(u,v))
            if e not in internal_edges:
                internal_edges.append(e)
        for u, v in internal_edges:
            edges.append([u, v])
        return edges

    def make_seed_k610():
        # K_{6,10}: 60 edges. Need 5 more.
        edges = []
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        b_verts = list(range(6, 16))
        internal_edges = []
        while len(internal_edges) < 5:
            u, v = random.sample(b_verts, 2)
            e = (min(u,v), max(u,v))
            if e not in internal_edges:
                internal_edges.append(e)
        for u, v in internal_edges:
            edges.append([u, v])
        return edges

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)

    best_edges = None
    best_score = float('inf')
    best_es = None

    random.seed(42)

    # Try multiple seeds
    seeds = [make_seed_k412(), make_seed_k511(), make_seed_k610()]
    # Add a random 65-edge graph
    random_edges = random.sample(all_pairs, 65)
    seeds.append(random_edges)

    for seed_edges in seeds:
        es = edges_to_set(seed_edges)
        if len(es) != 65: continue
        adj, deg = build_adj(es)
        s = score_from_adj(adj, deg)
        if s < best_score:
            best_score = s
            best_edges = edges_to_list(es)
            best_es = es

    # Local Search Phase 1: Simulated Annealing
    if time.time() < deadline - 30:
        random.seed(123)
        current_es = set(best_es)
        current_adj, current_deg = build_adj(current_es)
        current_score = best_score
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        max_iter = 100000
        
        non_edges = [p for p in all_pairs if p not in current_es]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30: break
            if T < T_min: break
            
            e_remove = random.choice(list(current_es))
            if not non_edges:
                non_edges = [p for p in all_pairs if p not in current_es]
            if not non_edges: break
            e_add = random.choice(non_edges)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = build_adj(new_es)
            new_score = score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
                    best_es = current_es
                non_edges = [p for p in all_pairs if p not in current_es]
            
            T *= cooling

    # Local Search Phase 2: Systematic 1-Edge Swaps
    if time.time() < deadline - 10:
        current_es = set(best_es)
        current_adj, current_deg = build_adj(current_es)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing each edge
            for e_remove in list(current_es):
                if time.time() > deadline - 10: break
                # Try adding each non-edge
                for e_add in all_pairs:
                    if e_add in current_es: continue
                    new_es = set(current_es)
                    new_es.remove(e_remove)
                    new_es.add(e_add)
                    new_adj, new_deg = build_adj(new_es)
                    new_score = score_from_adj(new_adj, new_deg)
                    if new_score < current_score:
                        current_es = new_es
                        current_adj, current_deg = new_adj, new_deg
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = edges_to_list(current_es)
                            best_es = current_es
                        break # Restart loop with new state

    # Local Search Phase 3: Systematic 2-Edge Swaps (if stuck > 23)
    if best_score > 23 and time.time() < deadline - 2:
        current_es = set(best_es)
        current_adj, current_deg = build_adj(current_es)
        current_score = best_score
        
        # Sampled 2-edge swaps
        random.seed(999)
        max_attempts = 50000
        for _ in range(max_attempts):
            if time.time() > deadline - 2: break
            
            e_rem1, e_rem2 = random.sample(list(current_es), 2)
            non_edges = [p for p in all_pairs if p not in current_es]
            if len(non_edges) < 2: break
            e_add1, e_add2 = random.sample(non_edges, 2)
            
            new_es = set(current_es)
            new_es.discard(e_rem1)
            new_es.discard(e_rem2)
            new_es.add(e_add1)
            new_es.add(e_add2)
            
            if len(new_es) != 65: continue
            
            new_adj, new_deg = build_adj(new_es)
            new_score = score_from_adj(new_adj, new_deg)
            
            if new_score < current_score:
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
                    best_es = current_es

    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
