import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def popcount(x):
        return bin(x).count('1')

    def greedy_clique_cover(n, edges, adj, rng, deadline, mode='weighted'):
        """
        Greedy clique partition. Modes:
        - 'min_common': pick edge with fewest common neighbors
        - 'max_degree': pick vertex with highest remaining degree
        - 'weighted': composite score balancing common neighbors and degree
        """
        remaining = set(edges)
        if not remaining:
            return []
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        rem_deg = [0] * n
        for i in range(n):
            rem_deg[i] = popcount(rem_adj[i])
        
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            if mode == 'min_common':
                # Pick edge with fewest common neighbors
                edge_list = list(remaining)
                if len(edge_list) > 100:
                    sample = rng.sample(edge_list, min(100, len(edge_list)))
                else:
                    sample = edge_list
                
                best_edge = None
                best_common_mask = None
                best_count = float('inf')
                for u, v in sample:
                    common = rem_adj[u] & rem_adj[v]
                    count = popcount(common)
                    if count < best_count:
                        best_count = count
                        best_edge = (u, v)
                        best_common_mask = common
                
                u, v = best_edge
                clique = [u, v]
                common = best_common_mask
                
            elif mode == 'max_degree':
                # Pick vertex with highest remaining degree
                active = [v for v in range(n) if rem_deg[v] > 0]
                if not active:
                    break
                if len(active) > 50:
                    sample = rng.sample(active, min(50, len(active)))
                else:
                    sample = active
                best_vertex = max(sample, key=lambda v: rem_deg[v])
                clique = [best_vertex]
                common = rem_adj[best_vertex]
                
            else:  # weighted
                # Composite: score = w1 * (1/common_count) + w2 * degree
                # Pick edge that balances difficulty and potential
                edge_list = list(remaining)
                if len(edge_list) > 100:
                    sample = rng.sample(edge_list, min(100, len(edge_list)))
                else:
                    sample = edge_list
                
                best_edge = None
                best_common_mask = None
                best_score = -float('inf')
                for u, v in sample:
                    common = rem_adj[u] & rem_adj[v]
                    count = popcount(common)
                    # Weighted score: prefer edges with moderate common (not too hard, not too easy)
                    # and high combined degree
                    deg_sum = rem_deg[u] + rem_deg[v]
                    # Soft scoring: reward edges that are "tight" (low common) but in dense areas
                    score = 0.6 * (1.0 / max(1, count)) + 0.4 * (deg_sum / max(1, n))
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
                        best_common_mask = common
                
                u, v = best_edge
                clique = [u, v]
                common = best_common_mask
            
            # Expand clique: greedily add vertex with max degree in common
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                best_w = -1
                best_w_score = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w in clique:
                        continue
                    # Score candidates by how well they fit: degree within common
                    w_count = popcount(common & rem_adj[w])
                    # Also consider overall remaining degree for diversity
                    w_score = 0.7 * w_count + 0.3 * rem_deg[w]
                    if w_score > best_w_score:
                        best_w_score = w_score
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            # Remove all edges within clique from remaining
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                    rem_deg[u2] -= 1
                    rem_deg[v2] -= 1
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def local_improve(n, edges, cliques, deadline):
        """
        Try to reduce clique count by:
        1. Merging two cliques if their union is a clique
        2. Splitting large cliques to enable merges elsewhere
        """
        if len(cliques) <= 1:
            return cliques
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try merging pairs of cliques
            merged = False
            for i in range(len(cliques)):
                if merged:
                    break
                for j in range(i + 1, len(cliques)):
                    # Check if union is a clique
                    union = cliques[i] + cliques[j]
                    union_set = set(union)
                    is_clique = True
                    for k in range(len(union)):
                        for l in range(k + 1, len(union)):
                            u, v = union[k], union[l]
                            if u > v:
                                u, v = v, u
                            if (u, v) not in edges:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        # Merge
                        cliques[i] = union
                        cliques.pop(j)
                        improved = True
                        merged = True
                        break
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        if n <= 32:
            num_restarts = min(400, max(10, int(remaining_time / 0.04)))
        else:
            num_restarts = min(250, max(5, int(remaining_time / 0.08)))
        
        modes = ['weighted', 'min_common', 'max_degree']
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            rng = random.Random(rng_master.randint(0, 2**31))
            mode = modes[restart % len(modes)]
            result = greedy_clique_cover(n, edges, adj, rng, deadline, mode=mode)
            if result is not None:
                # Apply local improvement
                result = local_improve(n, edges, result, deadline)
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    # Allocate time: give more to the larger instance
    # First instance (n=32) gets ~40% of time, second (n=64) gets ~60%
    t1_deadline = start_time + (DEADLINE - start_time) * 0.40
    result1 = solve_instance(32, 0.6, 601, t1_deadline)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
