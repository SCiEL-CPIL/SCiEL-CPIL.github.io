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

    def greedy_clique_cover(n, edges, adj, deadline):
        """
        Greedy: repeatedly find a large clique among remaining edges.
        Strategy: pick edge with minimum common neighbors, expand to max clique.
        Uses efficient data structures.
        """
        if not edges:
            return []
        
        # Track which edges are still uncovered
        edge_set = set(edges)
        if not edge_set:
            return []
        
        # Build adjacency for remaining edges
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        remaining_count = len(edges)
        
        while remaining_count > 0:
            if time.time() > deadline:
                return None
            
            # Find edge with fewest common neighbors in remaining graph
            # Sample if too many edges
            if remaining_count > 200:
                sample = random.sample(list(edge_set), min(200, remaining_count))
            else:
                sample = list(edge_set)
            
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
            clique_set = {u, v}
            
            # Expand clique: greedily add vertex with max degree in common
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                # Find candidate in common
                best_w = -1
                best_w_count = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w in clique_set:
                        continue
                    w_count = popcount(common & rem_adj[w])
                    if w_count > best_w_count:
                        best_w_count = w_count
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                clique_set.add(best_w)
                common &= rem_adj[best_w]
                added = True
            
            # Remove all edges within clique from remaining
            edges_in_clique = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (min(a, b), max(a, b))
                    if e in edge_set:
                        edges_in_clique.add(e)
                        edge_set.remove(e)
                        rem_adj[a] &= ~(1 << b)
                        rem_adj[b] &= ~(1 << a)
            
            cliques.append(clique)
            remaining_count -= len(edges_in_clique)
        
        return cliques

    def try_merge_cliques(n, edges, adj, cliques, deadline):
        """Try to reduce clique count by merging compatible cliques."""
        if not cliques or len(cliques) < 2:
            return cliques
        
        if time.time() > deadline:
            return cliques
        
        # Build edge-to-clique mapping
        edge_to_clique = {}
        for ci, clique in enumerate(cliques):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (min(a, b), max(a, b))
                    edge_to_clique[e] = ci
        
        # Try to merge pairs of cliques if their union is a clique
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            new_cliques = []
            used = [False] * len(cliques)
            
            for i in range(len(cliques)):
                if used[i]:
                    continue
                for j in range(i + 1, len(cliques)):
                    if used[j]:
                        continue
                    
                    # Check if union of cliques[i] and cliques[j] is a clique
                    union = set(cliques[i]) | set(cliques[j])
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u >= v:
                                continue
                            if not (adj[u] >> v) & 1:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        new_cliques.append(list(union))
                        used[i] = True
                        used[j] = True
                        improved = True
                        break
                if used[i]:
                    continue
                new_cliques.append(cliques[i])
                used[i] = True
            
            cliques = new_cliques
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        
        if not edges:
            return []
        
        # Baseline: each edge is its own clique
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        # Try multiple greedy restarts with different random seeds
        rng_master = random.Random(12345)
        
        # Determine number of restarts based on remaining time
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed
        if n <= 32:
            num_restarts = min(300, max(5, int(remaining_time / 0.05)))
        else:
            num_restarts = min(150, max(5, int(remaining_time / 0.1)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Vary the strategy slightly
            if restart % 3 == 0:
                result = greedy_clique_cover(n, edges, adj, deadline)
            elif restart % 3 == 1:
                # Try with shuffled edges
                shuffled = edges[:]
                random.Random(restart).shuffle(shuffled)
                result = greedy_clique_cover(n, shuffled, adj, deadline)
            else:
                # Try with reversed edge order
                reversed_edges = list(reversed(edges))
                result = greedy_clique_cover(n, reversed_edges, adj, deadline)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        # Try to merge cliques
        if time.time() < deadline - 5:
            merged = try_merge_cliques(n, edges, adj, best_cliques, deadline)
            if len(merged) < best_count:
                best_cliques = merged
                best_count = len(merged)
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601, DEADLINE)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
