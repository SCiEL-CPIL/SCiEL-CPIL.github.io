"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def build_conflict_graph(values, K):
        """Build conflict graph: vertex per edge, adjacency = compatible (doesn't exceed K)."""
        num_edges = len(all_edges)
        # Precompute sum and product for each edge
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # For each edge, compute the set of sums/prods it introduces
        # Two edges are compatible if their combined distinct sums <= K and prods <= K
        # Actually: we need to check if adding both doesn't push either count over K
        # But for clique: we need a subset of 35 edges where total distinct sums <= K AND total distinct prods <= K
        
        # Simpler: two edges are "compatible" if they share the same sum OR the same product
        # This is a heuristic - edges that share sums/prods are more likely to form valid cliques
        adj = [set() for _ in range(num_edges)]
        for i in range(num_edges):
            for j in range(i+1, num_edges):
                # Compatible if they share sum or product (heuristic for collision)
                if edge_sums[i] == edge_sums[j] or edge_prods[i] == edge_prods[j]:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj, edge_sums, edge_prods
    
    def max_clique_branch_and_bound(adj, num_vertices, target_size, time_limit, rng):
        """Find a clique of size target_size using branch and bound."""
        best_clique = []
        
        def dfs(clique, candidates, depth):
            nonlocal best_clique
            if time.time() > time_limit:
                return False
            if len(clique) >= target_size:
                best_clique = list(clique)
                return True
            if len(clique) + len(candidates) < target_size:
                return False
            
            # Upper bound: current + remaining candidates
            remaining_needed = target_size - len(clique)
            if len(candidates) < remaining_needed:
                return False
            
            # Order candidates by degree in candidate subgraph (greedy heuristic)
            ordered = sorted(candidates, key=lambda v: sum(1 for u in candidates if u != v and u in adj[v]), reverse=True)
            
            for v in ordered:
                if time.time() > time_limit:
                    return len(best_clique) >= target_size
                new_clique = clique + [v]
                new_candidates = [u for u in candidates if u in adj[v] and u > v]
                if dfs(new_clique, new_candidates, depth + 1):
                    return True
                candidates = [u for u in candidates if u > v]
            return len(best_clique) >= target_size
        
        all_vertices = list(range(num_vertices))
        dfs([], all_vertices, 0)
        return best_clique
    
    def greedy_clique(adj, num_vertices, target_size, rng):
        """Greedy clique construction with randomization."""
        best_clique = []
        for attempt in range(50):
            clique = []
            candidates = set(range(num_vertices))
            while len(clique) < target_size and candidates:
                # Pick vertex with highest degree in current candidate set
                best_v = max(candidates, key=lambda v: sum(1 for u in candidates if u in adj[v]))
                clique.append(best_v)
                candidates = {u for u in candidates if u in adj[best_v]}
            if len(clique) > len(best_clique):
                best_clique = clique
            if len(best_clique) >= target_size:
                break
        return best_clique
    
    def find_edges_for_values(values, rng, time_limit):
        """Find 35 edges that minimize max(distinct_sums, distinct_prods)."""
        num_edges = len(all_edges)
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Strategy: group edges by (sum, product) pairs
        # Edges with same sum+product pair are ideal (they contribute only 1 distinct sum and 1 distinct product)
        pair_to_edges = {}
        for i in range(num_edges):
            key = (edge_sums[i], edge_prods[i])
            if key not in pair_to_edges:
                pair_to_edges[key] = []
            pair_to_edges[key].append(i)
        
        # Also group by sum alone and product alone
        sum_to_edges = {}
        prod_to_edges = {}
        for i in range(num_edges):
            s = edge_sums[i]
            p = edge_prods[i]
            if s not in sum_to_edges:
                sum_to_edges[s] = []
            sum_to_edges[s].append(i)
            if p not in prod_to_edges:
                prod_to_edges[p] = []
            prod_to_edges[p].append(i)
        
        # Greedy: pick edges that maximize reuse of existing sums/prods
        selected = []
        sums_used = set()
        prods_used = set()
        remaining = list(range(num_edges))
        
        # First pass: pick edges that share both sum and product with already selected
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_idx = None
            best_score = float('inf')
            
            for idx in remaining:
                s = edge_sums[idx]
                p = edge_prods[idx]
                score = 0
                if s in sums_used:
                    score -= 10
                if p in prods_used:
                    score -= 10
                # Prefer edges where both are already used
                if s in sums_used and p in prods_used:
                    score -= 20
                # Tie-break: prefer smaller values
                score += (s + p) / 100000.0
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            sums_used.add(edge_sums[best_idx])
            prods_used.add(edge_prods[best_idx])
        
        # Second pass: try to improve by swapping
        for iteration in range(100):
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(selected)):
                old_idx = selected[i]
                # Try swapping with an unselected edge
                for new_idx in remaining:
                    new_sums = set(sums_used)
                    new_prods = set(prods_used)
                    new_sums.discard(edge_sums[old_idx])
                    new_prods.discard(edge_prods[old_idx])
                    new_sums.add(edge_sums[new_idx])
                    new_prods.add(edge_prods[new_idx])
                    
                    # Check if this improves
                    if len(new_sums) < len(sums_used) or len(new_prods) < len(prods_used):
                        selected[i] = new_idx
                        remaining.remove(new_idx)
                        remaining.append(old_idx)
                        sums_used = new_sums
                        prods_used = new_prods
                        improved = True
                        break
            if not improved:
                break
        
        return [all_edges[i] for i in selected]
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges."""
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                ns, np_, new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values with multiplicative structure focus."""
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(50):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                # Generate candidates focused on multiplicative structure
                candidates = set()
                # Nearby values
                for delta in range(-10, 11):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                # Multiples of common bases
                for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]:
                    for mult in range(1, min(50, max_val // base + 1)):
                        candidates.add(base * mult)
                # Powers
                for base in [2, 3, 4, 5]:
                    v = 1
                    while v <= max_val:
                        candidates.add(v)
                        v *= base
                # Random
                for _ in range(15):
                    candidates.add(rng.randint(1, max_val))
                
                best_new_val = None
                best_new_score = current_score
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in values:
                        continue
                    values[idx] = new_val
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    else:
                        values[idx] = old_val
                        break
                    values[idx] = old_val
                
                if best_new_val is not None:
                    values[idx] = best_new_val
                    current_score = best_new_score
                    improved = True
        
        return values, current_score
    
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(12345)
    
    # Value strategies: focus on numbers with many common factors
    # Key insight: if all values share a common factor d, then all products are multiples of d^2
    # and all sums are multiples of d, reducing distinct values
    value_strategies = [
        # All multiples of 6 (highly composite)
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        # All multiples of 12
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        # All multiples of 4
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        # All multiples of 2
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Mix: powers of 2 and 3
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Arithmetic progression with step 6
        [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79],
        # Small primes and their multiples
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
        # Geometric-like: 2^k * 3^m
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # All even, spread out
        [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106],
        # Multiples of 3
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    ]
    
    # Phase 1: Try each value strategy with greedy edge selection
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
        
        edges = find_edges_for_values(init_values, rng, deadline - 10.0)
        if len(edges) < m:
            # Fill with remaining edges
            used = set(tuple(e) for e in edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if tuple(e) not in used:
                    edges.append(e)
                    used.add(tuple(e))
        
        _, _, score = evaluate(init_values, edges)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Local search on edges
        edges, score_e = local_search_edges(init_values, edges, rng, deadline - 10.0)
        if score_e < best_overall_score:
            best_overall_score = score_e
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # Local search on values
        values = list(init_values)
        values, score_v = local_search_values(values, edges, rng, deadline - 5.0)
        if score_v < best_overall_score:
            best_overall_score = score_v
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Refine edges with new values
        if (time.time() - start_time) < deadline - 3.0:
            edges3, score3 = local_search_edges(values, edges, rng, deadline - 3.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': [list(e) for e in edges3]}
    
    # Phase 2: More aggressive local search on best result
    if best_result is not None:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        for iteration in range(20):
            if (time.time() - start_time) > deadline - 2.0:
                break
            
            # Alternate between value and edge optimization
            values, s1 = local_search_values(values, edges, rng, deadline - 2.0)
            edges, s2 = local_search_edges(values, edges, rng, deadline - 2.0)
            
            current_score = max(s1, s2)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
