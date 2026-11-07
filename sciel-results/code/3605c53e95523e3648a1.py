"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91

    def evaluate(values, edges):
        """Calculate the score: max(number of distinct sums, number of distinct products)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def optimize_edges(values, edges, rng, time_limit):
        """
        Optimize the set of edges for a fixed set of values.
        Uses greedy initialization followed by local search.
        """
        current_edges = list(edges)
        edges_set = set(tuple(e) for e in current_edges)
        
        # Greedy Initialization
        greedy_edges = []
        greedy_set = set()
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        # Precompute sums and products for all edges to speed up greedy
        edge_props = {}
        for e in all_edges:
            u, v = e
            edge_props[e] = (values[u] + values[v], values[u] * values[v])
            
        sums_seen = set()
        prods_seen = set()
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_new_sums = 0
            best_new_prods = 0
            
            # Sample a subset if remaining is large, otherwise check all
            # 91 edges is small enough to check all
            for edge in remaining:
                s, p = edge_props[edge]
                new_s = 1 if s not in sums_seen else 0
                new_p = 1 if p not in prods_seen else 0
                
                # We want to minimize the max of the final counts.
                # Approximate score: max(len(sums_seen)+new_s, len(prods_seen)+new_p)
                score = max(len(sums_seen) + new_s, len(prods_seen) + new_p)
                
                if score < best_score:
                    best_score = score
                    best_edge = edge
                    best_new_sums = new_s
                    best_new_prods = new_p
            
            if best_edge is None:
                break
                
            greedy_edges.append(best_edge)
            greedy_set.add(tuple(best_edge))
            remaining.remove(best_edge)
            
            s, p = edge_props[best_edge]
            sums_seen.add(s)
            prods_seen.add(p)
            
        current_edges = greedy_edges
        edges_set = set(tuple(e) for e in current_edges)
        current_score = evaluate(values, current_edges)

        # Local Search
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try multiple random swaps
            for _ in range(100):
                if (time.time() - start_time) > time_limit:
                    break
                    
                # Pick a random edge to remove
                i = rng.randint(0, len(current_edges) - 1)
                old_edge = current_edges[i]
                
                # Pick a random edge to add (not in current set)
                # To avoid picking from the small remaining set repeatedly, 
                # we can pick from all_edges and check membership
                new_edge = rng.choice(all_edges)
                
                if tuple(new_edge) in edges_set:
                    continue
                
                # Perform swap
                current_edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                new_score = evaluate(values, current_edges)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    current_edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
                    
        return current_edges, current_score

    def optimize_values(values, edges, rng, time_limit):
        """
        Optimize the values for a fixed set of edges.
        Uses targeted perturbation based on existing sums/products.
        """
        current_values = list(values)
        current_score = evaluate(current_values, edges)
        
        # Precompute neighbors for each vertex
        neighbors = {i: [] for i in range(n)}
        for u, v in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
            
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Get current sums and products to target
            current_sums = set()
            current_prods = set()
            for u, v in edges:
                current_sums.add(current_values[u] + current_values[v])
                current_prods.add(current_values[u] * current_values[v])
                
            # Try to improve each vertex
            for i in range(n):
                if (time.time() - start_time) > time_limit:
                    break
                    
                old_val = current_values[i]
                best_val = old_val
                best_score = current_score
                
                candidates = set()
                
                # 1. Simple perturbations
                for delta in [-2, -1, 1, 2, 5, 10]:
                    candidates.add(old_val + delta)
                    
                # 2. Targeted candidates based on neighbors and existing sums/prods
                for j in neighbors[i]:
                    v_j = current_values[j]
                    
                    # Try to match an existing sum
                    # v_i + v_j = S_existing => v_i = S_existing - v_j
                    for s in current_sums:
                        cand = s - v_j
                        if 1 <= cand <= max_val:
                            candidates.add(cand)
                            
                    # Try to match an existing product
                    # v_i * v_j = P_existing => v_i = P_existing / v_j
                    if v_j != 0:
                        for p in current_prods:
                            if p % v_j == 0:
                                cand = p // v_j
                                if 1 <= cand <= max_val:
                                    candidates.add(cand)
                
                # 3. Random candidates
                candidates.add(rng.randint(1, max_val))
                candidates.add(rng.randint(1, 100))
                
                # Filter candidates
                valid_candidates = []
                for cand in candidates:
                    if cand == old_val:
                        continue
                    if cand < 1 or cand > max_val:
                        continue
                    # Check distinctness
                    if cand in current_values:
                        continue
                    valid_candidates.append(cand)
                    
                # Evaluate candidates
                # Limit number of candidates to evaluate to avoid timeout
                if len(valid_candidates) > 50:
                    valid_candidates = rng.sample(valid_candidates, 50)
                
                for new_val in valid_candidates:
                    current_values[i] = new_val
                    new_score = evaluate(current_values, edges)
                    if new_score < best_score:
                        best_score = new_score
                        best_val = new_val
                        improved = True
                    
                current_values[i] = best_val
                
                if improved:
                    current_score = best_score
                    # Update sums/prods if we improved? 
                    # It's cheaper to just recompute in the next outer loop iteration 
                    # or at the start of the next vertex iteration.
                    # Since we break on improved, we restart the vertex loop?
                    # No, we continue to next vertex, but the 'current_sums' is stale.
                    # To be safe and simple, if we improved, we break out of vertex loop 
                    # and restart the while loop to refresh sums/prods.
                    break

        return current_values, current_score

    # Main search loop
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Initial value strategies
    value_strategies = [
        list(range(1, 15)),
        list(range(2, 16)),
        [2**i for i in range(14)],
        [100 + i for i in range(14)],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Random distinct values
        sorted(rng.sample(range(1, 10000), 14)),
        sorted(rng.sample(range(1, 1000), 14)),
    ]
    
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline:
            break
        
        # Ensure distinct and valid
        if len(set(init_values)) != n or any(v < 1 or v > max_val for v in init_values):
            continue
            
        current_values = list(init_values)
        
        # 1. Optimize edges
        current_edges, score_e = optimize_edges(current_values, list(all_edges[:m]), rng, deadline - 10.0)
        
        if score_e < best_overall_score:
            best_overall_score = score_e
            best_result = {'values': list(current_values), 'edges': current_edges}
            
        # 2. Optimize values
        if (time.time() - start_time) < deadline - 5.0:
            new_values, score_v = optimize_values(current_values, current_edges, rng, deadline - 5.0)
            if score_v < best_overall_score:
                best_overall_score = score_v
                best_result = {'values': list(new_values), 'edges': current_edges}
                current_values = new_values
                
        # 3. Re-optimize edges with new values
        if (time.time() - start_time) < deadline - 2.0:
            new_edges, score_e2 = optimize_edges(current_values, current_edges, rng, deadline - 2.0)
            if score_e2 < best_overall_score:
                best_overall_score = score_e2
                best_result = {'values': list(current_values), 'edges': new_edges}
                
    # Final polish: Alternate optimization
    if best_result is not None and (time.time() - start_time) < deadline - 2.0:
        values = list(best_result['values'])
        edges = list(best_result['edges'])
        
        for _ in range(5):
            if (time.time() - start_time) > deadline - 2.0:
                break
            
            # Optimize values
            values, s1 = optimize_values(values, edges, rng, deadline - 2.0)
            if s1 < best_overall_score:
                best_overall_score = s1
                best_result = {'values': list(values), 'edges': edges}
                
            # Optimize edges
            edges, s2 = optimize_edges(values, edges, rng, deadline - 2.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': edges}

    if best_result is None:
        # Fallback
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
