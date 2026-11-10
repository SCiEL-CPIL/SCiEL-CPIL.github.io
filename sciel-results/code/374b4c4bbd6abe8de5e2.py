"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    def _smooth_max(sums_set, prods_set):
        """
        Smooth surrogate for max(len(sums), len(prods)) using log-sum-exp.
        This helps in ranking candidates when exact counts are close,
        providing a gradient-like signal for which set is 'tighter'.
        """
        ls = len(sums_set)
        lp = len(prods_set)
        # Use a large temperature to approximate max closely but smoothly
        # Actually, for discrete integers, max is already smooth in the sense of steps.
        # The "smoothness" here is more about the ranking stability.
        # We return a tuple (max, min) to break ties towards balancing.
        return max(ls, lp), min(ls, lp)

    def optimize_edges_exact(values, time_budget=10.0):
        """
        Exact branch-and-bound to find the minimum K such that there exists a set of m edges
        with at most K distinct sums and at most K distinct products.
        
        We iterate K from lower bound to upper bound.
        For a fixed K, we check if a valid set of m edges exists.
        Since m=35 is small relative to total edges (91), and K is likely small,
        we can use a greedy construction guided by the structure of K.
        
        However, exact B&B on 35 edges from 91 is too large for naive backtracking.
        Instead, we use a "Label Covering" approach:
        1. Enumerate candidate sets of S sums and P products (size <= K).
        2. Check if the union of edges having sums in S and products in P contains a matching of size m.
        
        Given the complexity, we use a hybrid:
        - Greedy construction to find a good K0.
        - Local search with strict acceptance (only if max distinct decreases or stays same but sum+prod decreases).
        - This is "exact" in the sense that it explores the neighborhood thoroughly without random noise.
        """
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sum and product for each edge
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i,j)] = values[i] + values[j]
            edge_prod[(i,j)] = values[i] * values[j]
        
        # Map sum -> list of edges
        sum_edges = {}
        prod_edges = {}
        for e in all_edges:
            s = edge_sum[e]
            p = edge_prod[e]
            if s not in sum_edges: sum_edges[s] = []
            if p not in prod_edges: prod_edges[p] = []
            sum_edges[s].append(e)
            prod_edges[p].append(e)
        
        # Initial greedy solution to establish upper bound
        # Sort edges by "rarity" of sum and product (prefer common ones? No, prefer ones that fit into small sets)
        # Actually, for minimizing distinct sums, we want to pick edges that share sums.
        
        best_K = float('inf')
        best_edges = None
        
        # Heuristic Upper Bound
        # Try to pick edges that maximize sum collisions
        sum_groups = sorted(sum_edges.items(), key=lambda x: -len(x[1]))
        current_edges = set()
        for s, edges in sum_groups:
            for e in edges:
                if len(current_edges) >= m:
                    break
                current_edges.add(e)
            if len(current_edges) >= m:
                break
        if len(current_edges) < m:
            for e in all_edges:
                if len(current_edges) >= m: break
                current_edges.add(e)
        
        best_edges = list(current_edges)
        best_K = _raw_value(values, best_edges)
        
        # Local Search Phase (Deterministic)
        # We perform a steepest descent on the objective: max(distinct_sums, distinct_prods)
        # Tie-breaker: min(distinct_sums + distinct_prods)
        
        def get_score(edges_set):
            sums = set()
            prods = set()
            for e in edges_set:
                sums.add(edge_sum[e])
                prods.add(edge_prod[e])
            # Smooth surrogate: use a weighted sum to break ties in favor of balance
            # But primary objective is max.
            return max(len(sums), len(prods)), len(sums) + len(prods)
        
        current_edges = set(best_edges)
        current_score, current_tie = get_score(current_edges)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            # Try all possible swaps: remove one edge, add one other
            # 35 * (91-35) = 1960 swaps per iteration. Very fast.
            for e_out in list(current_edges):
                for e_in in all_edges:
                    if e_in in current_edges:
                        continue
                    
                    # Quick check: if e_in has a sum not in current sums and prod not in current prods,
                    # and e_out is the only edge with that sum/prod, it might increase count.
                    # Just compute score.
                    
                    new_edges = set(current_edges)
                    new_edges.remove(e_out)
                    new_edges.add(e_in)
                    
                    new_score, new_tie = get_score(new_edges)
                    
                    # Accept if better, or same max but better tie (balanced)
                    if (new_score < current_score) or (new_score == current_score and new_tie < current_tie):
                        current_edges = new_edges
                        current_score = new_score
                        current_tie = new_tie
                        improved = True
                        # Restart loop to find new best swap
                        break
                if improved:
                    break
            
            if current_score < best_K:
                best_K = current_score
                best_edges = list(current_edges)
        
        return best_edges, best_K

    # Candidate value sets
    value_sets = [
        list(range(1, 15)),
        list(range(1, 29, 2)),
        [1,2,3,4,5,6,8,9,10,12,15,16,20,24],
        [1,2,3,4,5,6,8,9,10,12,15,18,20,24],
        [1,2,3,4,5,6,8,10,12,15,16,20,24,30],
        [1,2,4,8,3,6,12,24,5,10,20,40,7,14],
        [1,2,3,4,6,8,9,12,16,18,24,27,32,36],
        [1,2,3,5,8,13,21,34,55,89,144,233,377,610],
    ]
    
    # Add some structured random sets
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20: break
        # Biased towards small numbers
        vals = random.sample(range(1, 30), 14)
        if len(set(vals)) == 14:
            value_sets.append(vals)
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    for vs in value_sets:
        if time.time() > deadline - 5:
            break
        
        # Validate
        if len(set(vs)) != 14 or not all(isinstance(v, int) and 1 <= v <= 10000 for v in vs):
            continue
            
        edges, score = optimize_edges_exact(vs, time_budget=5.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vs
            best_overall_edges = edges
            
        # Early stopping if we hit a very low score (theoretical min is likely around 10-15)
        if best_overall_score <= 12:
            break

    # Fallback
    if best_overall_edges is None:
        best_overall_values = list(range(1, 15))
        best_overall_edges = list(combinations(range(14, 2), 2))[:35]
        best_overall_score = _raw_value(best_overall_values, best_overall_edges)

    # Final Validation
    assert len(best_overall_values) == 14
    assert len(set(best_overall_values)) == 14
    assert all(isinstance(v, int) and v > 0 and v <= 10000 for v in best_overall_values)
    assert len(best_overall_edges) == 35
    assert all(len(e) == 2 for e in best_overall_edges)
    assert all(e[0] != e[1] for e in best_overall_edges)
    assert len(set(tuple(sorted(e)) for e in best_overall_edges)) == 35
    
    return {
        'values': list(best_overall_values),
        'edges': [list(e) for e in best_overall_edges]
    }
# EVOLVE-BLOCK-END
