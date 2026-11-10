"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def _count_sums_prods(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return len(sums), len(prods)
    
    best_result = None
    best_score = float('inf')
    
    # Generate candidate value sets: small base values with multiplicity
    # Key insight: values from small base sets create many sum/product collisions
    
    candidate_values = []
    
    # Base set {1,2,3,4} with various multiplicities
    base_sets = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 4, 5],
    ]
    
    for base in base_sets:
        # Generate all multiplicity distributions
        for c1 in range(1, 9):
            for c2 in range(1, 9):
                for c3 in range(1, 9):
                    for c4 in range(1, 9):
                        if len(base) >= 4:
                            total = c1 + c2 + c3 + c4
                            if total == n:
                                vals = []
                                for v, c in zip(base[:4], [c1, c2, c3, c4]):
                                    vals.extend([v] * c)
                                if len(vals) == n:
                                    candidate_values.append(sorted(vals))
                        
                        if len(base) >= 5:
                            for c5 in range(1, 9):
                                total = c1 + c2 + c3 + c4 + c5
                                if total == n:
                                    vals = []
                                    for v, c in zip(base[:5], [c1, c2, c3, c4, c5]):
                                        vals.extend([v] * c)
                                    if len(vals) == n:
                                        candidate_values.append(sorted(vals))
    
    # Deduplicate
    candidate_values = list(set(tuple(v) for v in candidate_values))
    candidate_values = [list(v) for v in candidate_values]
    
    # Add some random small-value sets
    random.seed(123)
    for _ in range(30):
        vals = [random.randint(1, 8) for _ in range(n)]
        candidate_values.append(sorted(vals))
    
    for vals_tuple in candidate_values:
        if time.time() > deadline - 3.0:
            break
        
        vals = list(vals_tuple)
        
        # Precompute sum and product for all edges
        all_edges = list(combinations(range(n), 2))
        edge_sums = {}
        edge_prods = {}
        for i, j in all_edges:
            edge_sums[(i, j)] = vals[i] + vals[j]
            edge_prods[(i, j)] = vals[i] * vals[j]
        
        # Joint greedy: pick edges to minimize max(distinct sums, distinct products)
        # Strategy: at each step, pick the edge that minimizes the increase in max(sums, prods)
        
        def greedy_select(vals, edge_sums, edge_prods, all_edges):
            selected = set()
            edges = []
            sums = set()
            prods = set()
            
            for _ in range(m):
                if time.time() > deadline - 3.0:
                    break
                best_edge = None
                best_score_local = float('inf')
                
                for e in all_edges:
                    if e in selected:
                        continue
                    s = edge_sums[e]
                    p = edge_prods[e]
                    new_sums = sums | {s}
                    new_prods = prods | {p}
                    score = max(len(new_sums), len(new_prods))
                    if score < best_score_local:
                        best_score_local = score
                        best_edge = e
                
                if best_edge is None:
                    break
                
                selected.add(best_edge)
                edges.append(best_edge)
                sums.add(edge_sums[best_edge])
                prods.add(edge_prods[best_edge])
            
            return edges, max(len(sums), len(prods))
        
        edges_greedy, score_greedy = greedy_select(vals, edge_sums, edge_prods, all_edges)
        
        if len(edges_greedy) == m and score_greedy < best_score:
            best_score = score_greedy
            best_result = {'values': list(vals), 'edges': [[i, j] for i, j in edges_greedy]}
        
        # Simulated annealing to improve
        if time.time() > deadline - 3.0:
            continue
        
        if len(edges_greedy) == m:
            edges_sa = [list(e) for e in edges_greedy]
            current_score = score_greedy
            current_sums, current_prods = _count_sums_prods(vals, edges_sa)
            
            # Precompute edge index
            edge_set = set(map(tuple, edges_sa))
            
            for step in range(200):
                if time.time() > deadline - 3.0:
                    break
                
                # Pick a random edge to replace
                idx = random.randint(0, m - 1)
                old_edge = tuple(edges_sa[idx])
                
                # Pick a random new edge
                new_edge = random.choice(all_edges)
                if new_edge in edge_set:
                    continue
                
                # Compute new score
                new_sums = set()
                new_prods = set()
                for i, j in edges_sa:
                    if (i, j) == old_edge:
                        continue
                    new_sums.add(vals[i] + vals[j])
                    new_prods.add(vals[i] * vals[j])
                new_sums.add(vals[new_edge[0]] + vals[new_edge[1]])
                new_prods.add(vals[new_edge[0]] * vals[new_edge[1]])
                
                new_score = max(len(new_sums), len(new_prods))
                
                if new_score < current_score:
                    edges_sa[idx] = list(new_edge)
                    edge_set.remove(old_edge)
                    edge_set.add(new_edge)
                    current_score = new_score
                    current_sums, current_prods = len(new_sums), len(new_prods)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_result = {'values': list(vals), 'edges': [list(e) for e in edges_sa]}
                else:
                    # Accept with some probability (annealing)
                    if random.random() < 0.1:
                        edges_sa[idx] = list(new_edge)
                        edge_set.remove(old_edge)
                        edge_set.add(new_edge)
                        current_score = new_score
                        current_sums, current_prods = len(new_sums), len(new_prods)
    
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
