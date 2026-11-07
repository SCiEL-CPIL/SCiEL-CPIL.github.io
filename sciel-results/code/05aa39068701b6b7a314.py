import random
import itertools
import time

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of distinct sums and products.
    Uses greedy edge selection followed by local search refinement.
    """
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(10, n + 10)))
    
    # 2. Multiples
    candidates.append([2 * i for i in range(1, n + 1)])
    candidates.append([3 * i for i in range(1, n + 1)])
    candidates.append([4 * i for i in range(1, n + 1)])
    candidates.append([6 * i for i in range(1, n + 1)])
    
    # 3. Structured set with many factor collisions
    candidates.append([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48])
    
    # 4. Random sets with bias towards small numbers
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 51), n)
        candidates.append(vals)

    best_score = float('inf')
    best_edges = None
    best_values = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_select_edges(values):
        all_edges = list(itertools.combinations(range(n), 2))
        
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_cost = float('inf')
            
            for edge in remaining_edges:
                i, j = edge
                s = values[i] + values[j]
                p = values[i] * values[j]
                
                new_sums_count = len(current_sums) + (1 if s not in current_sums else 0)
                new_prods_count = len(current_prods) + (1 if p not in current_prods else 0)
                
                cost = max(new_sums_count, new_prods_count)
                
                if cost < best_cost:
                    best_cost = cost
                    best_edge = edge
            
            if best_edge is None:
                break
                
            i, j = best_edge
            current_sums.add(values[i] + values[j])
            current_prods.add(values[i] * values[j])
            selected_edges.append(best_edge)
            remaining_edges.remove(best_edge)
            
        return selected_edges, best_cost

    for values in candidates:
        if len(set(values)) != n or max(values) > 10000 or min(values) < 1:
            continue
            
        edges, score = greedy_select_edges(values)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values
            
        if best_score <= 15:
            break

    # Local search refinement on the best found
    if best_edges is not None:
        current_edges = set(best_edges)
        current_score = best_score
        all_edges = list(itertools.combinations(range(n), 2))
        
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < 5:
            improved = False
            edges_list = list(current_edges)
            for i_in in range(len(edges_list)):
                edge_in = edges_list[i_in]
                while True:
                    edge_out = tuple(sorted(random.sample(range(n), 2)))
                    if edge_out not in current_edges:
                        break
                
                new_edges = set(current_edges)
                new_edges.remove(edge_in)
                new_edges.add(edge_out)
                
                new_score = evaluate(best_values, new_edges)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)

    if best_edges is None:
        best_values = list(range(1, n + 1))
        all_edges = list(itertools.combinations(range(n), 2))
        best_edges = all_edges[:m]

    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
