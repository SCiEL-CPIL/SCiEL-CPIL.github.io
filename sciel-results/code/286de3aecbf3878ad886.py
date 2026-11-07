import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 55  # Leave 5s margin for safety
    
    n = 14
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_cost(vals, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))
    
    best_cost = float('inf')
    best_values = None
    best_edges = None
    
    # We can try a few restarts if time permits
    # But a single long annealing run is usually better than many short ones
    # Let's do one long run with a good schedule.
    
    current_values = list(range(1, n + 1))
    current_edges = set(random.sample(all_possible_edges, 35))
    current_cost = get_cost(current_values, current_edges)
    
    T = 5.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    while time.time() - start_time < timeout and T > min_T:
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap two values
            i, j = random.sample(range(n), 2)
            new_values = current_values[:]
            new_values[i], new_values[j] = new_values[j], new_values[i]
            new_edges = current_edges
            new_cost = get_cost(new_values, new_edges)
        else:
            # Swap one edge
            non_current = [e for e in all_possible_edges if e not in current_edges]
            if not non_current:
                continue
            edge_out = random.choice(list(current_edges))
            edge_in = random.choice(non_current)
            
            new_edges = set(current_edges)
            new_edges.remove(edge_out)
            new_edges.add(edge_in)
            new_values = current_values
            new_cost = get_cost(new_values, new_edges)
        
        delta = new_cost - current_cost
        
        if delta <= 0:
            current_values = new_values
            current_edges = new_edges
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_values = current_values[:]
                best_edges = set(current_edges)
        else:
            # Accept worse move
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_values = new_values
                    current_edges = new_edges
                    current_cost = new_cost
        
        T *= cooling_rate

    # Fallback if something went wrong
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = set(random.sample(all_possible_edges, 35))
        
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validation
    if len(final_values) != 14 or len(set(final_values)) != 14:
        final_values = list(range(1, 15))
    if len(final_edges) != 35:
        final_edges = [list(e) for e in random.sample(all_possible_edges, 35)]
        
    return {'values': final_values, 'edges': final_edges}
