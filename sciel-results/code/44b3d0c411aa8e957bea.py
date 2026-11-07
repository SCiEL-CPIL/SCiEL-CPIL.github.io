import random
import time

def construct():
    start_time = time.time()
    timeout = 55  # Leave 5s margin for safety
    
    # Pre-compute all possible edges for 14 vertices
    all_possible_edges = [(i, j) for i in range(14) for j in range(i+1, 14)]
    
    # Initialize with random values (1-14 is a good start, but we allow swapping)
    # and random edges
    current_values = list(range(1, 15))
    current_edges = set(random.sample(all_possible_edges, 35))
    
    def get_cost(vals, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))
    
    current_cost = get_cost(current_values, current_edges)
    best_cost = current_cost
    best_values = current_values[:]
    best_edges = set(current_edges)
    
    # Local Search Loop
    while time.time() - start_time < timeout:
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap two values
            i, j = random.sample(range(14), 2)
            new_values = current_values[:]
            new_values[i], new_values[j] = new_values[j], new_values[i]
            new_edges = current_edges
            new_cost = get_cost(new_values, new_edges)
        else:
            # Swap one edge: remove one, add one
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
        
        # Accept move if it improves or ties (to escape plateaus)
        if new_cost <= current_cost:
            current_values = new_values
            current_edges = new_edges
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_values = current_values[:]
                best_edges = set(current_edges)
                
    # Ensure correct types and structure
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validation
    if len(final_values) != 14 or len(set(final_values)) != 14:
        final_values = list(range(1, 15))
    if len(final_edges) != 35:
        final_edges = [list(e) for e in random.sample(all_possible_edges, 35)]
        
    return {'values': final_values, 'edges': final_edges}
