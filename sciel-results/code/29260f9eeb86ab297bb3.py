import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))

def _generate_initial_values():
    """Generate a few diverse initial value sets."""
    inits = []
    # 1. Small consecutive
    inits.append(list(range(1, 15)))
    # 2. Random small
    inits.append(random.sample(range(1, 100), 14))
    # 3. Random medium
    inits.append(random.sample(range(1, 1000), 14))
    # 4. Powers of 2 (scaled)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    inits.append([2**i for i in range(14)])
    # 5. Arithmetic progression with step 2
    inits.append([2*i + 1 for i in range(14)])
    return inits

def _generate_initial_edges():
    """Generate a few diverse initial edge sets."""
    inits = []
    all_edges = list(combinations(range(14), 2))
    # 1. First 35
    inits.append([list(e) for e in all_edges[:35]])
    # 2. Random 35
    inits.append([list(e) for e in random.sample(all_edges, 35)])
    # 3. Complete graph minus 21 edges (random)
    inits.append([list(e) for e in random.sample(all_edges, 35)])
    return inits

def _local_search(values, edges, deadline):
    """Perform local search on the given state."""
    current_score = _score(values, edges)
    best_score = current_score
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    all_edges = set(frozenset(e) for e in combinations(range(14), 2))
    current_edge_set = set(frozenset(e) for e in edges)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        move_type = random.choice(['swap_val', 'change_val', 'swap_edge'])
        
        if move_type == 'swap_val':
            i, j = random.sample(range(14), 2)
            values[i], values[j] = values[j], values[i]
            new_score = _score(values, edges)
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_values = values[:]
                    best_edges = [e[:] for e in edges]
            else:
                # Revert
                values[i], values[j] = values[j], values[i]
                no_improve_count += 1
                
        elif move_type == 'change_val':
            i = random.randint(0, 13)
            old_val = values[i]
            # Try a few random new values
            improved = False
            for _ in range(10):
                new_val = random.randint(1, 10000)
                if new_val in values and new_val != old_val:
                    continue
                values[i] = new_val
                new_score = _score(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_values = values[:]
                        best_edges = [e[:] for e in edges]
                    improved = True
                    break
            if not improved:
                values[i] = old_val
                no_improve_count += 1
                
        elif move_type == 'swap_edge':
            # Remove a random edge, add a random non-edge
            e_to_remove = random.choice(edges)
            e_to_remove_set = frozenset(e_to_remove)
            
            # Find a non-edge
            non_edges = list(all_edges - current_edge_set)
            if not non_edges:
                no_improve_count += 1
                continue
                
            e_to_add_set = random.choice(non_edges)
            e_to_add = list(e_to_add_set)
            
            # Apply
            edges.remove(e_to_remove)
            current_edge_set.remove(e_to_remove_set)
            edges.append(e_to_add)
            current_edge_set.add(e_to_add_set)
            
            new_score = _score(values, edges)
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_values = values[:]
                    best_edges = [e[:] for e in edges]
            else:
                # Revert
                edges.remove(e_to_add)
                current_edge_set.remove(e_to_add_set)
                edges.append(e_to_remove)
                current_edge_set.add(e_to_remove_set)
                no_improve_count += 1

    return best_values, best_edges, best_score

def construct():
    deadline = time.time() + 55  # Safety margin
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    initial_values_list = _generate_initial_values()
    initial_edges_list = _generate_initial_edges()
    
    # Try a few combinations of initial states
    # To be efficient, we will interleave restarts
    random.seed(42) # For reproducibility in development, but evaluator might not care? 
    # Actually, better to use random seed based on time or just default random.
    
    # Let's do a few restarts
    num_restarts = 50 # We will check time in the loop
    
    restart_count = 0
    while time.time() < deadline:
        # Pick random initial state
        vals = random.choice(initial_values_list)[:]
        # Ensure distinct
        while len(set(vals)) < 14:
            idx = random.randint(0, 13)
            vals[idx] = random.randint(1, 10000)
            
        edges = random.choice(initial_edges_list)
        # Ensure 35 unique edges
        # The generator ensures this, but let's be safe if we modify it
        
        # Run local search for a limited time or iterations
        # We'll run local search until it hits its internal deadline or no improve
        # To allow multiple restarts, we limit the time per restart
        restart_deadline = time.time() + 10 # 10 seconds per restart max
        
        v, e, s = _local_search(vals, edges, restart_deadline)
        
        if s < best_overall_score:
            best_overall_score = s
            best_overall_values = v
            best_overall_edges = e
            
        restart_count += 1
        
        # If we found a very good score, we might want to polish it
        # But for now, just keep restarting if time permits
        
    # Final polish on the best found
    if best_overall_values is not None:
        # Run local search again on the best found with remaining time
        final_deadline = time.time() + 5
        v, e, s = _local_search(best_overall_values, best_overall_edges, final_deadline)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_values = v
            best_overall_edges = e

    # Ensure types are correct
    return {
        'values': [int(x) for x in best_overall_values],
        'edges': [[int(i), int(j)] for i, j in best_overall_edges]
    }
