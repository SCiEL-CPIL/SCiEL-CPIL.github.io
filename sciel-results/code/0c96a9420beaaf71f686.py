"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    # Initialize with random values and a random set of edges
    def random_init():
        values = random.sample(range(1, max_val + 1), n)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:m]
        return values, edges
    
    best_values, best_edges = random_init()
    best_score = evaluate(best_values, best_edges)
    
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 100.0
    alpha = 0.995
    min_T = 0.01
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while T > min_T and time.time() < deadline:
        # Generate a neighbor
        new_values = current_values[:]
        new_edges = current_edges[:]
        
        # Randomly decide what to perturb
        if random.random() < 0.5:
            # Perturb values
            idx = random.randint(0, n - 1)
            new_values[idx] = random.randint(1, max_val)
        else:
            # Perturb edges
            # Swap an existing edge with a non-existing edge
            if len(current_edges) > 0:
                edge_idx = random.randint(0, len(current_edges) - 1)
                old_edge = current_edges[edge_idx]
                
                # Find a candidate edge to swap in
                # To avoid O(m * (m-n)) checks every time, we can just pick a random pair
                # and check if it's in the set.
                # Optimization: maintain a set of current edges for O(1) lookup?
                # For n=14, m=35, total edges=91. Set lookup is fast.
                edge_set = set(tuple(e) for e in current_edges)
                
                attempts = 0
                while attempts < 10:
                    u, v = random.choice(all_possible_edges)
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        new_edges[edge_idx] = (u, v)
                        break
                    attempts += 1
                else:
                    # If we couldn't find a valid swap, just skip this move
                    continue
        
        new_score = evaluate(new_values, new_edges)
        
        # Acceptance criterion
        if new_score < current_score:
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values[:]
                best_edges = current_edges[:]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
        
        T *= alpha
        
    # Final local search polish (greedy)
    # Try to improve the best solution found
    if time.time() < deadline - 5:
        for _ in range(1000):
            if time.time() > deadline - 2: break
            
            # Try changing one value
            improved = False
            for i in range(n):
                old_val = best_values[i]
                # Try a few random values
                for _ in range(5):
                    new_val = random.randint(1, max_val)
                    if new_val == old_val: continue
                    test_vals = best_values[:]
                    test_vals[i] = new_val
                    score = evaluate(test_vals, best_edges)
                    if score < best_score:
                        best_values[i] = new_val
                        best_score = score
                        improved = True
                        break
                if improved: break
            
            # Try swapping one edge
            if not improved:
                for _ in range(5):
                    if time.time() > deadline - 2: break
                    edge_idx = random.randint(0, len(best_edges) - 1)
                    old_edge = best_edges[edge_idx]
                    edge_set = set(tuple(e) for e in best_edges)
                    
                    # Find a replacement
                    u, v = random.choice(all_possible_edges)
                    if (u, v) in edge_set or (v, u) in edge_set:
                        continue
                    
                    test_edges = best_edges[:]
                    test_edges[edge_idx] = (u, v)
                    score = evaluate(best_values, test_edges)
                    
                    if score < best_score:
                        best_edges[edge_idx] = (u, v)
                        best_score = score
                        break
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
