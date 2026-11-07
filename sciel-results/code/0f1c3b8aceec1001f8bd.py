import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def get_score(values, edges):
        """Calculate the raw score: max(distinct_sums, distinct_prods)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def joint_sa(initial_values, initial_edges, max_iter=5000, T_start=1.0, T_end=0.05):
        """
        Joint Simulated Annealing for values and edges.
        """
        # State: values (list of ints), edges (set of tuples)
        # We need to ensure values are distinct and in [1, 10000]
        
        current_values = list(initial_values)
        current_edges = set(initial_edges)
        
        # Precompute edge properties for current values
        # This is expensive if values change often, but n=14, m=35 is small.
        # We will recompute sum/prod for the 35 active edges on every evaluation.
        # 35 additions + 35 multiplications is very fast.
        
        current_score = get_score(current_values, current_edges)
        
        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)
        
        # Precompute all possible edge indices for random selection
        edge_list = all_edges
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Decide move type: Value perturbation or Edge perturbation
            # Values are harder to optimize, so maybe weight towards edges?
            # Or 50/50. Let's try 50/50 first.
            move_type = random.random()
            
            new_values = current_values
            new_edges = current_edges
            
            if move_type < 0.5:
                # Edge Move: Swap one edge for another
                if len(current_edges) == 0:
                    continue
                    
                # Pick edge to remove
                e_out = random.choice(list(current_edges))
                # Pick edge to add (not in current set)
                # To avoid infinite loops if set is full (it's not, 35 < 91)
                while True:
                    e_in = random.choice(edge_list)
                    if e_in not in current_edges:
                        break
                        
                new_edges = (current_edges - {e_out}) | {e_in}
            else:
                # Value Move: Perturb one or two values
                # Strategy: Pick a random index, change value slightly or swap with another
                idx = random.randint(0, n - 1)
                # Option A: Small random step
                # Option B: Swap with another index
                # Option C: Random new value in range
                
                # To maintain distinctness and range, swapping is safer and keeps "structure"
                # Let's try swapping two indices
                if random.random() < 0.7:
                    idx2 = random.randint(0, n - 1)
                    if idx2 == idx:
                        continue
                    new_values = list(current_values)
                    new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
                else:
                    # Random walk: add small delta
                    delta = random.randint(-5, 5)
                    new_val = current_values[idx] + delta
                    if 1 <= new_val <= 10000 and new_val not in current_values:
                        new_values = list(current_values)
                        new_values[idx] = new_val
                    else:
                        continue

            # Evaluate new state
            new_score = get_score(new_values, new_edges)
            
            delta_score = new_score - current_score
            
            # Acceptance criterion
            if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
                    best_edges = set(current_edges)
                    
                    # If we hit a very good score, we might want to cool down faster?
                    # Standard SA schedule is fine for now.
                    
        return best_score, best_values, best_edges

    # Initial Strategy
    # 1. Generate a few diverse initial configurations
    # 2. Run joint SA on each
    
    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    # Seed for reproducibility in this run? No, random is fine, but let's fix it for consistency in testing
    random.seed(42)
    
    # Initial Values: Random distinct integers
    # We want to encourage collisions.
    # Small numbers tend to have more sum/product collisions?
    # Let's try ranges: [1, 50], [1, 100], [1, 500]
    
    initial_configs = []
    
    # Config 1: Small range
    v1 = random.sample(range(1, 51), n)
    e1 = set(random.sample(all_edges, m))
    initial_configs.append((v1, e1))
    
    # Config 2: Medium range
    v2 = random.sample(range(1, 101), n)
    e2 = set(random.sample(all_edges, m))
    initial_configs.append((v2, e2))
    
    # Config 3: Arithmetic progression (structured)
    v3 = [i * 10 + 1 for i in range(n)]
    e3 = set(random.sample(all_edges, m))
    initial_configs.append((v3, e3))
    
    # Config 4: Powers of 2 (structured for products)
    v4 = [2**i for i in range(n)] # 1, 2, 4, ... 4096
    if max(v4) <= 10000:
        e4 = set(random.sample(all_edges, m))
        initial_configs.append((v4, e4))
        
    # Config 5: Random large
    v5 = random.sample(range(1, 10001), n)
    e5 = set(random.sample(all_edges, m))
    initial_configs.append((v5, e5))

    # Run SA for each config
    # Time budget per config: ~15 seconds
    # Total 5 configs * 15s = 75s. Fits in 85s.
    
    time_per_config = (deadline - time.time()) / len(initial_configs)
    
    for vals, edges in initial_configs:
        if time.time() > deadline:
            break
            
        # Determine max_iter based on time budget
        # Each iteration is ~ O(35) for eval + overhead.
        # Let's estimate 10000 iters/sec? 
        # 15s -> 150k iters? Might be too many.
        # Let's try 20000 iters per config.
        
        max_iter = int(time_per_config * 1000) # Rough estimate
        # Cap it to avoid hanging if time_per_config is huge
        max_iter = min(max_iter, 50000)
        # Floor it
        max_iter = max(max_iter, 1000)
        
        score, best_v, best_e = joint_sa(vals, edges, max_iter=max_iter, T_start=1.0, T_end=0.01)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = best_v
            best_global_edges = best_e
            
    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)
        
    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
