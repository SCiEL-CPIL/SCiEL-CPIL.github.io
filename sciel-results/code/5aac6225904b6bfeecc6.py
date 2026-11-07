"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    num_possible_edges = len(all_edges)
    
    # Helper to evaluate a candidate (values, edges) using NumPy for speed
    def evaluate(values, edges):
        # values: list of ints
        # edges: list of [u, v]
        # Convert to numpy arrays for vectorized ops
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        # Get values for u and v
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        # Count distinct
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    # Helper to get random distinct values
    def random_values():
        # Random sample of 14 distinct integers from 1 to 10000
        return random.sample(range(1, max_val + 1), n)
    
    # Helper to get random 35 edges
    def random_edges():
        return random.sample(all_edges, m)
    
    # Initialize best solution
    # Start with a structured guess: geometric progression often creates distinct products
    # but we want collisions, so maybe small integers are better for collisions?
    # Let's start with random and let SA find the structure.
    current_values = random_values()
    current_edges = random_edges()
    current_score = evaluate(current_values, current_edges)
    
    best_values = current_values.copy()
    best_edges = [list(e) for e in current_edges]
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    # Main Loop
    while time.time() < deadline and T > T_min:
        # 1. Generate a neighbor
        # Strategy: Perturb either values or edges
        if random.random() < 0.5:
            # Perturb Values: Change 1 or 2 random values
            new_values = current_values.copy()
            num_changes = random.randint(1, 2)
            for _ in range(num_changes):
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, max_val)
                # Ensure distinctness
                while new_val in new_values:
                    new_val = random.randint(1, max_val)
                new_values[idx] = new_val
            
            new_edges = current_edges
        else:
            # Perturb Edges: Swap one edge with a random other edge
            new_values = current_values
            new_edges = current_edges.copy()
            if len(new_edges) < 2:
                continue
                
            # Pick an edge to remove
            rem_idx = random.randint(0, m - 1)
            rem_edge = new_edges[rem_idx]
            
            # Pick a new edge to add (must not be already in list)
            # To be efficient, just pick a random edge from all_possible and check if it's the one we removed or already present
            # Since m=35 and total=91, probability of collision is low but possible.
            # Simple approach: sample until we find a valid new edge not in the set.
            # But checking membership in a list is O(m).
            # Let's just swap with a random edge from all_edges.
            add_edge = random.choice(all_edges)
            
            # If add_edge is already in new_edges (and is not rem_edge), we might end up with duplicates or less than m edges if we just replace.
            # Better: Remove rem_edge, add add_edge. If add_edge == rem_edge, no change.
            # If add_edge is already in new_edges (at index != rem_idx), then after removing rem_edge, we have m-1 edges, and adding add_edge brings it back to m, but add_edge is already there? No, we removed rem_edge.
            # Wait, if add_edge is in new_edges at index k != rem_idx:
            # new_edges becomes: [ ... add_edge at k ... rem_edge at rem_idx ... ]
            # Remove rem_idx -> [ ... add_edge at k ... ] (size m-1)
            # Add add_edge -> [ ... add_edge at k ... add_edge ] (size m, but duplicate add_edge)
            # This is invalid.
            
            # Correct Swap Logic:
            # 1. Remove rem_edge from list.
            # 2. Generate add_edge.
            # 3. If add_edge is in the remaining list, try again or pick another.
            
            # Efficient way:
            # Convert to set for O(1) lookup? m=35, so O(m) is fine.
            # Let's just do:
            new_edges_set = set(map(tuple, new_edges))
            new_edges_set.remove(tuple(rem_edge))
            
            # Sample new edge
            valid_add = False
            for _ in range(10): # Try a few times
                cand = random.choice(all_edges)
                if cand not in new_edges_set:
                    new_edges_set.add(cand)
                    valid_add = True
                    break
            
            if not valid_add:
                # Restore original
                new_edges_set.add(tuple(rem_edge))
                # Just skip this iteration or use original
                continue
            
            new_edges = [list(e) for e in new_edges_set]
            # Ensure length is m
            if len(new_edges) != m:
                 # Should not happen if logic is correct, but safety
                 new_edges = current_edges

        # 2. Evaluate Neighbor
        new_score = evaluate(new_values, new_edges)
        
        # 3. Acceptance Probability
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values.copy()
                best_edges = [list(e) for e in current_edges]
        
        # 4. Cool down
        T *= cooling_rate
        
        # Optional: Restart if stuck?
        # For now, simple SA.
        
    # Final Result
    # Ensure types are correct
    res = {
        'values': [int(v) for v in best_values],
        'edges': [[int(u), int(v)] for u, v in best_edges]
    }
    
    return res

# EVOLVE-BLOCK-END
