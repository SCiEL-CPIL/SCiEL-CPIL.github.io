"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # We want to maximize n. Published construction is 34.
    # Strategy:
    # 1. Use Simulated Annealing with NumPy vectorization for fast evaluation.
    # 2. Try n values from 15 up to 34.
    # 3. For each n, run SA with a time budget.
    # 4. If a valid coloring (0 violations) is found, save it and try n+1.
    # 5. Return the best n found.
    
    best_n = 9
    best_colors = None
    
    # Helper to generate precomputed indices for fast NumPy evaluation
    def get_precomputed_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 triple indices
        k4_triples = []
        for k4 in combinations(range(n), 4):
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32)
        
        # Precompute K5 triple indices
        k5_triples = []
        for k5 in combinations(range(n), 5):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32)
        
        return triples, num_triples, k4_arr, k5_arr

    def count_violations_vec(colors, k4_arr, k5_arr):
        """
        colors: 1D NumPy array of 0s and 1s.
        k4_arr: (num_k4, 4) array of triple indices.
        k5_arr: (num_k5, 10) array of triple indices.
        Returns total number of monochromatic K4 (red) and K5 (blue).
        """
        # Red K4: all 4 triples are 0
        # colors[k4_arr] has shape (num_k4, 4)
        # We want to count rows where all are 0.
        # np.sum(colors[k4_arr], axis=1) == 0
        red_k4_count = np.sum(colors[k4_arr], axis=1)
        red_violations = np.sum(red_k4_count == 0)
        
        # Blue K5: all 10 triples are 1
        # colors[k5_arr] has shape (num_k5, 10)
        # We want to count rows where all are 1.
        # np.sum(colors[k5_arr], axis=1) == 10
        blue_k5_count = np.sum(colors[k5_arr], axis=1)
        blue_violations = np.sum(blue_k5_count == 10)
        
        return int(red_violations + blue_violations)

    # Try increasing n
    # Start from a reasonable lower bound where SA can converge quickly
    # and go up to 34.
    # We allocate more time to larger n as they are harder.
    
    # Fixed time budget per n? Or adaptive?
    # Let's do a linear allocation of the remaining time to the range [15, 34].
    # Total 115s.
    # Let's try n from 15 to 34.
    
    n_start_val = 15
    n_end_val = 34
    
    # If we find a valid coloring for n, we can try n+1.
    # If we fail for n, we should probably move to n+1? 
    # Actually, if we fail for n, it's likely we fail for n+1 too, but sometimes
    # a larger n might be easier due to more degrees of freedom? Unlikely.
    # But let's just iterate. If we find valid for n, we keep it.
    
    # To maximize score, we want the highest valid n.
    # We will iterate n from 15 to 34.
    # If we find a valid coloring for n, we store it.
    # We continue to n+1 to see if we can do better.
    # If we time out, we return the best stored.
    
    # Optimization: If we find a valid coloring for n, we can use it as a 
    # "seed" for n+1 by adding a new vertex and randomizing its edges, 
    # then running SA on the larger graph. This is a powerful strategy.
    
    current_valid_colors = None
    current_valid_n = 0
    
    for n in range(n_start_val, n_end_val + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Determine time budget for this n
        # Simple heuristic: give equal time to each n, but cap total.
        # 20 values of n. 115s / 20 = ~5.75s per n.
        # But we might find valid n early and want to spend more time on larger n.
        # Let's use a fixed 4s per n, but if we have a valid coloring from n-1,
        # we might converge faster.
        
        time_budget = 4.0
        n_start_time = time.time()
        
        # Get precomputed indices
        triples, num_triples, k4_arr, k5_arr = get_precomputed_indices(n)
        
        # Initialize colors
        if current_valid_colors is not None and current_valid_n == n - 1:
            # Extend previous valid coloring
            # We need to map old triples to new indices.
            # This is tricky because adding a vertex changes indices of all triples.
            # Simpler: Just start random. The SA is fast enough.
            # Or, we can try to "embed" the old solution.
            # Given the complexity of index remapping, let's just start random 
            # but use the fact that we know a solution exists for n-1.
            # Actually, let's just start random. It's robust.
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        else:
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            
        # Simulated Annealing
        current_v = count_violations_vec(colors, k4_arr, k5_arr)
        best_v = current_v
        best_cols = colors.copy()
        
        # SA Parameters
        # Initial temperature should be high enough to accept bad moves.
        # Max violations for n=34: C(34,4) + C(34,5) ~ 46k + 278k = 324k.
        # Let's scale T with n.
        T = 50.0
        T_end = 0.1
        cooling_rate = 0.999
        
        # We want to run as many steps as possible.
        # Estimate steps: 10000 steps/sec. 4s = 40000 steps.
        
        no_improve_steps = 0
        max_no_improve = 10000 # Reset T if stuck? Or just stop?
        
        while time.time() < n_start_time + time_budget:
            # Pick random triple
            idx = random.randint(0, num_triples - 1)
            
            # Flip
            old_val = colors[idx]
            colors[idx] = 1 - old_val
            
            new_v = count_violations_vec(colors, k4_arr, k5_arr)
            delta_v = new_v - current_v
            
            if delta_v < 0:
                # Improvement
                current_v = new_v
                if new_v < best_v:
                    best_v = new_v
                    best_cols = colors.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Worse or equal
                if delta_v == 0:
                    # Accept equal moves to explore plateaus
                    current_v = new_v
                    no_improve_steps += 1
                else:
                    # Accept worse move with probability exp(-delta/T)
                    if T > T_end:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            current_v = new_v
                        else:
                            # Revert
                            colors[idx] = old_val
                    else:
                        # Revert
                        colors[idx] = old_val
                        no_improve_steps += 1
            
            # Cool down
            T *= cooling_rate
            
            if best_v == 0:
                break
                
            # Optional: Restart if stuck too long
            if no_improve_steps > max_no_improve:
                # Reset T to try to escape local minimum
                T = 50.0
                no_improve_steps = 0
                # Also randomize some fraction of colors?
                # Let's just keep the current best as the new start if we are stuck?
                # No, just increase T.
                
        if best_v == 0:
            # Found valid coloring for n
            best_n = n
            best_colors = best_cols.tolist()
            current_valid_colors = best_cols
            current_valid_n = n
        else:
            # Did not find valid for n
            # Should we continue to n+1?
            # If n is hard, n+1 is likely harder.
            # But maybe the specific random seed failed.
            # Let's continue, but with less time?
            # Actually, if we can't solve n, we probably can't solve n+1.
            # Let's break if we fail for two consecutive n's?
            # For now, let's just continue to use remaining time.
            # But to be efficient, if we fail for n, we might as well stop?
            # The task asks to maximize n. If we can't get n, we can't get n+1.
            # So if best_v > 0, we should probably stop trying larger n?
            # Wait, is it possible that a valid coloring for n+1 exists but not n?
            # No, if a coloring for n+1 exists, removing one vertex gives a valid coloring for n.
            # So the set of valid n is an interval [min_n, max_n].
            # If we fail to find a valid coloring for n, it doesn't mean none exists,
            # but it means our search failed.
            # However, if we fail for n, it is very likely we will fail for n+1.
            # So we should stop?
            # Let's add a flag: if we fail for n, we break.
            break
            
    # Fallback if we didn't find anything >= 15
    if best_n < 15:
        # Try a quick random search for small n?
        # Or just return the seed solution.
        # The seed solution n=9 is valid.
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    else:
        return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
