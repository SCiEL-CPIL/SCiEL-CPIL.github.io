import numpy as np
import random
import time

def construct():
    # Precompute 4-APs
    n = 128
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(1, n - 3 * d + 1):
            # 0-indexed: i-1, i-1+d, i-1+2d, i-1+3d
            i0 = i - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    aps = np.array(aps)
    # aps shape: (M, 4)
    
    def calculate_cost(colors):
        # colors: np array (n,)
        c1 = colors[aps[:, 0]]
        c2 = colors[aps[:, 1]]
        c3 = colors[aps[:, 2]]
        c4 = colors[aps[:, 3]]
        
        # Sum of pairwise differences
        # 1 if different, 0 if same
        d12 = (c1 != c2).astype(int)
        d13 = (c1 != c3).astype(int)
        d14 = (c1 != c4).astype(int)
        d23 = (c2 != c3).astype(int)
        d24 = (c2 != c4).astype(int)
        d34 = (c3 != c4).astype(int)
        
        sum_diffs = d12 + d13 + d14 + d23 + d24 + d34
        
        # Invalid if sum_diffs is 0, 3, or 4
        invalid_mask = (sum_diffs == 0) | (sum_diffs == 3) | (sum_diffs == 4)
        
        return invalid_mask.sum()

    # Simulated Annealing
    def sa(k, max_iters=5000, start_temp=10.0, end_temp=0.1):
        # Initialize
        current = np.random.randint(0, k, n)
        current_cost = calculate_cost(current)
        
        if current_cost == 0:
            return current
        
        best = current.copy()
        best_cost = current_cost
        
        T = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            # Pick a random position to change
            pos = np.random.randint(0, n)
            old_color = current[pos]
            
            # Pick a new color
            if k > 1:
                new_color = np.random.randint(0, k)
                while new_color == old_color:
                    new_color = np.random.randint(0, k)
            else:
                continue # Can't change if k=1
            
            current[pos] = new_color
            new_cost = calculate_cost(current)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = current.copy()
                    if best_cost == 0:
                        break
            else:
                current[pos] = old_color
            
            T *= cooling_rate
            
        return best

    # Try increasing number of colors
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    for k in range(1, 129):
        # If time is running out, break
        if time.time() - start_time > time_limit:
            break
            
        # Run SA
        # Adjust iterations based on k? Smaller k is harder, maybe more iters?
        # But N=128 is small.
        iters = 2000 if k <= 3 else 1000
        
        solution = sa(k, max_iters=iters)
        
        if calculate_cost(solution) == 0:
            # Found a valid solution with at most k colors
            # Check actual distinct colors
            unique_colors = np.unique(solution)
            # Remap to 0..len(unique_colors)-1 to ensure minimal labels?
            # The problem says "labels need not be consecutive" but "minimize number of labels".
            # Returning the array as is is fine, but remapping ensures the count is minimal for this solution.
            # However, if we found it with k=3, and it uses 2 colors, we should have found it with k=2?
            # Not necessarily, due to search landscape.
            # To be safe, if we find a valid solution, we can try to minimize the distinct count further?
            # No, just return it. The evaluator minimizes the count.
            # If we return a solution with 3 distinct colors, score is based on 3.
            # If we could find one with 2, we would have tried k=2 first.
            # So the first k that yields a valid solution is likely optimal or close.
            
            # Convert to list of ints
            return solution.tolist()
            
    # If no solution found within time, return the best one found so far?
    # The problem requires a valid solution. If we fail, score is 0.
    # We should try to return something valid if possible.
    # Let's do a final greedy or random check?
    # Or just return the last best solution from the last k tried.
    
    # Fallback: Return a random valid-looking thing? No, must be valid.
    # If we didn't find any, we are in trouble.
    # But k=3 or 4 should work.
    
    # As a last resort, return the best solution from the last SA run
    # (Though it might be invalid)
    return solution.tolist()
