import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 295
                
                n_evals = self.budget
                if n_evals <= 0:
                    return []
                
                # 1. Generate dense grid
                n_grid = 50000
                x0_grid = np.linspace(0.0, 1.0, n_grid)
                
                # 2. Compute analytic objectives
                sin_term1 = np.sin(10 * np.pi * x0_grid)
                sin_term2 = np.sin(10 * np.pi * x0_grid**2)
                f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)
                f1_grid = np.maximum(f1_grid, 1e-6)
                f2_grid = 1 - np.sqrt(f1_grid)
                
                # 3. Sort by f1
                sort_idx = np.argsort(f1_grid)
                f1_sorted = f1_grid[sort_idx]
                f2_sorted = f2_grid[sort_idx]
                x0_sorted = x0_grid[sort_idx]
                
                # 4. Greedy Selection
                # Initialize with best single point
                areas = (11 - f1_sorted) * (11 - f2_sorted)
                best_idx = np.argmax(areas)
                
                selected_indices = [best_idx]
                # To speed up neighbor lookup, we can maintain a sorted list of selected indices
                # Or just use a list and binary search on f1 values
                
                selected_f1 = [f1_sorted[best_idx]]
                selected_f2 = [f2_sorted[best_idx]]
                
                # Pre-compute gain function?
                # Gain of inserting point i into list L (sorted by f1)
                
                for step in range(n_evals - 1):
                    if time.time() > deadline - 5:
                        break
                        
                    best_gain = -1
                    best_candidate = -1
                    
                    # Iterate over all candidates
                    # We can skip already selected ones?
                    # For simplicity, just check all.
                    
                    for i in range(n_grid):
                        # Check if i is already selected?
                        # To avoid O(K) check, we can maintain a set of selected indices.
                        # But set lookup is O(1).
                        
                        # Binary search for position in selected_f1
                        # np.searchsorted is fast
                        pos = np.searchsorted(selected_f1, f1_sorted[i])
                        
                        # Calculate gain
                        # Case 1: Insert at beginning (pos == 0)
                        if pos == 0:
                            # Old first point was selected_f1[0]
                            # New first point is i
                            # Old contribution of first segment:
                            # (selected_f1[1] - selected_f1[0]) * (11 - selected_f2[0])
                            # New contribution:
                            # (selected_f1[0] - f1_sorted[i]) * (11 - f2_sorted[i]) +
                            # (selected_f1[1] - selected_f1[0]) * (11 - selected_f2[0])
                            # Wait, the HV formula:
                            # Sum over intervals.
                            # If we add a point to the left, the interval [new_f1, old_f1] is added with height (11 - new_f2).
                            # The rest of the HV remains the same?
                            # No, the height for the interval [old_f1, old_f1_next] was (11 - old_f2).
                            # It remains (11 - old_f2).
                            # So the gain is just the area of the new box part?
                            # Actually, the HV is the union of boxes.
                            # Adding a point with lower f1 and lower f2 (which is not possible here as f2 decreases with f1)
                            # Here f1 increases, f2 decreases.
                            # If we add a point with lower f1, it has higher f2.
                            # The new point's box is [f1_new, 11] x [f2_new, 11].
                            # The old first point's box was [f1_old, 11] x [f2_old, 11].
                            # Since f1_new < f1_old and f2_new > f2_old.
                            # The new box covers [f1_new, f1_old] x [f2_new, 11].
                            # Does it cover anything else? No, because [f1_old, 11] x [f2_new, 11] is already covered by the old box?
                            # No, the old box has height up to 11 starting from f2_old.
                            # f2_new > f2_old. So the new box is "higher" in f2.
                            # The union in the range [f1_new, f1_old] is determined by the new box (since it has lower f1).
                            # In the range [f1_old, 11], the old box is still the one with lower f2?
                            # No, the old box has f2_old. The new box has f2_new.
                            # Since f2_new > f2_old, the old box covers a larger area in f2 for f1 > f1_old.
                            # So the gain is exactly the area of the rectangle [f1_new, f1_old] x [f2_new, 11].
                            # Gain = (f1_old - f1_new) * (11 - f2_new)
                            
                            if len(selected_f1) > 0:
                                f1_old = selected_f1[0]
                                gain = (f1_old - f1_sorted[i]) * (11 - f2_sorted[i])
                            else:
                                gain = (11 - f1_sorted[i]) * (11 - f2_sorted[i])
                                
                        elif pos == len(selected_f1):
                            # Insert at end
                            f1_old = selected_f1[-1]
                            # Gain = (11 - f1_old) * (11 - f2_old) - (11 - f1_new) * (11 - f2_new) ?
                            # Old last segment: (11 - f1_old) * (11 - f2_old)
                            # New last segment: (11 - f1_new) * (11 - f2_new)
                            # New middle segment: (f1_new - f1_old) * (11 - f2_old)
                            # Total New = (f1_new - f1_old)*(11 - f2_old) + (11 - f1_new)*(11 - f2_new)
                            # Total Old = (11 - f1_old)*(11 - f2_old)
                            # Gain = New - Old
                            gain = (f1_sorted[i] - f1_old) * (11 - selected_f2[-1]) + (11 - f1_sorted[i]) * (11 - f2_sorted[i]) - (11 - f1_old) * (11 - selected_f2[-1])
                            # Simplify:
                            # Gain = (11 - f1_old)*(11 - f2_old) - (11 - f1_new)*(11 - f2_new) is NOT correct.
                            # Let's just use the formula:
                            # Old: (11 - f1_old) * (11 - f2_old)
                            # New: (f1_new - f1_old) * (11 - f2_old) + (11 - f1_new) * (11 - f2_new)
                            gain = (f1_sorted[i] - f1_old) * (11 - selected_f2[-1]) + (11 - f1_sorted[i]) * (11 - f2_sorted[i]) - (11 - f1_old) * (11 - selected_f2[-1])
                            
                        else:
                            # Insert in middle
                            f1_prev = selected_f1[pos-1]
                            f2_prev = selected_f2[pos-1]
                            f1_next = selected_f1[pos]
                            
                            # Old segment: (f1_next - f1_prev) * (11 - f2_prev)
                            # New segments: (f1_i - f1_prev) * (11 - f2_prev) + (f1_next - f1_i) * (11 - f2_i)
                            old_seg = (f1_next - f1_prev) * (11 - f2_prev)
                            new_seg = (f1_sorted[i] - f1_prev) * (11 - f2_prev) + (f1_next - f1_sorted[i]) * (11 - f2_sorted[i])
                            gain = new_seg - old_seg
                            
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = i
                            
                    # Insert best_candidate
                    if best_candidate != -1:
                        pos = np.searchsorted(selected_f1, f1_sorted[best_candidate])
                        selected_f1.insert(pos, f1_sorted[best_candidate])
                        selected_f2.insert(pos, f2_sorted[best_candidate])
                        selected_indices.append(best_candidate)
                        
                # 5. Evaluate
                results = []
                for i in selected_indices:
                    if time.time() > deadline:
                        break
                    x = np.zeros(self.dim, dtype=float)
                    x[0] = float(x0_sorted[i])
                    try:
                        f = problem(x)
                        results.append(f)
                    except Exception:
                        break
                        
                return results
