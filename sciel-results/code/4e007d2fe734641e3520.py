"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using constrained random search with shrinking radius."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        
        # Use the global random state (seeded by evaluator) for reproducibility
        rng = np.random
        
        # Population size: use a reasonable chunk of the budget
        # WFG9 has deceptive regions, so we need decent coverage
        pop_size = min(50, self.budget // 4)
        if pop_size < 10:
            pop_size = 10
        
        # Initialize with random points in the hypercube
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(self.dim)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points to compute hypervolume at the end
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Main loop: constrained random search with shrinking radius
        # The Pareto front of WFG9 is near the origin (in decision space)
        # We sample within a hyper-sphere centered at center, with shrinking radius
        
        n_remaining = self.budget - used
        if n_remaining > 0:
            # Divide remaining budget into batches for adaptive radius
            # Start with large radius (full space), shrink to small radius
            # Use a few batches to allow adaptation
            
            # Strategy: 
            # - First 50% of remaining budget: radius from full_range to 0.5*full_range
            # - Next 30%: radius from 0.5*full_range to 0.2*full_range  
            # - Last 20%: radius from 0.2*full_range to 0.05*full_range
            
            full_radius = np.sqrt(np.sum(half_range ** 2))
            
            # Define radius schedule
            n_batches = 5
            batch_sizes = [int(n_remaining * w) for w in [0.3, 0.25, 0.2, 0.15, 0.1]]
            # Adjust to ensure sum equals n_remaining
            diff = n_remaining - sum(batch_sizes)
            batch_sizes[0] += diff
            
            radius_schedule = [
                full_radius * 1.0,
                full_radius * 0.6,
                full_radius * 0.3,
                full_radius * 0.15,
                full_radius * 0.05
            ]
            
            for batch_idx, batch_size in enumerate(batch_sizes):
                if batch_size <= 0:
                    continue
                if time.time() > deadline:
                    break
                
                radius = radius_schedule[batch_idx]
                
                # Generate batch_size random points within the hyper-sphere
                batch_X = np.zeros((batch_size, self.dim))
                for i in range(batch_size):
                    # Sample direction uniformly on sphere
                    dir_vec = rng.standard_normal(self.dim)
                    norm = np.linalg.norm(dir_vec)
                    if norm < 1e-10:
                        continue
                    dir_vec /= norm
                    
                    # Sample radius with bias toward the center (use power distribution)
                    # This helps focus search near the Pareto front
                    r = radius * rng.random() ** 0.5  # Sqrt bias toward center
                    
                    point = center + dir_vec * r
                    point = np.clip(point, lo, hi)
                    batch_X[i] = point
                
                # Evaluate the batch
                batch_F = np.array([problem(x) for x in batch_X])
                used += batch_size
                
                all_X.append(batch_X)
                all_F.append(batch_F)
                
                # Check deadline
                if time.time() > deadline:
                    break
        
        # Combine all points and compute non-dominated front
        if all_X:
            X_all = np.vstack(all_X)
            F_all = np.vstack(all_F)
        else:
            X_all = X
            F_all = F
        
        # Find non-dominated points for the final result
        # Use vectorized dominance check
        n = len(F_all)
        dominated = np.zeros(n, dtype=bool)
        
        # Vectorized dominance: for each point, check if any other point dominates it
        # F_all[i] is dominated by F_all[j] if F_all[j] <= F_all[i] and F_all[j] < F_all[i]
        # We can do this in chunks to avoid memory issues
        chunk_size = min(100, n)
        
        for i in range(0, n, chunk_size):
            j_end = min(i + chunk_size, n)
            if dominated[i:j_end].all():
                continue
            
            F_chunk = F_all[i:j_end]
            F_rest = F_all[j_end:]
            
            # Check dominance from rest
            if len(F_rest) > 0:
                # For each point in chunk, check if dominated by any in rest
                # F_rest[k] dominates F_chunk[m] if all(F_rest[k] <= F_chunk[m]) and any(F_rest[k] < F_chunk[m])
                for k in range(len(F_rest)):
                    if dominated[j_end + k]:
                        continue
                    fr = F_rest[k]
                    # Vectorized check for the chunk
                    le = np.all(F_chunk <= fr, axis=1)  # Wait, this is wrong direction
                    # F_rest[k] <= F_chunk[m] means F_chunk[m] >= F_rest[k]
                    # Let's redo: F_rest[k] dominates F_chunk[m] if F_rest[k] <= F_chunk[m] elementwise
                    # i.e., F_chunk[m] >= F_rest[k] elementwise, and at least one strict
                    ge = np.all(F_chunk >= fr, axis=1)
                    gt = np.any(F_chunk > fr, axis=1)
                    dom_mask = ge & gt
                    if np.any(dom_mask):
                        dominated[i:j_end][dom_mask] = True
            
            # Check dominance within chunk
            for m in range(len(F_chunk)):
                if dominated[i + m]:
                    continue
                fc = F_chunk[m]
                # Check against other points in chunk
                for p in range(len(F_chunk)):
                    if p == m or dominated[i + p]:
                        continue
                    fp = F_chunk[p]
                    if np.all(fp <= fc) and np.any(fp < fc):
                        dominated[i + m] = True
                        break
        
        # Also check if points in later chunks dominate points in earlier chunks
        # This is already handled by the loop above if we process in order
        # But we need to ensure that if point i is dominated by point j (j > i), it's marked
        # The current loop handles this because when we process chunk containing j,
        # we check F_rest which includes earlier points... wait, no.
        
        # Let me redo this more carefully with a simpler approach
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            fi = F_all[i]
            # Check if any other point dominates fi
            # Vectorized: check all other points
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                fj = F_all[j]
                if np.all(fj <= fi) and np.any(fj < fi):
                    dominated[i] = True
                    break
        
        # Return non-dominated points
        if not dominated.all():
            X_non_dom = X_all[~dominated]
            F_non_dom = F_all[~dominated]
        else:
            # Fallback: return all points
            X_non_dom = X_all
            F_non_dom = F_all
        
        return X_non_dom
# EVOLVE-BLOCK-END
