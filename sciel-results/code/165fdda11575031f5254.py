import numpy as np
import time

def construct():
            N = 200
            start_time = time.time()
            max_time = 110.0 # 10s safety margin
            
            best_score = 0
            best_centers = None
            
            # Helper to compute score
            def score(centers):
                # centers: (N, 2)
                # Pairwise distances
                # Use broadcasting for speed
                # d[i, j] = norm(c_i - c_j)
                # This is O(N^2) memory and compute. 200x200 is fine.
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                # Exclude self
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists)
                
                # Wall distances
                # Distance to boundaries: 0.5 - |x|, 0.5 - |y|
                wall_dists = np.minimum(
                    np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])),
                    0.5 # Just in case, though min handles it
                )
                # Actually, dist to wall for point (x,y) is min(0.5-x, x+0.5, 0.5-y, y+0.5)
                # which is min(0.5 - |x|, 0.5 - |y|)
                min_wall = np.min(wall_dists)
                
                return min(min_pair, min_wall)
            
            # Helper to optimize
            def optimize(centers, max_iters=5000):
                centers = centers.copy()
                current_score = score(centers)
                best_local_score = current_score
                best_local_centers = centers.copy()
                
                step_size = 0.01
                for i in range(max_iters):
                    if time.time() - start_time > max_time:
                        break
                    
                    # Compute gradients/forces
                    # Find min pair
                    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dists, np.inf)
                    
                    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                    i_idx, j_idx = min_idx
                    d_min = dists[i_idx, j_idx]
                    
                    # Wall check
                    wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
                    min_wall_idx = np.argmin(wall_dists)
                    d_wall = wall_dists[min_wall_idx]
                    
                    move_made = False
                    
                    if d_min < d_wall:
                        # Repel i and j
                        vec = centers[i_idx] - centers[j_idx]
                        norm = d_min
                        if norm > 1e-10:
                            grad_i = vec / norm
                            grad_j = -vec / norm
                            # Move
                            centers[i_idx] += step_size * grad_i
                            centers[j_idx] += step_size * grad_j
                            move_made = True
                    else:
                        # Repel from wall
                        idx = min_wall_idx
                        # Determine which wall
                        x, y = centers[idx]
                        # dist to left: x - (-0.5) = x + 0.5
                        # dist to right: 0.5 - x
                        # dist to bottom: y + 0.5
                        # dist to top: 0.5 - y
                        dl = x + 0.5
                        dr = 0.5 - x
                        db = y + 0.5
                        dt = 0.5 - y
                        
                        if dl == min(dl, dr, db, dt):
                            centers[idx, 0] += step_size
                        elif dr == min(dl, dr, db, dt):
                            centers[idx, 0] -= step_size
                        elif db == min(dl, dr, db, dt):
                            centers[idx, 1] += step_size
                        else:
                            centers[idx, 1] -= step_size
                        move_made = True

                    if move_made:
                        new_score = score(centers)
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_local_centers = centers.copy()
                            # Increase step size slightly? Or keep constant.
                            # Adaptive step size: if improved, increase, else decrease?
                            # Simple: keep constant.
                        else:
                            # Revert? Or accept with probability? 
                            # For pure local optimization, revert is safer to ensure progress.
                            # But it might get stuck. 
                            # Let's try a simple hill climbing with reversion.
                            pass # Centers are already modified. We need to revert if we want strict hill climbing.
                            # Actually, let's just track the best found.
                            # If we don't revert, we are doing a random walk biased by forces.
                            # This is essentially Simulated Annealing with T=0 (greedy) but with noise if we don't revert?
                            # If we don't revert, we accept all moves. This is a "steepest descent" but with fixed step.
                            # It might oscillate. 
                            # Let's revert if it doesn't improve the *current* score?
                            # No, let's just keep the best found so far.
                            # The loop continues.
                    
                    # Decay step size?
                    # step_size *= 0.999
                    # If step size gets too small, break.
                    if step_size < 1e-6:
                        break
                    step_size *= 0.999

                return best_local_centers, best_local_score

            # Initializations
            # 1. Grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            init1 = grid[:N]
            
            # 2. Grid with noise
            init2 = init1 + np.random.normal(0, 0.01, (N, 2))
            
            # 3. Random
            init3 = np.random.uniform(-0.4, 0.4, (N, 2))
            
            inits = [init1, init2, init3]
            
            for init in inits:
                if time.time() - start_time > max_time:
                    break
                c, s = optimize(init)
                if s > best_score:
                    best_score = s
                    best_centers = c
            
            return best_centers
