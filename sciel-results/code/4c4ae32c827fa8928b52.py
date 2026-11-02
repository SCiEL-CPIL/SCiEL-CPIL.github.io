"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Helper: compute validity and score for a batch of circles and aspect
    def fast_eval(circles, aspect):
        """
        circles: (N, 3) array [x, y, r]
        aspect: scalar w/h
        Returns: (score, is_valid)
        """
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Boundary check
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return 0.0, False
        
        # Pairwise distance check (vectorized)
        # diff[i, j] = distance between i and j
        # We only need upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dists = np.sqrt(dx*dx + dy*dy)
        min_dists = dists - (rs[i_idx] + rs[j_idx])
        
        if np.any(min_dists < -1e-9):
            return 0.0, False
            
        return float(np.sum(rs)), True

    def make_grid_state(aspect, rows, cols):
        """Create a grid-like initial state."""
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        circles = np.zeros((N, 3))
        # Fill rows x cols grid
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                x = (c + 0.5) / cols * w
                y = (r + 0.5) / rows * h
                # Max radius limited by grid spacing and walls
                r_max = min(w/cols, h/rows, w/2, h/2) * 0.45
                circles[count, 0] = x
                circles[count, 1] = y
                circles[count, 2] = r_max
                count += 1
        return circles, aspect

    def make_random_state(aspect):
        """Create a random initial state."""
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        circles = np.zeros((N, 3))
        # Random positions
        # Start with small radii to ensure validity
        r_min = 0.01
        r_max_init = 0.1
        circles[:, 0] = np.random.uniform(r_max_init, w - r_max_init, N)
        circles[:, 1] = np.random.uniform(r_max_init, h - r_max_init, N)
        circles[:, 2] = np.random.uniform(r_min, r_max_init, N)
        return circles, aspect

    def repair_and_grow(circles, aspect, max_iters=50):
        """
        Local search: push apart overlapping circles, then grow radii.
        Returns improved circles and aspect.
        """
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        circles = circles.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            score, valid = fast_eval(circles, aspect)
            if valid:
                # Try to grow radii
                improved = False
                for i in range(N):
                    old_r = circles[i, 2]
                    # Binary search for max radius? Or just step.
                    # Simple step
                    new_r = old_r + 0.005
                    # Check bounds
                    if new_r > circles[i, 0] - 1e-6 or new_r > w - circles[i, 0] - 1e-6 or \
                       new_r > circles[i, 1] - 1e-6 or new_r > h - circles[i, 1] - 1e-6:
                        continue
                    circles[i, 2] = new_r
                    s, v = fast_eval(circles, aspect)
                    if v:
                        improved = True
                    else:
                        circles[i, 2] = old_r
                if not improved:
                    break
            else:
                # Find most violating pair and push apart
                xs = circles[:, 0]
                ys = circles[:, 1]
                rs = circles[:, 2]
                i_idx, j_idx = np.triu_indices(N, k=1)
                dx = xs[i_idx] - xs[j_idx]
                dy = ys[i_idx] - ys[j_idx]
                dists = np.sqrt(dx*dx + dy*dy)
                min_dists = dists - (rs[i_idx] + rs[j_idx])
                
                min_val = np.min(min_dists)
                if min_val >= -1e-9:
                    # Only boundary violations?
                    # Check bounds
                    bound_viol = 0
                    for i in range(N):
                        if xs[i] < rs[i] - 1e-9: bound_viol += 1
                        if xs[i] > w - rs[i] + 1e-9: bound_viol += 1
                        if ys[i] < rs[i] - 1e-9: bound_viol += 1
                        if ys[i] > h - rs[i] + 1e-9: bound_viol += 1
                    if bound_viol == 0:
                        break # Should be valid?
                    
                    # Push away from boundary
                    for i in range(N):
                        if xs[i] < rs[i]:
                            circles[i, 0] = rs[i]
                        if xs[i] > w - rs[i]:
                            circles[i, 0] = w - rs[i]
                        if ys[i] < rs[i]:
                            circles[i, 1] = rs[i]
                        if ys[i] > h - rs[i]:
                            circles[i, 1] = h - rs[i]
                    continue

                # Find index of min violation
                min_idx = np.argmin(min_dists)
                i = i_idx[min_idx]
                j = j_idx[min_idx]
                
                # Vector from i to j
                vx = xs[j] - xs[i]
                vy = ys[j] - ys[i]
                dist = np.sqrt(vx*vx + vy*vy)
                if dist < 1e-9:
                    vx, vy = 1.0, 0.0
                    dist = 1.0
                else:
                    vx /= dist
                    vy /= dist
                
                # Required separation
                req = rs[i] + rs[j]
                # Current separation is dist
                # Push each by half the deficit
                deficit = req - dist
                push = deficit * 0.5 + 0.001
                
                circles[i, 0] -= vx * push
                circles[i, 1] -= vy * push
                circles[j, 0] += vx * push
                circles[j, 1] += vy * push
                
                # Clamp to bounds
                for idx in [i, j]:
                    circles[idx, 0] = np.clip(circles[idx, 0], circles[idx, 2], w - circles[idx, 2])
                    circles[idx, 1] = np.clip(circles[idx, 1], circles[idx, 2], h - circles[idx, 2])

        return circles, aspect

    # Population setup
    POP_SIZE = 20
    population = []
    
    # Seed with diverse grids and aspects
    aspects = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 0.3]
    layouts = [
        (3, 7), (7, 3), (4, 5), (5, 4), (6, 4), (4, 6), (2, 10), (10, 2), (21, 1), (1, 21)
    ]
    
    for i in range(POP_SIZE):
        aspect = aspects[i % len(aspects)]
        if i < len(layouts):
            r, c = layouts[i]
            circles, asp = make_grid_state(aspect, r, c)
        else:
            circles, asp = make_random_state(aspect)
        
        # Initial repair
        circles, asp = repair_and_grow(circles, asp, max_iters=20)
        score, valid = fast_eval(circles, asp)
        if not valid:
            score = 0.0
        population.append((circles, asp, score))
    
    best_score = max(p[2] for p in population)
    best_circles, best_aspect = max(population, key=lambda x: x[2])[0].copy(), max(population, key=lambda x: x[2])[1]

    # Evolutionary Loop
    # We will mutate individuals, repair them, and select the best.
    # This is essentially a (mu+lambda) EA with repair.
    
    mutation_rate_pos = 0.05
    mutation_rate_rad = 0.05
    mutation_rate_aspect = 0.1
    
    while time.time() < deadline - 10.0:
        new_population = []
        
        # Generate offspring
        for _ in range(POP_SIZE):
            # Select parent (tournament of 3)
            idxs = np.random.choice(len(population), 3, replace=False)
            parent_idx = idxs[np.argmax([population[i][2] for i in idxs])]
            parent = population[parent_idx]
            circles, aspect, _ = parent[0].copy(), parent[1], parent[2]
            
            # Mutate
            if np.random.random() < mutation_rate_aspect:
                aspect *= (1.0 + np.random.normal(0, 0.1))
                aspect = np.clip(aspect, 0.2, 5.0)
                # Rescale positions
                old_w = 2.0 * parent[1] / (1.0 + parent[1])
                old_h = 2.0 / (1.0 + parent[1])
                new_w = 2.0 * aspect / (1.0 + aspect)
                new_h = 2.0 / (1.0 + aspect)
                circles[:, 0] *= new_w / old_w
                circles[:, 1] *= new_h / old_h
            
            # Mutate positions
            if np.random.random() < mutation_rate_pos:
                mask = np.random.random(N) < 0.2 # Mutate 20% of circles
                circles[mask, 0] += np.random.normal(0, 0.02, np.sum(mask))
                circles[mask, 1] += np.random.normal(0, 0.02, np.sum(mask))
            
            # Mutate radii
            if np.random.random() < mutation_rate_rad:
                mask = np.random.random(N) < 0.2
                circles[mask, 2] *= (1.0 + np.random.normal(0, 0.1, np.sum(mask)))
                circles[:, 2] = np.clip(circles[:, 2], 0.01, 1.0)
            
            # Repair
            circles, aspect = repair_and_grow(circles, aspect, max_iters=10)
            
            score, valid = fast_eval(circles, aspect)
            if not valid:
                score = 0.0
            
            new_population.append((circles, aspect, score))
            
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                best_aspect = aspect
        
        # Selection: Keep top POP_SIZE from union of old and new
        all_pop = population + new_population
        all_pop.sort(key=lambda x: x[2], reverse=True)
        population = all_pop[:POP_SIZE]
        
        # Occasionally inject a fresh random/grid to maintain diversity
        if np.random.random() < 0.05:
            aspect = np.random.uniform(0.3, 3.0)
            if np.random.random() < 0.5:
                r = np.random.randint(2, 8)
                c = max(1, N // r)
                circles, asp = make_grid_state(aspect, r, c)
            else:
                circles, asp = make_random_state(aspect)
            circles, asp = repair_and_grow(circles, asp, max_iters=20)
            score, valid = fast_eval(circles, asp)
            if not valid: score = 0.0
            population[-1] = (circles, asp, score)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                best_aspect = asp

    # Final Polish on best solution
    if best_circles is not None:
        # More aggressive repair and growth
        best_circles, best_aspect = repair_and_grow(best_circles, best_aspect, max_iters=100)
        s, v = fast_eval(best_circles, best_aspect)
        if v and s > best_score:
            best_score = s

    # Final clamp and return
    w = 2.0 * best_aspect / (1.0 + best_aspect)
    h = 2.0 / (1.0 + best_aspect)
    
    for i in range(N):
        r = min(best_circles[i, 2], best_circles[i, 0] - 1e-6, w - best_circles[i, 0] - 1e-6,
                best_circles[i, 1] - 1e-6, h - best_circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        best_circles[i, 2] = r
    
    return best_circles
# EVOLVE-BLOCK-END
