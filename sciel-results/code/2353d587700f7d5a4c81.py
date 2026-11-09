import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin from 150s limit
    
    def objective(params, w, h):
        """
        Minimize negative sum of radii subject to constraints.
        Uses a penalty method for soft constraints to allow gradient-based optimization.
        params: [x0, y0, r0, x1, y1, r1, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Objective: Maximize sum of radii => Minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        pen = 0.0
        # x boundaries
        pen += np.sum(np.maximum(0, r - x)**2)
        pen += np.sum(np.maximum(0, x + r - w)**2)
        # y boundaries
        pen += np.sum(np.maximum(0, r - y)**2)
        pen += np.sum(np.maximum(0, y + r - h)**2)
        
        # Penalty for overlaps
        # dist < ri + rj => (dist - (ri+rj)) < 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2 * 100.0 # Strong penalty for overlap
        
        return obj + pen

    def run_optimization(w, h, x0, y0, r0):
        """Run L-BFGS-B optimization from a specific start"""
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds:
        # x: [0, w], y: [0, h], r: [0, min(w,h)/2]
        # Note: L-BFGS-B bounds are (lower, upper) for each variable
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))      # x
            bounds.append((0.0, h))      # y
            bounds.append((0.0, min(w, h)/2)) # r
            
        # We rely on the penalty function to keep things valid, 
        # but strict bounds prevent extreme values.
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        except Exception:
            return None, 0.0
            
        x = res.x[0::3]
        y = res.x[1::3]
        r = res.x[2::3]
        
        # Final validation: ensure strict non-overlap and boundary compliance
        # We need to shrink slightly if there are tiny violations due to numerical precision
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-8 or x[i] > w - r[i] + 1e-8 or y[i] < r[i] - 1e-8 or y[i] > h - r[i] + 1e-8:
                valid = False
                break
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-8:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Attempt repair: shrink all radii by a small factor until valid
            scale = 1.0
            for _ in range(100):
                r_new = r * scale
                ok = True
                for i in range(N):
                    if x[i] < r_new[i] - 1e-8 or x[i] > w - r_new[i] + 1e-8 or y[i] < r_new[i] - 1e-8 or y[i] > h - r_new[i] + 1e-8:
                        ok = False
                        break
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_new[i] + r_new[j] - 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    r = r_new
                    valid = True
                    break
                scale *= 0.95
                
            if not valid:
                return None, 0.0

        score = np.sum(r)
        return np.column_stack((x, y, r)), score

    best_solution = None
    best_score = 0.0
    
    # Define several aspect ratios to explore
    # Perimeter 2(w+h)=4 => w+h=2
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.7, 1.3, 1.7, 0.6, 2.5]
    
    # Shuffle to mix up order, but keep deterministic seed for reproducibility if needed
    random.seed(42)
    random.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Generate multiple starting points
        # 1. Grid-like start
        # 2. Random start
        starts = []
        
        # Start 1: Grid
        side = 5
        x_grid = np.array([(j + 0.5) / side for j in range(5) for i in range(5)][:N]) * w
        y_grid = np.array([(i + 0.5) / side for j in range(5) for i in range(5)][:N]) * h
        r_grid = np.full(N, 0.1) # Small initial radius
        starts.append((x_grid, y_grid, r_grid))
        
        # Start 2: Random
        for _ in range(2):
            x_rand = np.random.uniform(0.1, 0.9, N) * w
            y_rand = np.random.uniform(0.1, 0.9, N) * h
            r_rand = np.full(N, 0.05)
            starts.append((x_rand, y_rand, r_rand))
            
        # Start 3: Hexagonal-ish packing heuristic
        # Try to place circles in a pattern that might pack better
        x_hex = np.zeros(N)
        y_hex = np.zeros(N)
        r_hex = np.full(N, 0.1)
        
        # Simple hex packing logic for initialization
        cols = 5
        rows = 5
        spacing = 0.18
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                x_pos = (col + 0.5) * spacing
                y_pos = (row + 0.5) * spacing
                if row % 2 == 1:
                    x_pos += spacing * 0.5
                x_hex[idx] = x_pos
                y_hex[idx] = y_pos
                idx += 1
        
        # Scale to fit w,h roughly
        max_x = np.max(x_hex)
        max_y = np.max(y_hex)
        if max_x > 0: x_hex *= (w * 0.8) / max_x
        if max_y > 0: y_hex *= (h * 0.8) / max_y
        starts.append((x_hex, y_hex, r_hex))

        for x0, y0, r0 in starts:
            if time.time() > deadline - 5:
                break
                
            sol, score = run_optimization(w, h, x0, y0, r0)
            if sol is not None and score > best_score:
                best_score = score
                best_solution = sol.copy()
                
    # Final local refinement: try to increase radii slightly if possible
    if best_solution is not None:
        x = best_solution[:, 0]
        y = best_solution[:, 1]
        r = best_solution[:, 2]
        
        # Determine w,h from the best solution? 
        # We need to know which w,h was used. 
        # Since we iterate over aspect ratios, we should store w,h with best_solution.
        # Let's restructure slightly to track w,h.
        
        # Actually, the loop above updates best_solution but not w,h. 
        # I will modify the loop to store w,h.
        
        pass # The logic below assumes we have w,h. Let's do a quick re-run or store it.
        
    # Re-doing the loop structure to properly track w,h for the final result
    best_solution_final = None
    best_score_final = 0.0
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        starts = []
        
        # Start 1: Grid
        side = 5
        x_grid = np.array([(j + 0.5) / side for j in range(5) for i in range(5)][:N]) * w
        y_grid = np.array([(i + 0.5) / side for j in range(5) for i in range(5)][:N]) * h
        r_grid = np.full(N, 0.1)
        starts.append((x_grid, y_grid, r_grid))
        
        # Start 2: Random
        for _ in range(2):
            x_rand = np.random.uniform(0.1, 0.9, N) * w
            y_rand = np.random.uniform(0.1, 0.9, N) * h
            r_rand = np.full(N, 0.05)
            starts.append((x_rand, y_rand, r_rand))
            
        # Start 3: Hexagonal-ish
        x_hex = np.zeros(N)
        y_hex = np.zeros(N)
        r_hex = np.full(N, 0.1)
        cols = 5
        rows = 5
        spacing = 0.18
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                x_pos = (col + 0.5) * spacing
                y_pos = (row + 0.5) * spacing
                if row % 2 == 1:
                    x_pos += spacing * 0.5
                x_hex[idx] = x_pos
                y_hex[idx] = y_pos
                idx += 1
        max_x = np.max(x_hex)
        max_y = np.max(y_hex)
        if max_x > 0: x_hex *= (w * 0.8) / max_x
        if max_y > 0: y_hex *= (h * 0.8) / max_y
        starts.append((x_hex, y_hex, r_hex))

        for x0, y0, r0 in starts:
            if time.time() > deadline - 5:
                break
                
            sol, score = run_optimization(w, h, x0, y0, r0)
            if sol is not None and score > best_score_final:
                best_score_final = score
                best_solution_final = sol.copy()
                
    if best_solution_final is None:
        # Fallback: simple grid
        side = 5
        x_list, y_list, r_list = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / side)
                y_list.append((i + 0.5) / side)
                r_list.append(0.5 / side - 0.01)
        best_solution_final = np.column_stack((x_list, y_list, r_list))
        
    return best_solution_final
