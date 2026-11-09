import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin from 150s limit
    
    best_score = 0
    best_solution = None
    
    def get_solution_from_params(params, w, h, shrink_factor=1.0):
        """Extract and validate solution from parameters"""
        x = params[0::3] * w
        y = params[1::3] * h
        r = params[2::3] * shrink_factor
        
        # Check validity
        # Boundary
        if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
            return None, 0
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-8:
                    return None, 0
        
        sum_r = np.sum(r)
        return np.column_stack((x, y, r)), sum_r

    def simulate(w, h, max_steps=1000, dt=0.01, temp_start=1.0, temp_end=0.01):
        """Physics-based simulation with variable aspect ratio"""
        nonlocal best_score, best_solution
        
        # Initialize positions and radii
        x = np.random.uniform(0.2, 0.8, N) * w
        y = np.random.uniform(0.2, 0.8, N) * h
        r = np.random.uniform(0.05, 0.15, N)
        
        # Ensure initial validity by shrinking
        for _ in range(100):
            valid = True
            for i in range(N):
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                    valid = False
                    break
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            r *= 0.9
            
        if not valid:
            return None, 0
            
        current_score = np.sum(r)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Cooling schedule
            temp = temp_start * (temp_end / temp_start) ** (step / max_steps)
            
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Repulsion between circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist == 0:
                        dx = 0.01
                        dist = 0.01
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        force = 10.0 * (min_dist - dist) / dist
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
                    else:
                        # Weak attraction to keep them close but not overlapping
                        force = 0.1 * (dist - min_dist) / dist
                        fx[i] -= force * dx
                        fy[i] -= force * dy
                        fx[j] += force * dx
                        fy[j] += force * dy
                        
            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 50.0 * (r[i] - x[i])
                elif x[i] > w - r[i]:
                    fx[i] -= 50.0 * (x[i] - (w - r[i]))
                if y[i] < r[i]:
                    fy[i] += 50.0 * (r[i] - y[i])
                elif y[i] > h - r[i]:
                    fy[i] -= 50.0 * (y[i] - (h - r[i]))
                    
                # Radius growth: try to grow if not touching anything
                # Simple heuristic: grow proportional to local free space
                fr[i] = 0.1 * temp # Slow growth
            
            # Update positions
            x += fx * dt + np.random.normal(0, temp, N)
            y += fy * dt + np.random.normal(0, temp, N)
            r += fr * dt + np.random.normal(0, temp * 0.1, N)
            
            # Clamp
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0.01, min(w, h) / 2)
            
            # Check validity and score
            valid = True
            for i in range(N):
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
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
            
            if valid:
                score = np.sum(r)
                if score > current_score:
                    current_score = score
                    if score > best_score:
                        best_score = score
                        # Normalize to [0,1] for storage, but keep w,h for reconstruction
                        best_solution = (x.copy(), y.copy(), r.copy(), w, h)
            else:
                # Revert to last valid state if we have one
                if best_solution is not None:
                    x_prev, y_prev, r_prev, w_prev, h_prev = best_solution
                    # We can't easily revert mid-simulation without storing history, 
                    # so just let the forces push it back.
                    pass
                    
        return best_solution, best_score

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.7, 1.3, 1.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        # Perimeter = 2(w+h) = 4 => w+h = 2
        # w/h = ar => w = ar*h => h(ar+1) = 2 => h = 2/(ar+1), w = 2ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Run simulation
        sol, score = simulate(w, h, max_steps=2000)
        
        if sol is not None and score > best_score:
            best_score = score
            # sol is already (x, y, r, w, h)
            
    # Final refinement: if we have a solution, try to grow radii further
    if best_solution is not None:
        x, y, r, w, h = best_solution
        
        # Local search to grow radii
        for _ in range(100):
            if time.time() > deadline - 2:
                break
                
            improved = False
            for i in range(N):
                old_r = r[i]
                # Try to grow
                new_r = old_r * 1.01
                r[i] = new_r
                
                valid = True
                # Check boundary
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                    valid = False
                else:
                    # Check overlap
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-8:
                            valid = False
                            break
                            
                if valid:
                    improved = True
                    best_score = np.sum(r)
                else:
                    r[i] = old_r
                    
            if not improved:
                break
                
        best_solution = np.column_stack((x, y, r))
        
    # If no valid solution found, return a simple grid
    if best_solution is None or not isinstance(best_solution, np.ndarray):
        # Simple 5x5 grid with 21 circles
        side = 5
        x_list, y_list, r_list = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / side)
                y_list.append((i + 0.5) / side)
                r_list.append(0.5 / side - 0.01)
        
        best_solution = np.column_stack((x_list, y_list, r_list))
        
    return best_solution
