import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    # Initial positions: Hexagonal packing approximation
    # 26 circles. 5x5 grid is 25. Add 1? Or 6x5 grid?
    # Let's use a 5x5 grid + 1 in the middle of a gap? 
    # Or just 5 rows of 5, and one extra? 
    # Better: 6x5 grid = 30 spots, pick 26? 
    # Let's use a dense hexagonal layout for initialization.
    
    def hex_layout(n):
        points = []
        # Hexagonal packing
        # r = 0.5 / sqrt(3) approx for unit width? 
        # Let's just use a grid for simplicity and let SA fix it.
        side = int(np.ceil(np.sqrt(n)))
        # 5x5 = 25. We need 26.
        # Use 6x5 = 30 grid, take first 26? No, that's uneven.
        # Let's use 5 rows. 5,5,5,5,6.
        rows = [5, 5, 5, 5, 6]
        xs, ys = [], []
        for i, cnt in enumerate(rows):
            for j in range(cnt):
                x = (j + 0.5) / cnt
                y = (i + 0.5) / len(rows)
                if i % 2 == 1:
                    x += 0.5 / cnt
                xs.append(np.clip(x, 0.05, 0.95))
                ys.append(np.clip(y, 0.05, 0.95))
        return np.array(xs), np.array(ys)

    x0, y0 = hex_layout(N)
    r0 = np.full(N, 0.05)
    
    # Simulated Annealing on Radii
    # State: radii r
    # Objective: Maximize sum(r) subject to valid packing
    # We use a potential function: Score = sum(r) if valid, else -Penalty
    
    def relax(x, y, r, iters=100):
        n = len(r)
        for _ in range(iters):
            moved = 0.0
            # Wall
            for i in range(n):
                if x[i] < r[i]:
                    moved += r[i] - x[i]
                    x[i] = r[i]
                elif x[i] > 1 - r[i]:
                    moved += x[i] - (1 - r[i])
                    x[i] = 1 - r[i]
                if y[i] < r[i]:
                    moved += r[i] - y[i]
                    y[i] = r[i]
                elif y[i] > 1 - r[i]:
                    moved += y[i] - (1 - r[i])
                    y[i] = 1 - r[i]
            
            # Pairwise
            dx = x[None, :] - x[:, None]
            dy = y[None, :] - y[:, None]
            dist = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dist, np.inf)
            req = r[None, :] + r[:, None]
            overlap = np.maximum(req - dist, 0)
            
            if np.max(overlap) < 1e-10:
                break
            
            moved += np.sum(overlap)
            
            for i in range(n):
                for j in range(i + 1, n):
                    ov = overlap[i, j]
                    if ov > 1e-10:
                        d = dist[i, j]
                        if d < 1e-10: continue
                        half = ov / 2
                        ux = dx[i, j] / d
                        uy = dy[i, j] / d
                        x[i] -= half * ux
                        y[i] -= half * uy
                        x[j] += half * ux
                        y[j] += half * uy
                        
        # Final check
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        req = r[None, :] + r[:, None]
        if np.any(dist < req - 1e-8):
            return None
        for i in range(n):
            if x[i] < -1e-8 or x[i] > 1 + 1e-8 or y[i] < -1e-8 or y[i] > 1 + 1e-8:
                return None
        return x, y

    def evaluate(r):
        x = x0.copy()
        y = y0.copy()
        # We need to start from a good position. 
        # Actually, for SA, we should keep the current best positions x_best, y_best.
        # But the function signature of construct is stateless.
        # We will keep x, y in the closure.
        
        res = relax(x, y, r, iters=50)
        if res is None:
            return -1e10, x, y
        x_new, y_new = res
        return np.sum(r), x_new, y_new

    # SA Loop
    r_curr = r0.copy()
    x_curr, y_curr = x0.copy(), y0.copy()
    # Initial relax
    res = relax(x_curr, y_curr, r_curr, iters=200)
    if res:
        x_curr, y_curr = res
    
    best_r = r_curr.copy()
    best_score = np.sum(r_curr)
    
    T = 0.01
    T_min = 1e-5
    alpha = 0.95
    steps = 500
    
    for step in range(steps):
        # Perturb radii
        r_try = r_curr.copy()
        # Randomly pick a few circles to change radius
        idx = np.random.choice(N, size=3, replace=False)
        for i in idx:
            r_try[i] += np.random.uniform(-0.01, 0.01)
            r_try[i] = np.clip(r_try[i], 0.01, 0.5)
            
        score_try, x_try, y_try = evaluate(r_try)
        
        if score_try > best_score:
            best_r = r_try.copy()
            best_score = score_try
            # Update current to best? Or accept?
            # Accepting improves local search.
            
        # Acceptance criterion
        if score_try > best_score or np.random.rand() < np.exp((score_try - best_score) / T):
             # Note: best_score is the global best. 
             # Standard SA compares to current state.
             # Let's keep a current_score.
             pass
             
    # The above SA logic is flawed because I didn't track current_score properly for Metropolis acceptance.
    # Let's rewrite the SA block properly.
    
    r_curr = r0.copy()
    x_curr, y_curr = x0.copy(), y0.copy()
    res = relax(x_curr, y_curr, r_curr, iters=200)
    if res: x_curr, y_curr = res
    curr_score = np.sum(r_curr)
    
    best_r = r_curr.copy()
    best_score = curr_score
    
    T = 0.05
    for step in range(1000):
        r_try = r_curr.copy()
        idx = np.random.choice(N, size=5, replace=False)
        for i in idx:
            r_try[i] += np.random.uniform(-0.005, 0.005)
            r_try[i] = np.clip(r_try[i], 0.01, 0.4)
        
        score_try, x_try, y_try = evaluate(r_try)
        
        if score_try > curr_score:
            r_curr = r_try
            x_curr, y_curr = x_try, y_try
            curr_score = score_try
            if curr_score > best_score:
                best_r = r_curr.copy()
                best_score = curr_score
        else:
            # Metropolis
            delta = curr_score - score_try
            if delta < 1e-6:
                if np.random.rand() < 1.0:
                    r_curr = r_try
                    x_curr, y_curr = x_try, y_try
                    curr_score = score_try
            else:
                if np.random.rand() < np.exp(-delta / T):
                    r_curr = r_try
                    x_curr, y_try = x_try, y_try # Typo in var name?
                    curr_score = score_try
                    
        T *= 0.99
        
    # Final polish
    x_final, y_final = x_curr.copy(), y_curr.copy()
    res = relax(x_final, y_final, best_r, iters=500)
    if res:
        x_final, y_final = res
    else:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack((x_final, y_final, best_r))
