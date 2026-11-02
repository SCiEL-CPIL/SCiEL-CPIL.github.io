import numpy as np

def construct():
    N = 32
    np.random.seed(42)
    
    # Initialize with hexagonal grid
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                centers.append([x, y])
    centers = np.array(centers)
    
    x = centers[:, 0].copy()
    y = centers[:, 1].copy()
    r = np.full(N, 0.05) # Start with small radii
    
    T_start = 0.05
    T_end = 1e-5
    steps = 5000
    alpha = (T_end / T_start) ** (1.0 / steps)
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Pre-compute current score
    def calc_score(x, y, r):
        score = np.sum(r)
        # Boundary
        for i in range(N):
            viol = 0
            if r[i] > x[i]: viol += (r[i] - x[i])**2
            if r[i] > 1 - x[i]: viol += (r[i] - (1 - x[i]))**2
            if r[i] > y[i]: viol += (r[i] - y[i])**2
            if r[i] > 1 - y[i]: viol += (r[i] - (1 - y[i]))**2
            score -= 50 * viol
        # Overlap
        for i, j in pairs:
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            overlap = r[i] + r[j] - d
            if overlap > 0:
                score -= 50 * overlap**2
        return score

    current_score = calc_score(x, y, r)
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = current_score
    
    for step in range(steps):
        T = T_start * (alpha ** step)
        
        # Propose
        n_moves = 5
        indices = np.random.choice(N, n_moves, replace=False)
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        for idx in indices:
            new_x[idx] = np.clip(x[idx] + np.random.normal(0, 0.02), 0, 1)
            new_y[idx] = np.clip(y[idx] + np.random.normal(0, 0.02), 0, 1)
            new_r[idx] = np.clip(r[idx] + np.random.normal(0, 0.01), 0.01, 0.5)
            
        new_score = calc_score(new_x, new_y, new_r)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            x = new_x
            y = new_y
            r = new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                
    # Final projection
    final_r = np.zeros(N)
    for _ in range(50):
        for i in range(N):
            r_max = min(best_x[i], 1-best_x[i], best_y[i], 1-best_y[i])
            for j in range(N):
                if i == j: continue
                d = np.sqrt((best_x[i] - best_x[j])**2 + (best_y[i] - best_y[j])**2)
                r_max = min(r_max, d - final_r[j])
            final_r[i] = max(0, r_max)
            
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_x[i], best_y[i], final_r[i]]
    return result
