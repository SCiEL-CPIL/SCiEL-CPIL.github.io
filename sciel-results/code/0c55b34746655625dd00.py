import numpy as np
from scipy.spatial.distance import cdist

N = 50

def construct():
    def objective(flat):
        c = flat.reshape(N, 3)
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        
        r = min(min_pair, min_wall)
        return -r

    def initial_guess():
        # 5x5x2 grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.array([-0.2, 0.2])
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        c = np.array(grid[:N])
        c += np.random.normal(0, 0.01, c.shape)
        return np.clip(c, -0.49, 0.49).flatten()

    best_c = None
    best_val = -1e10
    
    # Run 3 restarts to be safe within time limit
    for restart in range(3):
        c = initial_guess()
        current_val = objective(c)
        
        T = 0.005
        T_min = 1e-8
        cooling = 0.995
        iters = 10000
        step_size = 0.01
        
        for i in range(iters):
            c_new = c.copy()
            n_perturb = np.random.randint(1, 4)
            idxs = np.random.choice(N, n_perturb, replace=False)
            c_new[idxs] += np.random.normal(0, step_size, (n_perturb, 3))
            c_new = np.clip(c_new, -0.49, 0.49)
            
            new_val = objective(c_new)
            delta = new_val - current_val
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                c = c_new
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_c = c.copy()
            
            T *= cooling
            # Decay step size slightly
            step_size *= 0.999
            
    return best_c.reshape(N, 3)
