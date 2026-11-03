import numpy as np

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        # Autoconvolution
        c = np.convolve(h, h, mode='full')
        # Supremum norm of c
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    def render_atoms(params, n):
        # params: list of (center, width, amp)
        h = np.zeros(n)
        indices = np.arange(n)
        for c, w, a in params:
            if w <= 0 or a <= 0:
                continue
            # Triangular window
            # phi(x) = max(0, 1 - |x|/w)
            dist = np.abs(indices - c)
            window = np.maximum(0, 1 - dist / w)
            h += a * window
        return h

    def optimize_atoms(params, n, steps=50):
        # Local search on atom parameters
        current_score = calc_score(render_atoms(params, n))
        current_params = [list(p) for p in params]
        
        for _ in range(steps):
            improved = False
            # Try perturbing each parameter of each atom
            for k in range(len(current_params)):
                for dim in range(3): # center, width, amp
                    orig = current_params[k][dim]
                    # Determine perturbation size based on parameter type
                    if dim == 0: # center
                        delta = np.random.uniform(-2, 2)
                    elif dim == 1: # width
                        delta = np.random.uniform(-0.5, 0.5)
                    else: # amp
                        delta = np.random.uniform(-0.1, 0.1)
                    
                    new_val = orig + delta
                    # Constraints
                    if dim == 0:
                        new_val = np.clip(new_val, 0, n-1)
                    elif dim == 1:
                        new_val = max(0.5, new_val) # Min width
                    else:
                        new_val = max(0.0, new_val) # Min amp
                    
                    if new_val == orig:
                        continue
                        
                    trial_params = [p[:] for p in current_params]
                    trial_params[k][dim] = new_val
                    
                    trial_h = render_atoms(trial_params, n)
                    trial_score = calc_score(trial_h)
                    
                    if trial_score > current_score:
                        current_params = trial_params
                        current_score = trial_score
                        improved = True
            if not improved:
                break
                
        return render_atoms(current_params, n), current_score

    # Try different numbers of atoms
    max_restarts = 200
    for restart in range(max_restarts):
        # Randomly choose number of atoms between 1 and 10
        k_atoms = np.random.randint(1, 11)
        params = []
        for _ in range(k_atoms):
            center = np.random.uniform(0, n-1)
            width = np.random.uniform(1, 10)
            amp = np.random.uniform(0.1, 1.0)
            params.append([center, width, amp])
            
        h_opt, score_opt = optimize_atoms(params, n, steps=20)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # If no good set found, fallback to single spike
    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
