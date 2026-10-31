<<<<<<< SEARCH
    def objective(v):
        r = v[2::3][:N]
        return -np.sum(r)
        
    def make_constraints():
        """
        Returns a list of constraint dictionaries for SLSQP.
        """
        cons = []
        
        # 1. Boundary Constraints
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[0::3][:N] - v[2::3][:N]
        })
        # W - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[-1] - v[0::3][:N] - v[2::3][:N]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[1::3][:N] - v[2::3][:N]
        })
        # H - y - r >= 0  =>  (2 - W) - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: (2.0 - v[-1]) - v[1::3][:N] - v[2::3][:N]
        })
        
        # 2. Non-overlap Constraints
        # dist(i,j) - r_i - r_j >= 0
        def non_overlap(v):
            x = v[0::3][:N]
            y = v[1::3][:N]
            r = v[2::3][:N]
            cons_val = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons_val.append(dist - r[i] - r[j])
            return np.array(cons_val)
        
        cons.append({
            'type': 'ineq',
            'fun': non_overlap
        })
        
        return cons

    constraints = make_constraints()
    
    # Bounds for SLSQP
    # x, y in [0, 2] (since W+H=2, max dim is 2)
    # r in [0, 1]
    # W in [0.5, 1.5] (avoid degenerate)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds.append((0.5, 1.5)) # W
    
    np.random.seed(42)
    
    # Number of restarts to try
    # We have ~115s. SLSQP might take 1-5s per run. Let's try 10-20 restarts.
    max_restarts = 15
=======
    def objective(v):
        r = v[2::3][:N]
        return -np.sum(r)
        
    def make_constraints():
        """
        Returns a list of constraint dictionaries for SLSQP.
        """
        cons = []
        
        # 1. Boundary Constraints
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[0::3][:N] - v[2::3][:N]
        })
        # W - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[-1] - v[0::3][:N] - v[2::3][:N]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[1::3][:N] - v[2::3][:N]
        })
        # H - y - r >= 0  =>  (2 - W) - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: (2.0 - v[-1]) - v[1::3][:N] - v[2::3][:N]
        })
        
        # 2. Non-overlap Constraints
        # dist(i,j) - r_i - r_j >= 0
        def non_overlap(v):
            x = v[0::3][:N]
            y = v[1::3][:N]
            r = v[2::3][:N]
            cons_val = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons_val.append(dist - r[i] - r[j])
            return np.array(cons_val)
        
        cons.append({
            'type': 'ineq',
            'fun': non_overlap
        })
        
        return cons

    constraints = make_constraints()
    
    # Bounds for SLSQP
    # x, y in [0, 2] (since W+H=2, max dim is 2)
    # r in [0, 1]
    # W in [0.5, 1.5] (avoid degenerate)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds.append((0.5, 1.5)) # W
    
    np.random.seed(42)
    
    # Number of restarts to try
    # We have ~115s. SLSQP might take 1-5s per run. Let's try 10-20 restarts.
    max_restarts = 15
    
    def is_valid_solution(x, y, r, W, H, tol=1e-6):
        """
        Strictly check if the solution is valid.
        """
        if np.any(r < -tol):
            return False
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > W - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > H - r + tol):
            return False
        
        # Overlap checks
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True
>>>>>>> REPLACE

