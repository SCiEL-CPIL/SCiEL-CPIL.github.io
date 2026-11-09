import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    best_val = 0
    best_centers = None
    best_sides = None
    
    # We will generate a set of topologies and optimize each
    # To ensure diversity, we use different seeds for topology generation
    
    # Class to represent a node in the guillotine tree
    class Node:
        def __init__(self, is_leaf, left=None, right=None, dir=None, param_idx=-1):
            self.is_leaf = is_leaf
            self.left = left
            self.right = right
            self.dir = dir # 0: H split (top/bottom), 1: V split (left/right)
            self.param_idx = param_idx

    def generate_topology(k, rng):
        """Recursively generate a random binary tree topology."""
        if k == 1:
            return Node(True)
        
        # Choose a random split point i in [1, k-1]
        # This determines how many leaves go to the left vs right
        i = rng.randint(1, k)
        
        left_node = generate_topology(i, rng)
        right_node = generate_topology(k - i, rng)
        
        # Choose split direction: 0 for horizontal (split height), 1 for vertical (split width)
        dir = rng.randint(0, 2)
        
        return Node(False, left_node, right_node, dir)

    def assign_params(node, params_list):
        """Assign parameter indices to internal nodes."""
        if not node.is_leaf:
            node.param_idx = len(params_list)
            params_list.append(0.5) # Initial guess
            assign_params(node.left, params_list)
            assign_params(node.right, params_list)

    def compute_objective(x, root):
        """Compute negative sum of sides for a given set of split ratios."""
        total_sum = 0.0
        
        def recurse(node, w, h):
            nonlocal total_sum
            if node.is_leaf:
                # The square must fit inside w x h
                side = min(w, h)
                total_sum += side
                return
            
            split_ratio = x[node.param_idx]
            # Clamp to avoid numerical issues, though bounds should handle it
            split_ratio = np.clip(split_ratio, 1e-6, 1 - 1e-6)
            
            if node.dir == 0: # Horizontal split
                h_top = h * split_ratio
                h_bot = h * (1.0 - split_ratio)
                recurse(node.left, w, h_top)
                recurse(node.right, w, h_bot)
            else: # Vertical split
                w_left = w * split_ratio
                w_right = w * (1.0 - split_ratio)
                recurse(node.left, w_left, h)
                recurse(node.right, w_right, h)

        recurse(root, 1.0, 1.0)
        return -total_sum

    def extract_solution(x, root):
        """Extract centers and sides from the optimized tree."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        leaf_idx = 0
        
        def recurse(node, w, h, cx, cy):
            nonlocal leaf_idx
            if node.is_leaf:
                side = min(w, h)
                # Center the square in the rectangle
                centers[leaf_idx] = [cx, cy]
                sides[leaf_idx] = side
                leaf_idx += 1
                return
            
            split_ratio = x[node.param_idx]
            split_ratio = np.clip(split_ratio, 1e-6, 1 - 1e-6)
            
            if node.dir == 0: # Horizontal split
                h_top = h * split_ratio
                h_bot = h * (1.0 - split_ratio)
                
                # Top child
                cy_top = cy - h/2 + h_top/2
                recurse(node.left, w, h_top, cx, cy_top)
                
                # Bottom child
                cy_bot = cy - h/2 + h_top + h_bot/2
                recurse(node.right, w, h_bot, cx, cy_bot)
            else: # Vertical split
                w_left = w * split_ratio
                w_right = w * (1.0 - split_ratio)
                
                # Left child
                cx_left = cx - w/2 + w_left/2
                recurse(node.left, w_left, h, cx_left, cy)
                
                # Right child
                cx_right = cx - w/2 + w_left + w_right/2
                recurse(node.right, w_right, h, cx_right, cy)

        recurse(root, 1.0, 1.0, 0.5, 0.5)
        
        if leaf_idx != n:
            return None, None
            
        return centers, sides

    def validate(centers, sides):
        """Check if the packing is valid."""
        tol = 1e-7
        n = len(sides)
        
        # Check bounds and positive sides
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                return False
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = centers[i]
                xj, yj = centers[j]
                si = sides[i]
                sj = sides[j]
                t = (si + sj) / 2.0 - tol
                if abs(xi - xj) < t and abs(yi - yj) < t:
                    return False
        return True

    # Main search loop
    # We will try multiple random topologies
    # Using different seeds ensures diversity
    num_trials = 100
    
    for trial in range(num_trials):
        # Use a different seed for each trial to get different topologies
        rng = np.random.RandomState(42 + trial)
        
        # Generate topology
        root = generate_topology(n, rng)
        
        # Assign parameters
        params_list = []
        assign_params(root, params_list)
        n_params = len(params_list)
        
        if n_params == 0:
            continue
            
        # Define objective function
        def objective(x):
            return compute_objective(x, root)
            
        # Initial guess: all 0.5
        x0 = np.ones(n_params) * 0.5
        bounds = [(0.01, 0.99)] * n_params
        
        # Optimize
        # L-BFGS-B is good for box constraints
        # We can try multiple initial guesses if time permits, but let's start with one
        # and maybe add a second start if the first is poor
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10})
            x_opt = res.x
        except Exception:
            continue
            
        # Extract solution
        centers, sides = extract_solution(x_opt, root)
        
        if centers is None:
            continue
            
        # Validate
        if validate(centers, sides):
            val = np.sum(sides)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Fallback if no valid solution found (should not happen)
    if best_centers is None:
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Simple grid fallback
        # 4x3 grid? 12 cells, we need 10.
        # Let's just use 5x2 grid of 0.2x0.5 rects -> squares of 0.2
        for i in range(n):
            col = i % 5
            row = i // 5
            centers[i] = [(col + 0.5) / 5.0, (row + 0.5) / 2.0]
            sides[i] = 0.2
        return centers, sides

    return best_centers, best_sides
