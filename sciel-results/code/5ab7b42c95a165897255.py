import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Constants
    N = 11
    # Distance between centers of touching unit hexagons (same orientation)
    D_TOUCH = np.sqrt(3)
    
    # Initial configurations for the 11 hexagons
    # We use a hexagonal lattice base
    # p0 is center
    # p1-p6 are neighbors
    # p7-p10 are from the next ring
    
    base_7 = np.array([
        [0.0, 0.0],
        [D_TOUCH, 0.0],
        [D_TOUCH/2, 1.5],
        [-D_TOUCH/2, 1.5],
        [-D_TOUCH, 0.0],
        [-D_TOUCH/2, -1.5],
        [D_TOUCH/2, -1.5]
    ])
    
    # Candidates for the remaining 4 hexagons
    # We try a few symmetric arrangements
    # 1. 4 points at angles 0, 120, 240, 30 (asymmetric) -> maybe not
    # 2. 4 points at angles 0, 120, 240, 30? 
    # Let's pick 4 points from the 2nd ring (radius 2*sqrt(3))
    # Angles in 2nd ring: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # To maintain 3-fold symmetry, we could pick 3 points. But we need 4.
    # Maybe the optimal solution breaks 6-fold symmetry.
    
    # Let's just randomize the positions of the 4 extra hexagons slightly around the 2nd ring
    # and let the optimizer find the best spots.
    
    def get_initial_guess(seed):
        rng = np.random.RandomState(seed)
        p = base_7.copy()
        # Place 4 more hexagons
        # Start with a guess of them being in the 2nd ring
        # Pick 4 angles
        angles = [0, 120, 240, 30] # Just a guess
        # Or better, place them at the "corners" of a larger hexagon
        # Let's place them at radius 2*sqrt(3)
        r = 2 * D_TOUCH
        for i in range(4):
            # Add some noise
            a = np.deg2rad(angles[i]) + rng.uniform(-0.5, 0.5)
            p_new = np.array([r * np.cos(a), r * np.sin(a)])
            p = np.vstack([p, p_new])
        return p

    def compute_vertices(p, s=1.0, a=0.0):
        # p: (x, y)
        # Returns (6, 2) array of vertices
        angles = np.deg2rad(a) + np.deg2rad(60 * np.arange(6))
        verts = s * np.column_stack([np.cos(angles), np.sin(angles)])
        return verts + p

    def objective_and_constraints(x):
        # x: [x0, y0, ..., x10, y10, S, theta]
        positions = x[:2*N].reshape(N, 2)
        S = x[2*N]
        theta = x[2*N+1]
        
        # Objective: S
        obj = S
        
        # Constraints (all >= 0)
        cons = []
        
        # 1. Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                d_sq = np.sum((positions[i] - positions[j])**2)
                cons.append(d_sq - D_TOUCH**2)
                
        # 2. Containment
        # Outer hexagon normals
        # Normals are at theta + 30, theta + 90, theta + 150
        # (Since vertices are at theta, theta+60, ..., edges are between them, normals at midpoints)
        normal_angles = np.deg2rad(theta + 30 + 60 * np.arange(3))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        apothem = S * np.sqrt(3) / 2
        
        # Vertices of all inner hexagons
        # Assume inner angle is 0
        inner_angles = np.deg2rad(60 * np.arange(6))
        inner_verts_rel = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        
        for i in range(N):
            p_i = positions[i]
            # Vertices of inner hex i
            v_i = inner_verts_rel + p_i
            
            # Check each vertex against each normal
            for v in v_i:
                for n in normals:
                    dot = np.dot(v, n)
                    # v . n <= apothem  =>  apothem - dot >= 0
                    cons.append(apothem - dot)
                    # -v . n <= apothem =>  apothem + dot >= 0
                    cons.append(apothem + dot)
                    
        return obj, cons

    def run_optimization(x0):
        # Flatten x0
        x0_flat = np.concatenate([x0.flatten(), [8.0, 0.0]]) # S=8, theta=0
        
        # Define constraints for SLSQP
        # SLSQP expects constraints in form {'type': 'ineq', 'fun': lambda x: ...}
        # where fun(x) >= 0
        
        # We need to separate objective and constraints for SLSQP
        # Or use a wrapper
        
        def obj_func(x):
            obj, _ = objective_and_constraints(x)
            return obj
            
        def cons_func(x):
            _, cons = objective_and_constraints(x)
            return cons
            
        # SLSQP takes a list of constraints
        # We can pass a single function that returns an array of constraint values
        # and specify type='ineq'
        
        constraints = {'type': 'ineq', 'fun': cons_func}
        
        try:
            res = minimize(obj_func, x0_flat, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    best_S = np.inf
    best_x = None
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline - 30:
            break
            
        x0 = get_initial_guess(seed)
        x_opt = run_optimization(x0)
        
        if x_opt is not None:
            S_opt = x_opt[2*N]
            if S_opt < best_S:
                best_S = S_opt
                best_x = x_opt
                
    if best_x is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Extract results
    positions = best_x[:2*N].reshape(N, 2)
    S = best_x[2*N]
    theta = best_x[2*N+1]
    
    # Format output
    inner_hex_data = np.zeros((N, 3))
    inner_hex_data[:, 0] = positions[:, 0]
    inner_hex_data[:, 1] = positions[:, 1]
    inner_hex_data[:, 2] = 0.0 # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(theta)])
    
    return inner_hex_data, outer_hex_data, float(S)

if __name__ == "__main__":
    hexagon_packing_11()
