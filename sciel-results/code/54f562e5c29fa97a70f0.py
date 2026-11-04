import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0

def hex_vertices(x, y, angle=0.0):
    # Returns 6 vertices of unit hexagon at (x,y)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack((x + np.cos(angles), y + np.sin(angles)))

def is_inside_outer(x, y, S):
    # Check if point (x,y) is inside outer hexagon side S, center 0,0
    # Normals at 30, 90, ...
    normals = []
    for k in range(6):
        a = np.pi/6 + k * np.pi/3
        normals.append((np.cos(a), np.sin(a)))
    limit = S * np.sqrt(3) / 2
    for nx, ny in normals:
        if nx * x + ny * y > limit + 1e-9:
            return False
    return True

def objective_and_constraints(vars):
    # vars: [x1, y1, ..., x12, y12, S]
    xs = vars[0:2*N:2]
    ys = vars[1:2*N:2]
    S = vars[-1]
    
    # 1. Containment Penalties/Constraints
    # We want to return constraints for SLSQP
    # g(x) >= 0
    
    cons = []
    
    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            # dist_sq >= 3
            cons.append({'type': 'ineq', 'fun': lambda d=dist_sq: d - 3.0})
            
    # Containment
    # Vertices of inner hex i are (x_i + cos(a), y_i + sin(a))
    # Condition: n . v <= S * sqrt(3)/2
    # n . (x_i + cos(a)) <= S * sqrt(3)/2
    # S * sqrt(3)/2 - n_x x_i - n_y y_i - n . offset >= 0
    
    for i in range(N):
        for k in range(6): # 6 vertices
            a = k * np.pi / 3
            ox = np.cos(a)
            oy = np.sin(a)
            for m in range(6): # 6 sides of outer
                am = np.pi/6 + m * np.pi/3
                nx = np.cos(am)
                ny = np.sin(am)
                limit = S * np.sqrt(3) / 2
                # limit - (nx*(xi+ox) + ny*(yi+oy)) >= 0
                # limit - nx*xi - ny*yi - nx*ox - ny*oy >= 0
                # This depends on S, xi, yi.
                # SLSQP requires fun(x) where x is the vector.
                # We can't easily capture xi, yi, S in a lambda without closure issues or passing indices.
                # Better to define a generic constraint function.
                pass

# SLSQP constraint functions must be defined properly.
# Let's define them dynamically or use a penalty method if SLSQP is too slow/complex to set up with 100+ constraints.
# Penalty method is easier to code and robust enough for this size.

def penalty_objective(vars):
    xs = vars[0:2*N:2]
    ys = vars[1:2*N:2]
    S = vars[-1]
    
    # Overlap penalty
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 3.0:
                pen_overlap += (3.0 - dist_sq)**2
                
    # Containment penalty
    pen_contain = 0.0
    limit = S * np.sqrt(3) / 2
    for i in range(N):
        vi = hex_vertices(xs[i], ys[i], 0.0)
        for v in vi:
            x, y = v
            for m in range(6):
                am = np.pi/6 + m * np.pi/3
                nx = np.cos(am)
                ny = np.sin(am)
                val = nx * x + ny * y
                if val > limit:
                    pen_contain += (val - limit)**2
                    
    return S + 10.0 * pen_overlap + 10.0 * pen_contain

def construct():
    best_S = float('inf')
    best_vars = None
    
    # Initial guesses
    inits = []
    
    # 1. Grid
    xs_init = []
    ys_init = []
    # 4x3 grid roughly
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3)/2
            y = row * 1.5
            xs_init.append(x)
            ys_init.append(y)
    # Center
    cx = np.mean(xs_init)
    cy = np.mean(ys_init)
    xs_init = [x - cx for x in xs_init]
    ys_init = [y - cy for y in ys_init]
    vars_init = []
    for i in range(N):
        vars_init.append(xs_init[i])
        vars_init.append(ys_init[i])
    vars_init.append(4.0) # Initial S guess
    inits.append(vars_init)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        vars_rand = np.random.uniform(-2, 2, 2*N)
        vars_rand = list(vars_rand) + [4.0]
        inits.append(vars_rand)
        
    # 3. Hex cluster subset
    # Center + 6 + 5
    xs_c = [0.0]
    ys_c = [0.0]
    for k in range(6):
        a = k * np.pi / 3
        xs_c.append(np.sqrt(3) * np.cos(a))
        ys_c.append(np.sqrt(3) * np.sin(a))
    # 5 more on ring 2
    # Pick 5 from the 12 positions of ring 2
    # Ring 2 positions:
    # (2*sqrt(3), 0), (sqrt(3), 3), (0, 3*sqrt(3))...
    # Let's just add 5 random points far out and let optimizer pull them in?
    # Or specific:
    xs_c.append(2*np.sqrt(3))
    ys_c.append(0)
    xs_c.append(np.sqrt(3))
    ys_c.append(3)
    xs_c.append(0)
    ys_c.append(3*np.sqrt(3))
    xs_c.append(-np.sqrt(3))
    ys_c.append(3)
    xs_c.append(-2*np.sqrt(3))
    ys_c.append(0)
    
    vars_c = []
    for i in range(N):
        vars_c.append(xs_c[i])
        vars_c.append(ys_c[i])
    vars_c.append(4.0)
    inits.append(vars_c)

    for v0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        try:
            res = minimize(penalty_objective, v0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success or res.fun < best_S:
                # Verify validity
                xs = res.x[0:2*N:2]
                ys = res.x[1:2*N:2]
                S = res.x[-1]
                
                # Check overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        if dx*dx + dy*dy < 3.0 - 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: continue
                
                # Check containment
                limit = S * np.sqrt(3) / 2
                for i in range(N):
                    vi = hex_vertices(xs[i], ys[i], 0.0)
                    for v in vi:
                        x, y = v
                        for m in range(6):
                            am = np.pi/6 + m * np.pi/3
                            nx = np.cos(am)
                            ny = np.sin(am)
                            if nx * x + ny * y > limit + 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: continue
                
                if S < best_S:
                    best_S = S
                    best_vars = res.x
                    
        except Exception as e:
            pass

    if best_vars is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_vars[2*i], best_vars[2*i+1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
