import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 11

def get_hexagon_vertices(center, angle_deg, side=1.0):
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([
        x + side * np.cos(angles),
        y + side * np.sin(angles)
    ])
    return vertices

def min_enclosing_hexagon_side(hull_vertices, theta_deg):
    theta = np.deg2rad(theta_deg)
    normals = np.column_stack([
        np.cos(theta + np.deg2rad(np.arange(0, 360, 60))),
        np.sin(theta + np.deg2rad(np.arange(0, 360, 60)))
    ])
    projections = hull_vertices @ normals.T
    max_projections = np.max(projections, axis=0)
    A = np.max(max_projections)
    L = 2 * A / np.sqrt(3)
    return L

def objective(params):
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1:11] = params[0:20].reshape(10, 2)
    
    inner_angles = np.zeros(11)
    
    all_vertices = []
    for i in range(11):
        verts = get_hexagon_vertices(positions[i], inner_angles[i])
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    try:
        hull = ConvexHull(all_vertices)
        hull_vertices = all_vertices[hull.vertices]
    except Exception:
        return 1e6
        
    outer_angle = params[20]
    
    return min_enclosing_hexagon_side(hull_vertices, outer_angle)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    sqrt3 = np.sqrt(3)
    base_cluster = [
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5]
    ]
    
    best_score = 1e9
    best_params = None
    
    # Define a few initial guesses for the 4 extra hexagons
    guesses = []
    
    # Guess 1: Diagonal (30, 150, 210, 330)
    d = 4.0
    g1 = base_cluster + [
        [d * np.cos(np.deg2rad(30)), d * np.sin(np.deg2rad(30))],
        [d * np.cos(np.deg2rad(150)), d * np.sin(np.deg2rad(150))],
        [d * np.cos(np.deg2rad(210)), d * np.sin(np.deg2rad(210))],
        [d * np.cos(np.deg2rad(330)), d * np.sin(np.deg2rad(330))]
    ]
    guesses.append(g1)
    
    # Guess 2: Cardinal (0, 90, 180, 270)
    g2 = base_cluster + [
        [d * np.cos(np.deg2rad(0)), d * np.sin(np.deg2rad(0))],
        [d * np.cos(np.deg2rad(90)), d * np.sin(np.deg2rad(90))],
        [d * np.cos(np.deg2rad(180)), d * np.sin(np.deg2rad(180))],
        [d * np.cos(np.deg2rad(270)), d * np.sin(np.deg2rad(270))]
    ]
    guesses.append(g2)
    
    # Guess 3: Random
    np.random.seed(42)
    g3 = base_cluster + [
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
        [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
    ]
    guesses.append(g3)

    for guess in guesses:
        if time.time() > deadline:
            break
            
        guess_arr = np.array(guess)
        init_params = np.concatenate([guess_arr[1:].flatten(), [0.0]])
        
        try:
            res = minimize(objective, init_params, method='Powell', options={'maxiter': 50, 'xtol': 1e-5, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1:11] = best_params[0:20].reshape(10, 2)
    outer_angle = best_params[20]
    
    inner = []
    for i in range(11):
        inner.append([positions[i][0], positions[i][1], 0.0])
        
    return inner, (0.0, 0.0), best_score, outer_angle
# EVOLVE-BLOCK-END
