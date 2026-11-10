import numpy as np
from scipy.optimize import linprog
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 500

    # Lattice basis
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Pool of lattice points
    pool = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            pool.append(p)
    pool = np.array(pool)
    
    # Normals for outer hex (angle 0)
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def get_vertices(center):
        hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        verts = np.array([[np.cos(a), np.sin(a)] for a in hex_angles])
        return center + verts

    def calc_s_lp(cluster_centers):
        c = np.array([0, 0, 1])
        all_verts = np.vstack([get_vertices(cc) for cc in cluster_centers])
        
        A_ub = []
        b_ub = []
        
        for v in all_verts:
            for n in normals:
                row = np.array([-n[0], -n[1], -np.sqrt(3)/2])
                rhs = -np.dot(v, n)
                A_ub.append(row)
                b_ub.append(rhs)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            return res.fun, np.array([res.x[0], res.x[1]])
        else:
            return float('inf'), np.array([0,0])

    # Initial cluster: 12 closest to origin
    dists = np.linalg.norm(pool, axis=1)
    sorted_idx = np.argsort(dists)
    current_indices = set(sorted_idx[:12].tolist())
    initial_cluster = pool[list(current_indices)]
    
    best_s, best_C = calc_s_lp(initial_cluster)
    best_cluster = initial_cluster.copy()
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        idx_to_move = list(current_indices)[np.random.randint(12)]
        moving_point = pool[idx_to_move]
        
        available = [i for i in range(len(pool)) if i not in current_indices]
        if not available:
            break
            
        available.sort(key=lambda i: np.linalg.norm(pool[i] - moving_point))
        
        improved = False
        # Try top 5 candidates
        for cand_idx in available[:5]:
            new_indices = set(current_indices)
            new_indices.remove(idx_to_move)
            new_indices.add(cand_idx)
            new_cluster = pool[list(new_indices)]
            s_new, C_new = calc_s_lp(new_cluster)
            
            if s_new < best_s - 1e-9:
                best_s = s_new
                best_C = C_new
                best_cluster = new_cluster.copy()
                current_indices = new_indices
                improved = True
                break
        
        if not improved and iterations % 50 == 0:
            # Random restart
            # Generate a random connected cluster
            start_idx = np.random.randint(len(pool))
            curr = set([start_idx])
            while len(curr) < 12:
                neighbors = []
                for i in range(len(pool)):
                    if i in curr: continue
                    min_d = min(np.linalg.norm(pool[i] - pool[c]) for c in curr)
                    if abs(min_d - np.sqrt(3)) < 1e-6:
                        neighbors.append(i)
                if not neighbors:
                    break
                curr.add(neighbors[np.random.randint(len(neighbors))])
            if len(curr) == 12:
                current_indices = curr
                initial_cluster = pool[list(curr)]
                s_init, C_init = calc_s_lp(initial_cluster)
                if s_init < best_s:
                    best_s = s_init
                    best_C = C_init
                    best_cluster = initial_cluster.copy()

    final_centers = best_cluster - best_C
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
