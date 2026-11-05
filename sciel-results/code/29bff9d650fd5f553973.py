import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def generate_indices(n):
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_edges = len(triples)
        
        k4_vertices = list(itertools.combinations(range(n), 4))
        k4_edges = []
        for quad in k4_vertices:
            a, b, c, d = quad
            # Triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            k4_edges.append([
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            ])
        k4_edges = np.array(k4_edges, dtype=np.int32)
        
        k5_vertices = list(itertools.combinations(range(n), 5))
        k5_edges = []
        for quint in k5_vertices:
            edges = []
            # 10 combinations of 3 from 5
            for combo in itertools.combinations(quint, 3):
                edges.append(triple_to_idx[combo])
            k5_edges.append(edges)
        k5_edges = np.array(k5_edges, dtype=np.int32)
        
        return triple_to_idx, k4_edges, k5_edges, num_edges

    def solve_n(n, time_limit):
        t_start = time.time()
        if time.time() - t_start > time_limit:
            return None
            
        try:
            triple_to_idx, k4_edges, k5_edges, num_edges = generate_indices(n)
        except Exception:
            return None
            
        if time.time() - t_start > time_limit:
            return None
            
        # Pre-compute which edges are in which K4/K5 for faster updates?
        # No, vectorized check is fast enough.
        
        best_colors = None
        best_violations = np.inf
        
        # Multiple restarts
        max_restarts = 10
        steps_per_restart = 50000 # Adjust based on speed
        
        for r in range(max_restarts):
            if time.time() > t_start + time_limit:
                break
                
            colors = np.random.randint(0, 2, size=num_edges, dtype=np.uint8)
            current_violations = np.inf
            no_improve_count = 0
            
            for step in range(steps_per_restart):
                if time.time() > t_start + time_limit:
                    break
                    
                # Check violations
                red_k4_bad = np.all(colors[k4_edges] == 0, axis=1)
                blue_k5_bad = np.all(colors[k5_edges] == 1, axis=1)
                
                n_red_bad = np.sum(red_k4_bad)
                n_blue_bad = np.sum(blue_k5_bad)
                total_violations = n_red_bad + n_blue_bad
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    
                if total_violations == 0:
                    return {'n': n, 'colors': best_colors.tolist()}
                    
                if total_violations < current_violations:
                    current_violations = total_violations
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # If stuck, restart early
                if no_improve_count > 500:
                    break
                    
                # Select a random violation to fix
                if n_red_bad > 0 and (n_blue_bad == 0 or np.random.rand() < 0.5):
                    bad_indices = np.where(red_k4_bad)[0]
                    bad_idx = bad_indices[np.random.randint(len(bad_indices))]
                    edge_pos = np.random.randint(4)
                    edge_idx = k4_edges[bad_idx, edge_pos]
                    colors[edge_idx] = 1
                else:
                    bad_indices = np.where(blue_k5_bad)[0]
                    bad_idx = bad_indices[np.random.randint(len(bad_indices))]
                    edge_pos = np.random.randint(10)
                    edge_idx = k5_edges[bad_idx, edge_pos]
                    colors[edge_idx] = 0
                    
        # If we didn't find a valid one, check if best_colors is valid (it should be if 0)
        # But we only return if valid.
        return None

    # Try n=34
    result = solve_n(34, 135)
    if result:
        return result
        
    # If failed, try n=33
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 10:
            result = solve_n(33, remaining - 5)
            if result:
                return result
                
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
