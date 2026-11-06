import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit
    
    # Known best solution: Paley tournament on p=67
    p_known = 67
    squares = {x * x % p_known for x in range(1, p_known)}
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)
    
    best_n = p_known
    best_orientation = orientation_known
    
    # Try to find a solution for n = 66
    target_n = 66
    m = target_n * (target_n - 1) // 2
    
    # Helper to map (i, j) to index in orientation array
    # i < j
    # Index = i*(target_n - 1) - i*(i+1)//2 + (j - i - 1)
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-tuples
    quad_sets = []
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                for d in range(c + 1, target_n):
                    quad_sets.append((a, b, c, d))
    
    num_quads = len(quad_sets)
    
    # Precompute edge indices for each 4-set
    # For a set (a, b, c, d), the 6 edges are:
    # (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
    edge_indices_per_quad = np.zeros((num_quads, 6), dtype=np.int32)
    for i in range(num_quads):
        a, b, c, d = quad_sets[i]
        edge_indices_per_quad[i, 0] = get_idx(a, b, target_n)
        edge_indices_per_quad[i, 1] = get_idx(a, c, target_n)
        edge_indices_per_quad[i, 2] = get_idx(a, d, target_n)
        edge_indices_per_quad[i, 3] = get_idx(b, c, target_n)
        edge_indices_per_quad[i, 4] = get_idx(b, d, target_n)
        edge_indices_per_quad[i, 5] = get_idx(c, d, target_n)
        
    # Precompute which quads are affected by which edge
    # edge_to_quads[e] = list of quad indices
    edge_to_quads = [[] for _ in range(m)]
    for q in range(num_quads):
        for k in range(6):
            e_idx = edge_indices_per_quad[q, k]
            edge_to_quads[e_idx].append(q)
    
    def is_good_quad(orient, q_idx):
        """Check if quad q_idx has a dominator given current orient."""
        e_ab = orient[edge_indices_per_quad[q_idx, 0]]
        e_ac = orient[edge_indices_per_quad[q_idx, 1]]
        e_ad = orient[edge_indices_per_quad[q_idx, 2]]
        e_bc = orient[edge_indices_per_quad[q_idx, 3]]
        e_bd = orient[edge_indices_per_quad[q_idx, 4]]
        e_cd = orient[edge_indices_per_quad[q_idx, 5]]
        
        # a dominates: e_ab=0, e_ac=0, e_ad=0
        if e_ab == 0 and e_ac == 0 and e_ad == 0:
            return True
        # b dominates: e_ab=1, e_bc=0, e_bd=0
        if e_ab == 1 and e_bc == 0 and e_bd == 0:
            return True
        # c dominates: e_ac=1, e_bc=1, e_cd=0
        if e_ac == 1 and e_bc == 1 and e_cd == 0:
            return True
        # d dominates: e_ad=1, e_bd=1, e_cd=1
        if e_ad == 1 and e_bd == 1 and e_cd == 1:
            return True
        return False

    def calculate_delta_bad(orient, e_idx):
        """Calculate change in bad count if we flip edge e_idx."""
        delta = 0
        for q_idx in edge_to_quads[e_idx]:
            old_good = is_good_quad(orient, q_idx)
            
            # Temporarily flip
            old_val = orient[e_idx]
            orient[e_idx] = 1 - old_val
            
            new_good = is_good_quad(orient, q_idx)
            
            # Revert
            orient[e_idx] = old_val
            
            if old_good and not new_good:
                delta += 1
            elif not old_good and new_good:
                delta -= 1
        return delta

    def simulated_annealing(n, time_limit):
        m = n * (n - 1) // 2
        start_sa = time.time()
        
        # Initialize
        orient = np.random.randint(0, 2, size=m, dtype=np.int8)
        
        # Initial good mask and bad count
        good_mask = np.zeros(num_quads, dtype=bool)
        bad_count = 0
        for q in range(num_quads):
            if is_good_quad(orient, q):
                good_mask[q] = True
            else:
                bad_count += 1
        
        best_orient = orient.copy()
        best_bad = bad_count
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        iterations = 0
        while time.time() - start_sa < time_limit:
            iterations += 1
            # Pick random edge
            e_idx = random.randint(0, m - 1)
            
            delta = calculate_delta_bad(orient, e_idx)
            
            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept move
                orient[e_idx] = 1 - orient[e_idx]
                bad_count += delta
                
                # Update good_mask for affected quads
                for q_idx in edge_to_quads[e_idx]:
                    good_mask[q_idx] = is_good_quad(orient, q_idx)
                
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_orient = orient.copy()
                    if best_bad == 0:
                        return best_orient, True
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_orient, (best_bad == 0)

    # Try to find a solution for n = 66
    if time.time() - start_time < deadline - 30:
        try:
            sa_time = min(100, deadline - 30 - (time.time() - start_time))
            result_orient, success = simulated_annealing(target_n, sa_time)
            
            if success:
                orientation_list = result_orient.tolist()
                best_n = target_n
                best_orientation = orientation_list
            else:
                # If not successful, we could try a few more restarts or local search polish
                # But for now, fallback to 67
                pass
        except Exception as e:
            pass  # Fall back to known solution
    
    return {'n': best_n, 'orientation': best_orientation}
