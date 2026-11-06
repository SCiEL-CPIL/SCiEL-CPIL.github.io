import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    for n in range(34, 5, -1):
        if time.time() > deadline - 5:
            break
            
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        red_k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idxs = [
                triple_index[(a, b, c)],
                triple_index[(a, b, d)],
                triple_index[(a, c, d)],
                triple_index[(b, c, d)]
            ]
            red_k4s.append(idxs)
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for t in combinations(quint, 3):
                idxs.append(triple_index[t])
            blue_k5s.append(idxs)
            
        red_k4s_arr = np.array(red_k4s, dtype=np.int32)
        blue_k5s_arr = np.array(blue_k5s, dtype=np.int32)
        
        colors = np.random.randint(0, 2, num_triples).astype(np.int8)
        
        def get_violations(cols):
            red_sums = np.sum(cols[red_k4s_arr], axis=1)
            red_viol_idx = np.where(red_sums == 0)[0]
            
            blue_sums = np.sum(cols[blue_k5s_arr], axis=1)
            blue_viol_idx = np.where(blue_sums == 10)[0]
            
            return red_viol_idx, blue_viol_idx
        
        sa_deadline = time.time() + 30
        while time.time() < sa_deadline:
            red_viols, blue_viols = get_violations(colors)
            
            if len(red_viols) == 0 and len(blue_viols) == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Pick a random violation
            # Weighted random: if both exist, pick one randomly
            if len(red_viols) > 0 and len(blue_viols) > 0:
                if np.random.rand() < 0.5:
                    v_idx = red_viols[np.random.randint(len(red_viols))]
                    clause = red_k4s_arr[v_idx]
                else:
                    v_idx = blue_viols[np.random.randint(len(blue_viols))]
                    clause = blue_k5s_arr[v_idx]
            elif len(red_viols) > 0:
                v_idx = red_viols[np.random.randint(len(red_viols))]
                clause = red_k4s_arr[v_idx]
            else:
                v_idx = blue_viols[np.random.randint(len(blue_viols))]
                clause = blue_k5s_arr[v_idx]
            
            # Pick a random variable in the clause
            var_idx = clause[np.random.randint(len(clause))]
            
            # Flip
            colors[var_idx] = 1 - colors[var_idx]
            
    return {'n': 1, 'colors': []}
