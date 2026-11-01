import numpy as np
import time
import itertools
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_space = q ** n
    
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    balls = []
    for i in range(total_space):
        center = all_centers[i]
        indices = [i]
        if radius >= 1:
            for pos in range(n):
                orig = center[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_word = center.copy()
                    new_word[pos] = new_val
                    indices.append(center_to_idx[tuple(new_word)])
        balls.append(np.array(indices, dtype=np.int32))
        
    ball_indices_matrix = np.array(balls, dtype=np.int32)
    
    lp_solution = None
    try:
        lp_deadline = start_time + 30.0
        
        A = lil_matrix((total_space, total_space), dtype=np.int8)
        for j in range(total_space):
            for i in balls[j]:
                A[i, j] = 1
        A = A.tocsr()
        
        c = np.ones(total_space)
        A_ub = -A
        b_ub = -np.ones(total_space)
        bounds = [(0, 1)] * total_space
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if time.time() > lp_deadline:
            raise TimeoutError("LP took too long")
            
        if res.status == 0:
            lp_solution = res.x
            
    except Exception:
        lp_solution = None
        
    chosen_centers = []
    covered = np.zeros(total_space, dtype=bool)
    
    if lp_solution is not None:
        sorted_indices = np.argsort(lp_solution)[::-1]
        for idx in sorted_indices:
            if time.time() > deadline - 10.0:
                break
            if np.any(~covered[ball_indices_matrix[idx]]):
                chosen_centers.append(int(idx))
                covered[ball_indices_matrix[idx]] = True
                
        if not np.all(covered):
            while not np.all(covered) and time.time() < deadline - 5.0:
                scores = (~covered[ball_indices_matrix]).sum(axis=1)
                best_idx = np.argmax(scores)
                if scores[best_idx] == 0:
                    break
                chosen_centers.append(int(best_idx))
                covered[ball_indices_matrix[best_idx]] = True
    else:
        while not np.all(covered) and time.time() < deadline - 5.0:
            scores = (~covered[ball_indices_matrix]).sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            chosen_centers.append(int(best_idx))
            covered[ball_indices_matrix[best_idx]] = True
            
    if time.time() < deadline - 2.0:
        redundant_removed = True
        while redundant_removed and time.time() < deadline - 2.0:
            redundant_removed = False
            counts = np.zeros(total_space, dtype=np.int32)
            for c_idx in chosen_centers:
                counts[ball_indices_matrix[c_idx]] += 1
            
            for i, c_idx in enumerate(chosen_centers):
                if np.min(counts[ball_indices_matrix[c_idx]]) > 1:
                    chosen_centers.pop(i)
                    redundant_removed = True
                    break
                    
    return [all_centers[i] for i in chosen_centers]
