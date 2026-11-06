import time
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct():
    start_time = time.time()
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    all_5subs = list(combinations(points, 5))
    
    n_blocks = len(all_blocks)
    n_5subs = len(all_5subs)
    
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    A = np.zeros((n_5subs, n_blocks), dtype=np.int8)
    
    for j, block in enumerate(all_blocks):
        for sub in combinations(block, 5):
            i = sub_to_idx[sub]
            A[i, j] = 1
            
    c = -np.ones(n_blocks)
    constraint = LinearConstraint(A, ub=1)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_blocks)
    
    # Try ILP with a tighter time limit
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"presolve": True, "time_limit": 10})
        
        if res.success and res.x is not None:
            x = res.x.round().astype(int)
            selected_blocks = [list(all_blocks[i]) for i in range(n_blocks) if x[i] == 1]
            
            used_5subs = set()
            valid = True
            for b in selected_blocks:
                for s in combinations(b, 5):
                    if s in used_5subs:
                        valid = False
                        break
                    used_5subs.add(s)
                if not valid:
                    break
            
            if valid:
                return selected_blocks
    except Exception as e:
        pass

    # Fallback: Improved Greedy
    block_5subs = [list(combinations(b, 5)) for b in all_blocks]
    
    best_blocks = []
    rng = random.Random(42)
    
    for _ in range(50):
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        used_5subs = set()
        current_blocks = []
        
        for idx in indices:
            b = all_blocks[idx]
            conflict = False
            for s in block_5subs[idx]:
                if s in used_5subs:
                    conflict = True
                    break
            if not conflict:
                current_blocks.append(b)
                for s in block_5subs[idx]:
                    used_5subs.add(s)
                
                if len(current_blocks) == 132:
                    return [list(b) for b in current_blocks]
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            
    return [list(b) for b in best_blocks]
