import numpy as np
from scipy.optimize import milp, LinearConstraint
from itertools import combinations

def construct():
    points = range(12)
    # Generate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    # Generate all 5-subsets
    all_5_subsets = list(combinations(points, 5))
    
    n_blocks = len(all_blocks)
    n_5_subsets = len(all_5_subsets)
    
    # Create incidence matrix A: A[i, j] = 1 if 5_subset i is in block j
    # Using numpy for speed
    # 5-subsets are tuples, blocks are tuples.
    # We can map 5-subsets to indices.
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    A = np.zeros((n_5_subsets, n_blocks), dtype=np.int8)
    for j, b in enumerate(all_blocks):
        for s in combinations(b, 5):
            A[subset_to_idx[s], j] = 1
            
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_blocks)
    
    # Constraints: A x <= 1
    constraints = LinearConstraint(A, 0, 1)
    
    # Integrality: x must be integer (0 or 1)
    integrality = np.ones(n_blocks)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * n_blocks
    
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback to greedy if ILP fails
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

    x = res.x
    blocks = [list(all_blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
    return blocks
