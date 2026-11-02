"""Covering design C(24,7,3) - ILP-based optimization over candidate blocks."""
import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)  # C(24,3) = 2024
    
    # Map each t-subset to an index
    t_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Phase 1: Generate candidate blocks using greedy with randomization
    def greedy_block(uncovered_set, block_seed=None, rng=None):
        """Greedy construction of a block that covers many uncovered t-subsets."""
        if block_seed is not None:
            block = list(block_seed)
        else:
            # Pick a random uncovered t-subset as seed
            if rng is not None:
                idx = rng.choice(len(uncovered_set))
                sub = list(uncovered_set)[idx]
            else:
                sub = min(uncovered_set)
            block = list(sub)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is not None:
                block.append(best_x)
            else:
                break
        return sorted(block)
    
    # Generate initial greedy solution
    uncovered = set(all_t_subsets)
    blocks = []
    while uncovered:
        block = greedy_block(uncovered)
        blocks.append(block)
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
        if time.time() > deadline:
            break
    
    # Add randomized restarts to build a diverse candidate pool
    import random
    rng = random.Random(42)
    candidate_blocks = set(tuple(b) for b in blocks)
    
    # Try to add more diverse blocks
    max_candidates = 5000
    attempts = 0
    while len(candidate_blocks) < max_candidates and time.time() < deadline:
        attempts += 1
        if attempts % 100 == 0 and time.time() > deadline:
            break
        # Random seed: pick 2 random points and complete greedily
        seed = list(rng.sample(range(v), min(t, 2)))
        block = greedy_block(set(), block_seed=seed, rng=rng)
        candidate_blocks.add(tuple(block))
    
    candidate_list = [list(b) for b in candidate_blocks]
    n_candidates = len(candidate_list)
    
    # Phase 2: Build incidence matrix and solve ILP
    # A[i, j] = 1 if t-subset i is covered by candidate block j
    A = np.zeros((n_t, n_candidates), dtype=np.int8)
    for j, block in enumerate(candidate_list):
        for sub in itertools.combinations(block, t):
            A[t_to_idx[sub], j] = 1
    
    # ILP: minimize sum(x_j) subject to A @ x >= 1, x in {0,1}
    # Use scipy.optimize.milp
    c = np.ones(n_candidates)  # minimize number of blocks
    
    # Constraints: A @ x >= 1
    # milp uses form: lb <= A_ub @ x <= ub
    # We need A @ x >= 1, which is -A @ x <= -1
    A_ub = -A.astype(np.float64)
    b_ub = -np.ones(n_t)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n_candidates)
    bounds = Bounds(0, 1)
    
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if result.status == 0 and result.x is not None:
        selected = [candidate_list[j] for j in range(n_candidates) if result.x[j] > 0.5]
    else:
        # Fallback to greedy solution
        selected = blocks
    
    # Verify coverage and add any missing blocks if needed
    if time.time() < deadline:
        covered_set = set()
        for block in selected:
            covered_set.update(itertools.combinations(block, t))
        missing = set(all_t_subsets) - covered_set
        while missing and time.time() < deadline:
            block = greedy_block(missing)
            selected.append(block)
            covered_set.update(itertools.combinations(block, t))
            missing -= covered_set
    
    # Remove duplicates
    seen = set()
    final_blocks = []
    for b in selected:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            final_blocks.append(list(b))
    
    return final_blocks
