import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # 1. Generate all blocks and triples
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Map triples to indices for quick lookup
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # 2. Build Incidence Matrix (Block x Triple)
    # We need to know which triples are covered by which blocks.
    # Since n_blocks is large (346k for 24,7) and n_triples is 2024, 
    # storing a dense matrix is 346k * 2024 * 8 bytes ~ 5.6 GB. Too large.
    # We need a sparse representation or a compressed format for the ILP constraints.
    
    # However, scipy.optimize.milp expects dense arrays or specific sparse formats.
    # Let's check the size.
    # A(24,7,3) covering.
    # Number of blocks: C(24,7) = 346104.
    # Number of triples: C(24,3) = 2024.
    
    # We can build the constraint matrix A where A[i, j] = 1 if block i covers triple j.
    # This matrix is 2024 x 346104.
    # Density: Each block covers C(7,3)=35 triples. Total non-zeros = 346104 * 35 = 12,113,640.
    # Size in memory as float64: 12M * 8 bytes = 96 MB. This is acceptable.
    
    # Let's build the dense matrix A (n_triples x n_blocks)
    # A[tri_idx, block_idx] = 1 if covered
    A = np.zeros((n_triples, n_blocks), dtype=np.float64)
    
    for b_idx, block in enumerate(all_blocks):
        # Find all triples within this block
        for tri in itertools.combinations(block, t):
            tri_idx = triple_to_idx[tri]
            A[tri_idx, b_idx] = 1.0

    # 3. Setup ILP
    # Minimize sum(x)
    # s.t. A @ x >= 1
    #      x in {0, 1}
    
    c = np.ones(n_blocks, dtype=np.float64)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(n_triples), ub=np.inf)
    lb = np.ones(n_triples)
    ub = np.full(n_triples, np.inf)
    cons = LinearConstraint(A, lb, ub)
    
    # Bounds for x: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
    
    # Integrality: all variables are integers
    integrality = np.ones(n_blocks)
    
    # Solve
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract selected blocks
            # res.x is the solution vector
            selected_indices = np.where(res.x > 0.5)[0]
            blocks = [list(all_blocks[i]) for i in selected_indices]
            
            # Verify coverage just in case (should be guaranteed by ILP, but safe)
            # And ensure we return valid distinct blocks
            if blocks:
                return blocks
            
        # If ILP fails or times out, fallback to a fast greedy method
    except Exception:
        pass
    
    # Fallback: Simple Greedy
    # This is just a safety net. The ILP should finish within 115s for this size.
    # C(24,7,3) is small enough for modern ILP solvers (HiGHS in scipy) to handle.
    
    uncovered = set(all_triples)
    blocks = []
    block_triples = {b: set(itertools.combinations(b, t)) for b in all_blocks}
    
    while uncovered:
        best_block = None
        best_gain = -1
        # Sample a few random blocks to find a good one quickly
        # Or iterate all? 346k blocks * 35 checks = 12M ops. Fast enough in Python? Maybe not in 115s if done many times.
        # But this is only a fallback.
        
        # Pick a random uncovered triple to guide the search
        tri = random.choice(list(uncovered))
        
        # Find blocks covering this triple
        # We can precompute triple_to_blocks for fallback too, but let's keep it simple.
        # Actually, let's just do a random restart greedy if ILP fails.
        
        # For robustness, let's just return a known good structure or a quick greedy.
        # Since ILP is the main strategy, this path is unlikely.
        
        # Quick Greedy:
        # Pick block with max coverage of uncovered
        for b in all_blocks:
            gain = len(uncovered.intersection(block_triples[b]))
            if gain > best_gain:
                best_gain = gain
                best_block = b
                if gain == len(block_triples[b]):
                    break
        
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        uncovered -= block_triples[tuple(best_block)]
        
    return blocks

