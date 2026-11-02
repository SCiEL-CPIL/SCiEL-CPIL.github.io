"""Covering design C(24,7,3) - ILP-based set cover using PuLP with greedy warm-start."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible blocks
    # A block is a k-subset of {0,...,v-1}
    # For each block, compute which t-subsets it covers
    # There are C(24,7) = 346104 possible blocks
    # Each block covers C(7,3) = 35 t-subsets
    
    # Build incidence structure: for each t-subset, which blocks cover it?
    # Actually, we need the reverse: for each block, which t-subsets does it cover?
    
    # Let's precompute the list of all blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the set of t-subset indices it covers
    block_to_covered = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
        block_to_covered.append(covered)
    
    # Also build reverse: for each t-subset, which blocks cover it
    # This is useful for greedy and for ILP constraints
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)
    
    def greedy_cover(seed=0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            
            # Find all blocks that cover this t-subset
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(all_blocks[best_block])
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        
        return blocks
    
    def try_ilp(initial_blocks, time_limit=100.0):
        """
        Use ILP to minimize the number of blocks.
        Variables: x_j in {0,1} for each block j
        Minimize: sum(x_j)
        Subject to: for each t-subset i, sum(x_j for j covering i) >= 1
        """
        try:
            import pulp
        except ImportError:
            return initial_blocks
        
        # Create ILP model
        prob = pulp.LpProblem("CoveringDesign", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for j in range(n_blocks):
            x[j] = pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary)
        
        # Objective
        prob += pulp.lpSum(x[j] for j in range(n_blocks))
        
        # Constraints: each t-subset must be covered
        for i in range(n_t):
            covering_blocks = t_subset_to_blocks[i]
            if covering_blocks:
                prob += pulp.lpSum(x[j] for j in covering_blocks) >= 1
        
        # Solve with time limit
        solver_time = min(time_limit, deadline - time.time() - 10.0)
        if solver_time <= 0:
            return initial_blocks
            
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time)
        prob.solve(solver)
        
        # Extract solution
        if pulp.LpStatus[prob.status] == "Optimal" or pulp.LpStatus[prob.status] == "Not Solved":
            # If not solved, we may still have a feasible solution
            solution_blocks = []
            for j in range(n_blocks):
                if x[j].value() is not None and x[j].value() > 0.5:
                    solution_blocks.append(all_blocks[j])
            
            # Validate coverage
            if solution_blocks:
                covered = set()
                for block in solution_blocks:
                    for ts in itertools.combinations(block, t):
                        covered.add(t_subset_to_idx[ts])
                if len(covered) == n_t:
                    return solution_blocks
        
        # If ILP didn't give a good solution, return initial
        return initial_blocks
    
    # Initial greedy solution
    best_blocks = greedy_cover(0)
    best_len = len(best_blocks)
    
    # Try ILP to improve
    ilp_result = try_ilp(best_blocks, time_limit=100.0)
    if len(ilp_result) < best_len:
        best_len = len(ilp_result)
        best_blocks = ilp_result
    
    # If we have time, try a few more greedy starts and ILP on the best
    for seed in range(1, 10):
        if time.time() > deadline - 15.0:
            break
        
        init = greedy_cover(seed)
        if len(init) < best_len:
            best_len = len(init)
            best_blocks = init
    
    # Final validation
    if best_blocks is None:
        return []
        
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
            
    if len(covered) != n_t:
        # Fallback to a valid greedy solution
        return greedy_cover(0)
        
    return best_blocks
