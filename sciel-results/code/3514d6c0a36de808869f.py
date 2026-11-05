import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Fallback: Greedy construction
    def greedy_fallback():
        random.seed(42)
        uncovered = set(all_triplets)
        selected = []
        
        # Precompute mask for each block as a set of triplet IDs for speed in fallback
        # Actually, just use sets of triplets for simplicity in fallback, 
        # but for v=24, k=7, t=3, a block covers C(7,3)=35 triplets.
        # Total triplets C(24,3)=2024.
        
        # To make fallback fast, we use bitmasks if possible, but let's stick to simple sets
        # or precomputed lists.
        
        # Let's use a slightly smarter greedy: pick block covering most uncovered.
        # Since n_blocks is ~73k, and we do this ~70 times, that's 5 million ops. Fine.
        
        block_triplet_sets = []
        for b in all_blocks:
            t_set = set(itertools.combinations(b, t))
            block_triplet_sets.append(t_set)
            
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            
            # Randomly shuffle order to break ties and allow diversity if we restart
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            for i in indices:
                gain = len(uncovered & block_triplet_sets[i])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    
            if best_block_idx == -1:
                break # Should not happen
                
            selected.append(all_blocks[best_block_idx])
            uncovered -= block_triplet_sets[best_block_idx]
            
        return selected

    # Try CP-SAT
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        # If OR-Tools is not available, use the fallback
        return greedy_fallback()

    model = cp_model.CpModel()
    
    # Binary variable for each block
    block_vars = [model.NewBoolVar(f'block_{i}') for i in range(n_blocks)]
    
    # Objective: minimize number of blocks
    model.Minimize(sum(block_vars))
    
    # Constraints: each triplet must be covered by at least one block
    # For each triplet, find all blocks containing it.
    # To avoid iterating through all blocks for each triplet (2024 * 73628 ~ 150M ops),
    # we precompute the mapping: triplet_id -> list of block indices containing it.
    
    # This precomputation might be slow in Python. 
    # C(24,3) = 2024 triplets.
    # For each triplet, how many blocks contain it?
    # A block of size 7 contains a specific triplet if the other 4 points are chosen from the remaining 21.
    # Number of blocks containing a specific triplet = C(21, 4) = 5985.
    # Total constraints entries = 2024 * 5985 ~ 12 million.
    # Building this list in Python might take a few seconds.
    
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    
    # Efficient way to populate triplet_to_blocks:
    # Iterate through all blocks, and for each block, iterate through its triplets.
    # Block i contains triplets T. For each t in T, append i to triplet_to_blocks[id(t)].
    # Total iterations: n_blocks * C(k, t) = 73628 * 35 ~ 2.5 million. This is fast.
    
    for i, block in enumerate(all_blocks):
        for trip in itertools.combinations(block, t):
            tid = triplet_to_id[trip]
            triplet_to_blocks[tid].append(i)
            
    # Add constraints
    for tid in range(n_triplets):
        blocks_covering = triplet_to_blocks[tid]
        if not blocks_covering:
            # This shouldn't happen for v=24, k=7, t=3
            return greedy_fallback()
        model.Add(sum(block_vars[i] for i in blocks_covering) >= 1)
        
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = deadline - time.time() - 5 # Leave 5s for extraction
    solver.parameters.num_workers = 8 # Use multiple threads if available
    solver.parameters.random_seed = 42
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_blocks = []
        for i in range(n_blocks):
            if solver.Value(block_vars[i]) == 1:
                selected_blocks.append(all_blocks[i])
                
        # Verify validity just in case (solver should be correct, but safety first)
        # Quick check: ensure no duplicates and all distinct
        if len(selected_blocks) != len(set(map(tuple, selected_blocks))):
             return greedy_fallback()
             
        # Exhaustive check is expensive, but we trust the solver's constraints.
        # However, the task says "checked exhaustively" by the evaluator.
        # We should ensure our output is valid.
        # Let's do a fast validation using counts.
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in selected_blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
                
        if np.min(counts) < 1:
             # Solver failed or bug? Fallback.
             return greedy_fallback()
             
        return selected_blocks
    else:
        # Solver didn't find a solution in time or failed
        return greedy_fallback()

if __name__ == "__main__":
    pass
