"""Covering design C(19,7,4) using Simulated Annealing."""
import random
import itertools
import numpy as np
from collections import defaultdict
import time
import math

def _precompute_blocks(v, k, t):
    """Precompute all blocks and their t-subset indices."""
    all_points = range(v)
    blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(blocks)
    
    # Map t-subset to index
    t_subs = list(itertools.combinations(all_points, t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices_list = []
    for b in blocks:
        indices = [t_sub_index[ts] for ts in itertools.combinations(b, t)]
        block_t_indices_list.append(indices)
    
    # Convert to numpy array for fast access
    # Note: variable length arrays are tricky in numpy. 
    # We will store as list of arrays or a flat array with offsets.
    # For C(19,7,4), n_blocks = 50388. Each block covers C(7,4)=35 t-subsets.
    # Total t-subsets per block is constant! So we can use a 2D numpy array.
    block_t_indices_arr = np.array(block_t_indices_list, dtype=np.int32)
    
    # Precompute for each t-subset, which blocks cover it
    # This is useful for initialization or specific heuristics, 
    # but for SA, we mostly need coverage counts.
    
    return blocks, block_t_indices_arr, n_blocks, n_t_subs

def construct(v, k, t):
    start_time = time.time()
    blocks, block_t_indices, n_blocks, n_t_subs = _precompute_blocks(v, k, t)
    
    # coverage_counts: how many blocks in current solution cover each t-subset
    coverage_counts = np.zeros(n_t_subs, dtype=np.int32)
    # in_solution: boolean mask for blocks
    in_solution = np.zeros(n_blocks, dtype=bool)
    
    solution_blocks = [] # List of block indices
    
    # 1. Greedy Initialization (Fast, valid)
    uncovered_mask = np.ones(n_t_subs, dtype=bool)
    
    while np.any(uncovered_mask):
        idx = block_t_indices
        vals = uncovered_mask[idx]
        sums = vals.sum(axis=1)
        
        best_block_idx = np.argmax(sums)
        if sums[best_block_idx] == 0:
            break
            
        solution_blocks.append(best_block_idx)
        in_solution[best_block_idx] = True
        
        t_idxs = block_t_indices[best_block_idx]
        coverage_counts[t_idxs] += 1
        uncovered_mask[t_idxs] = False
        
        if time.time() - start_time > 5:
            break

    best_len = len(solution_blocks)
    best_solution = solution_blocks.copy()
    
    # 2. Simulated Annealing
    T = 1.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # We need to pick random blocks not in solution.
    # Maintaining a list of "not in solution" indices and updating it is O(N) per removal/addition.
    # N is ~50k. This is acceptable if not done too frequently, but we can optimize.
    # Alternatively, just pick random indices and check in_solution. 
    # Given N=50k, probability of collision is low if solution is small (~150).
    # So random sampling with rejection is efficient.
    
    def is_valid_swap(A_idx, B_idx):
        """Check if swapping A (in sol) with B (not in sol) keeps validity."""
        t_idxs_A = block_t_indices[A_idx]
        t_idxs_B = block_t_indices[B_idx]
        
        # A t-subset becomes uncovered if it was only covered by A and is NOT in B.
        # Condition: coverage_counts[t] > 1 OR t in B.
        
        # Get t-subsets in A
        # We need to check for t in A such that coverage_counts[t] == 1 AND t not in B.
        
        # Efficient check:
        # 1. Get t_idxs_A
        # 2. Check which of these have coverage_counts == 1
        # 3. Check if any of those are NOT in t_idxs_B
        
        essential_mask = (coverage_counts[t_idxs_A] == 1)
        if not np.any(essential_mask):
            return True
        
        essential_t_idxs = t_idxs_A[essential_mask]
        
        # Check if essential_t_idxs are subset of t_idxs_B
        # Since t_idxs_B is an array, we can use a set for B for O(1) lookup, 
        # but creating a set every time is expensive.
        # Alternative: sort both? No.
        # Alternative: Use np.isin? 
        # np.isin(essential_t_idxs, t_idxs_B) returns boolean array.
        # If all True, then valid.
        
        if len(essential_t_idxs) == 0:
            return True
            
        # t_idxs_B is int32 array. essential_t_idxs is int32 array.
        # np.isin is reasonably fast for small arrays (max 35 elements).
        if not np.all(np.isin(essential_t_idxs, t_idxs_B)):
            return False
            
        return True

    def do_swap(A_idx, B_idx):
        """Perform swap, updating state."""
        t_idxs_A = block_t_indices[A_idx]
        t_idxs_B = block_t_indices[B_idx]
        
        coverage_counts[t_idxs_A] -= 1
        in_solution[A_idx] = False
        solution_blocks.remove(A_idx) # O(N) list removal. 
        # Optimization: Keep a set or dict for O(1) removal? 
        # Or just swap in the list if we track indices?
        # For simplicity and given N_sol ~ 150, list remove is fine.
        
        coverage_counts[t_idxs_B] += 1
        in_solution[B_idx] = True
        solution_blocks.append(B_idx)

    def do_remove(A_idx):
        """Remove block A, update state."""
        t_idxs_A = block_t_indices[A_idx]
        coverage_counts[t_idxs_A] -= 1
        in_solution[A_idx] = False
        solution_blocks.remove(A_idx)

    def do_add(B_idx):
        """Add block B, update state."""
        t_idxs_B = block_t_indices[B_idx]
        coverage_counts[t_idxs_B] += 1
        in_solution[B_idx] = True
        solution_blocks.append(B_idx)

    # Main SA Loop
    while time.time() - start_time < 110:
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Choose a move type
        # 60% Swap, 30% Remove, 10% Add
        move_type = random.random()
        
        accepted = False
        
        if move_type < 0.6:
            # Swap
            if len(solution_blocks) == 0: continue
            A_idx = random.choice(solution_blocks)
            # Pick random B not in solution
            # Rejection sampling
            while True:
                B_idx = random.randint(0, n_blocks - 1)
                if not in_solution[B_idx]:
                    break
            
            if is_valid_swap(A_idx, B_idx):
                do_swap(A_idx, B_idx)
                accepted = True
                # Cost change is 0. Always accept if valid? 
                # In SA for minimization of count, swap doesn't change count.
                # We accept to explore the landscape.
                
        elif move_type < 0.9:
            # Remove
            if len(solution_blocks) == 0: continue
            A_idx = random.choice(solution_blocks)
            t_idxs_A = block_t_indices[A_idx]
            if np.all(coverage_counts[t_idxs_A] > 1):
                do_remove(A_idx)
                accepted = True
                # Cost decreases by 1. Always accept.
                
        else:
            # Add
            while True:
                B_idx = random.randint(0, n_blocks - 1)
                if not in_solution[B_idx]:
                    break
            do_add(B_idx)
            accepted = True
            # Cost increases by 1. 
            # In standard SA, we might reject this. 
            # But we need to be able to add to escape. 
            # If we always accept adds, we will grow indefinitely.
            # We should only accept adds if we are "stuck" or with some probability?
            # Actually, if we always accept valid moves (swap, remove) and 
            # accept adds with probability exp(-1/T), we can control growth.
            # However, since we track `best_solution`, growing doesn't hurt 
            # as long as we can shrink later. 
            # But it wastes time. 
            # Let's use standard SA acceptance for the "Add" move (delta E = +1).
            if random.random() > math.exp(-1.0 / T):
                # Reject add, revert
                do_remove(B_idx)
                accepted = False
            else:
                # Keep add
                pass

        if accepted:
            if len(solution_blocks) < best_len:
                best_len = len(solution_blocks)
                best_solution = solution_blocks.copy()
                # Optional: Reheat or adjust T? 
                # Simple cooling is fine.
                
    final_blocks = [list(blocks[b_idx]) for b_idx in best_solution]
    return final_blocks
