"""New program using Simulated Annealing for Covering Design."""
import itertools
import random
import numpy as np
from collections import defaultdict
from math import comb
import time

def construct(v, k, t):
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subsets to indices for coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute: for each block, which t-subsets it covers
    block_to_tsubs = []
    tsub_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    def greedy_init():
        """Run standard greedy to get a starting solution."""
        covered = np.zeros(n_tsubs, dtype=bool)
        gain = np.full(n_blocks, comb(k, t), dtype=int)
        selected = []
        
        while True:
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = candidates[0]
            selected.append(b_idx)
            
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    for other_b in tsub_to_blocks[ts_idx]:
                        gain[other_b] -= 1
        return selected

    # Initial solution
    initial_selected = greedy_init()
    in_solution = np.zeros(n_blocks, dtype=bool)
    for b in initial_selected:
        in_solution[b] = True
        
    # Calculate initial cover counts
    cover_count = np.zeros(n_tsubs, dtype=int)
    for b in initial_selected:
        for ts_idx in block_to_tsubs[b]:
            cover_count[ts_idx] += 1
            
    best_len = len(initial_selected)
    best_solution = initial_selected.copy()
    current_solution = initial_selected.copy()
    
    # Count uncovered t-subsets (should be 0 for greedy)
    uncovered_count = np.sum(cover_count == 0)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    # We want to minimize: size + penalty * uncovered
    # If uncovered > 0, we must repair.
    # We will allow moves that change size.
    
    T = 1.0 # Temperature
    cooling_rate = 0.999 # Slow cooling
    min_T = 0.01
    
    # To speed up, we can use a "restart" strategy if stuck
    # But let's stick to standard SA first.
    
    def apply_move(b_idx, add):
        """
        Apply adding or removing a block.
        Returns change in uncovered count.
        """
        nonlocal uncovered_count
        delta_uncovered = 0
        if add:
            in_solution[b_idx] = True
            for ts_idx in block_to_tsubs[b_idx]:
                if cover_count[ts_idx] == 0:
                    delta_uncovered -= 1
                cover_count[ts_idx] += 1
        else:
            in_solution[b_idx] = False
            for ts_idx in block_to_tsubs[b_idx]:
                cover_count[ts_idx] -= 1
                if cover_count[ts_idx] == 0:
                    delta_uncovered += 1
        uncovered_count += delta_uncovered
        return delta_uncovered

    def get_cost():
        # Cost = size + large_penalty * uncovered
        # We prioritize valid solutions.
        return len(current_solution) + 1000 * uncovered_count

    current_cost = get_cost()
    best_cost = current_cost
    
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a random move
        # 1. Add a block not in solution
        # 2. Remove a block in solution
        # 3. Swap
        
        move_type = rng.randint(0, 2)
        
        if move_type == 0: # Add
            # Pick random block not in solution
            # To avoid bias, we can pick from all blocks and check
            b_idx = rng.randint(0, n_blocks - 1)
            if in_solution[b_idx]:
                continue
            
            # Tentatively add
            # We need to track size change
            old_size = len(current_solution)
            current_solution.append(b_idx)
            delta_uncov = apply_move(b_idx, True)
            
            new_cost = len(current_solution) + 1000 * uncovered_count
            delta_cost = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                current_cost = new_cost
                if uncovered_count == 0 and len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = current_solution.copy()
            else:
                # Reject
                current_solution.pop()
                apply_move(b_idx, False)
                
        elif move_type == 1: # Remove
            if not current_solution:
                continue
            b_idx = current_solution[rng.randint(0, len(current_solution) - 1)]
            
            # Tentatively remove
            current_solution.remove(b_idx)
            delta_uncov = apply_move(b_idx, False)
            
            new_cost = len(current_solution) + 1000 * uncovered_count
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                current_cost = new_cost
                if uncovered_count == 0 and len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = current_solution.copy()
            else:
                # Reject
                current_solution.append(b_idx)
                apply_move(b_idx, True)
                
        else: # Swap
            if not current_solution:
                continue
            # Pick one in, one out
            b_in = current_solution[rng.randint(0, len(current_solution) - 1)]
            b_out = rng.randint(0, n_blocks - 1)
            if b_out == b_in or in_solution[b_out]:
                continue
            
            # Remove b_in, Add b_out
            current_solution.remove(b_in)
            apply_move(b_in, False)
            
            current_solution.append(b_out)
            apply_move(b_out, True)
            
            # Size is same, so cost change is due to uncovered
            new_cost = len(current_solution) + 1000 * uncovered_count
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                current_cost = new_cost
                if uncovered_count == 0 and len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = current_solution.copy()
            else:
                # Reject
                current_solution.remove(b_out)
                apply_move(b_out, False)
                
                current_solution.append(b_in)
                apply_move(b_in, True)
                
        T *= cooling_rate

    # Final cleanup: try to remove redundant blocks from best_solution
    final_selected = best_solution.copy()
    in_final = np.zeros(n_blocks, dtype=bool)
    for b in final_selected:
        in_final[b] = True
        
    cover_final = np.zeros(n_tsubs, dtype=int)
    for b in final_selected:
        for ts_idx in block_to_tsubs[b]:
            cover_final[ts_idx] += 1
            
    i = 0
    while i < len(final_selected):
        b_idx = final_selected[i]
        is_redundant = True
        for ts_idx in block_to_tsubs[b_idx]:
            if cover_final[ts_idx] == 1:
                is_redundant = False
                break
        if is_redundant:
            in_final[b_idx] = False
            for ts_idx in block_to_tsubs[b_idx]:
                cover_final[ts_idx] -= 1
            final_selected.pop(i)
        else:
            i += 1
            
    return [list(all_blocks[i]) for i in final_selected]
