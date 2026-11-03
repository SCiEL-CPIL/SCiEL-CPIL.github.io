import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety and precomputation
    
    # 1. Precompute t-subsets and mapping
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Precompute k-subsets and their t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    block_t_subs = []
    for block in k_subs:
        subs = [t_sub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_subs.append(subs)
        
    num_blocks = len(k_subs)
    
    best_blocks = None
    best_len = float('inf')
    
    # Parameters for GRASP-like search
    # RCL_SIZE: Number of blocks to consider in each step
    # We can tune this. Larger RCL = better quality per restart, slower per step.
    # Let's start with 5000.
    RCL_SIZE = 5000
    if num_blocks < RCL_SIZE:
        RCL_SIZE = num_blocks
        
    # Run multiple restarts
    while time.time() < deadline:
        restart_start = time.time()
        
        # State
        covered_count = [0] * num_t_subs
        uncovered_count = num_t_subs
        current_blocks = []
        
        # Greedy Construction
        while uncovered_count > 0:
            # Check time
            if time.time() > deadline - 10:
                break
            
            # Build RCL: Random sample of blocks
            # Using random.sample is O(RCL_SIZE)
            if RCL_SIZE == num_blocks:
                rcl_indices = list(range(num_blocks))
            else:
                rcl_indices = random.sample(range(num_blocks), RCL_SIZE)
            
            max_gain = 0
            best_idx = -1
            # To add randomness, we can collect top candidates
            top_candidates = []
            
            # Calculate gain for blocks in RCL
            for idx in rcl_indices:
                gain = 0
                for t_idx in block_t_subs[idx]:
                    if covered_count[t_idx] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    top_candidates = [idx]
                elif gain == max_gain and gain > 0:
                    top_candidates.append(idx)
            
            if max_gain == 0:
                # Should not happen if covering exists and we have all blocks
                # But if RCL is small and we are unlucky?
                # Fallback: scan all blocks? Or just break and restart?
                # If we break, the solution is invalid.
                # We should ensure RCL is large enough or fallback to full scan if stuck.
                # Given 75k blocks and 5k RCL, it's highly unlikely max_gain is 0
                # unless uncovered_count is very small and the specific blocks covering
                # the remaining t-subsets were not in the RCL.
                # To be safe, if max_gain == 0, do a full scan.
                for idx in range(num_blocks):
                    gain = 0
                    for t_idx in block_t_subs[idx]:
                        if covered_count[t_idx] == 0:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        top_candidates = [idx]
                    elif gain == max_gain and gain > 0:
                        top_candidates.append(idx)
                        
                if max_gain == 0:
                    break # Truly stuck or error
            
            # Select randomly from top candidates to diversify
            chosen_idx = random.choice(top_candidates)
            current_blocks.append(k_subs[chosen_idx])
            
            # Update state
            for t_idx in block_t_subs[chosen_idx]:
                if covered_count[t_idx] == 0:
                    uncovered_count -= 1
                covered_count[t_idx] += 1
            
            # Optional: Early termination if we already beat best
            if len(current_blocks) >= best_len:
                break
                
        if len(current_blocks) < best_len and uncovered_count == 0:
            best_len = len(current_blocks)
            best_blocks = current_blocks
            
    return best_blocks
