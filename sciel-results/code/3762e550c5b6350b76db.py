"""Covering design C(19,8,5): Simulated Annealing local search.
Starts with a fast greedy covering, then minimizes block count using
drop and swap moves with simulated annealing acceptance criteria.
"""
import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # Precompute t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets
    # There are C(19,8) = 75582 possible blocks.
    # For each block, store the list of indices of t-subsets it covers.
    all_blocks = list(itertools.combinations(range(v), k))
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Convert to numpy arrays for faster access? 
    # Actually, lists of lists are fine, but let's keep it simple.
    # We need a way to quickly find a block that covers a specific set of uncovered subsets.
    # Invert the index: for each t-subset index, list of blocks that cover it.
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    def greedy_cover():
        """Fast greedy to get an initial valid solution."""
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            # Pick a random uncovered subset
            seed_idx = next(iter(uncovered))
            # Pick a random block covering it
            candidates = t_to_blocks[seed_idx]
            if not candidates:
                # Should not happen
                break
            best_b = None
            best_gain = -1
            # Sample a few candidates to speed up, or just pick one?
            # Greedy: pick block with max coverage of remaining uncovered
            # To keep it fast, we might just pick a random one or a few.
            # Let's do a mini-greedy: pick the one covering most uncovered.
            # But that's expensive if we do it for every step.
            # Let's just pick a random candidate to start, then refine?
            # No, let's do a simple greedy:
            # Actually, the current program's greedy is decent. 
            # Let's reuse the logic from the prompt's seed but simplified.
            
            # Pick a random uncovered t-subset
            s = all_t[seed_idx]
            # We need a block containing s.
            # Find candidates containing s.
            # We precomputed t_to_blocks for single t-subsets.
            # But we need blocks containing the whole s (which is a t-subset).
            # Wait, t_to_blocks[ti] gives blocks covering the t-subset ti.
            # So candidates = t_to_blocks[seed_idx] is correct.
            
            # To make it better, let's pick the block that covers the most *currently uncovered* subsets.
            # This is O(num_candidates * t_subsets_per_block).
            # num_candidates is roughly C(v-t, k-t) = C(14, 3) = 364.
            # t_subsets_per_block is C(8,5) = 56.
            # 364 * 56 is small. So we can do full greedy.
            
            best_b = -1
            max_cov = -1
            for bi in candidates:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
                    
            if best_b == -1:
                # Fallback
                best_b = candidates[0]
                
            blocks.append(best_b)
            # Update uncovered
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                
        return blocks

    # Initial solution
    init_blocks = greedy_cover()
    
    # State for SA
    # We maintain counts of how many times each t-subset is covered.
    # cnt[i] = number of blocks covering t-subset i.
    # A solution is valid if min(cnt) >= 1.
    
    def compute_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    # We will run multiple restarts of SA
    best_blocks = init_blocks
    best_score = len(best_blocks)
    
    rng = random.Random(42)
    
    # SA parameters
    # Cost function: penalty for invalidity + block count
    # We want to minimize: len(blocks) + penalty * (number of uncovered subsets)
    # If we start valid, we prefer moves that keep it valid or reduce size.
    # Standard SA on "cost = len(blocks) if valid else len(blocks) + large_penalty".
    
    while time.time() < deadline - 5.0:
        # Restart SA
        blocks = list(init_blocks) # Start from greedy each time? Or keep best?
        # Let's start from the current best to potentially improve, 
        # but also do some from scratch to explore.
        # Actually, starting from greedy every time is safer to avoid getting stuck in bad local optima of SA.
        # But SA itself explores. Let's start from best found so far.
        blocks = list(best_blocks)
        
        cnt = compute_counts(blocks)
        # Check validity
        min_cnt = np.min(cnt)
        if min_cnt < 1:
            # Should not happen if best_blocks is valid
            pass
            
        current_cost = len(blocks)
        best_local_cost = len(blocks)
        best_local_blocks = list(blocks)
        
        T = 10.0
        cooling = 0.999
        
        # Number of steps per restart
        # We have ~110s total. 
        # Let's run SA for a fixed number of steps or until time.
        # Let's do 5000 steps per restart, see how many fit.
        
        steps_in_restart = 0
        max_steps = 10000
        
        while steps_in_restart < max_steps and time.time() < deadline - 2.0:
            steps_in_restart += 1
            
            # Generate move
            # Move types:
            # 1. Drop a block (if it doesn't break validity)
            # 2. Swap a block (remove one, add another)
            # 3. Add a block (rarely useful unless we allow intermediate invalid states to escape?)
            #    Usually, for minimization, we focus on Drop and Swap.
            
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try Drop
                if len(blocks) == 0:
                    continue
                bi = rng.randrange(len(blocks))
                block_to_remove = blocks[bi]
                
                # Check if removing it keeps validity
                # A block can be removed if all its t-subsets are covered by other blocks.
                # i.e., cnt[ti] > 1 for all ti in block_cover[block_to_remove]
                can_remove = True
                for ti in block_cover[block_to_remove]:
                    if cnt[ti] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Apply move
                    blocks.pop(bi)
                    for ti in block_cover[block_to_remove]:
                        cnt[ti] -= 1
                    new_cost = len(blocks)
                    delta_cost = new_cost - current_cost
                    # Delta is always negative (improvement)
                    
                    # Accept
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_blocks = list(blocks)
                        
                    T *= cooling
                    continue
                else:
                    # Cannot remove directly.
                    # Maybe we can swap?
                    pass
            else:
                # Try Swap
                if len(blocks) == 0:
                    continue
                bi = rng.randrange(len(blocks))
                block_old = blocks[bi]
                
                # Find a block to add.
                # Strategy: Pick a t-subset that is "weakly" covered (cnt=1) and try to find a block covering it that also covers other weak subsets?
                # Or just pick a random block?
                # Better: Pick a random t-subset covered by block_old.
                # If we remove block_old, some subsets might become uncovered (those with cnt=1).
                # Let U be the set of t-subsets in block_old with cnt=1.
                # We need a new block that covers all of U.
                # If U is empty, we can just drop (handled above).
                # If U is not empty, find a block covering U.
                
                U = []
                for ti in block_cover[block_old]:
                    if cnt[ti] == 1:
                        U.append(ti)
                        
                if not U:
                    # This is a drop case, handled by the 30% branch mostly, 
                    # but if we are in swap branch, we can still try to drop.
                    # Let's just treat as drop.
                    blocks.pop(bi)
                    for ti in block_cover[block_old]:
                        cnt[ti] -= 1
                    new_cost = len(blocks)
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_blocks = list(blocks)
                    T *= cooling
                    continue
                
                # Find blocks covering all of U.
                # Intersection of t_to_blocks[ti] for ti in U.
                # Start with candidates from the first ti in U.
                if len(U) > 0:
                    candidates = set(t_to_blocks[U[0]])
                    for ti in U[1:]:
                        candidates &= set(t_to_blocks[ti])
                    
                    if not candidates:
                        # No block can cover all weak subsets. Swap not possible directly.
                        # We might need to add a block first? 
                        # Or accept a temporary increase in cost?
                        # For now, skip.
                        T *= cooling
                        continue
                        
                    # Pick a random candidate from those
                    # Exclude the one we are removing?
                    # block_old is in candidates? 
                    # block_old covers U, so yes.
                    # We want a DIFFERENT block.
                    candidates.discard(block_old)
                    
                    if not candidates:
                        T *= cooling
                        continue
                        
                    block_new = rng.choice(list(candidates))
                    
                    # Calculate cost change
                    # Remove block_old, add block_new.
                    # Size stays same. Cost stays same (if valid).
                    # But validity might change?
                    # We ensured U is covered by block_new.
                    # What about subsets NOT in U?
                    # Subsets in block_old with cnt > 1 remain covered.
                    # Subsets in block_new:
                    #   - If in block_old: cnt changes -1 + 1 = 0.
                    #   - If not in block_old: cnt changes +1.
                    # So validity is preserved.
                    
                    # Apply move
                    blocks[bi] = block_new
                    for ti in block_cover[block_old]:
                        cnt[ti] -= 1
                    for ti in block_cover[block_new]:
                        cnt[ti] += 1
                        
                    new_cost = len(blocks)
                    delta_cost = new_cost - current_cost
                    # Delta is 0.
                    # We accept with prob 1 if delta <= 0, else exp(-delta/T).
                    # Here delta = 0, so accept.
                    # However, this move doesn't change cost. 
                    # It changes the structure, potentially allowing future drops.
                    # So we accept it.
                    
                    current_cost = new_cost
                    # No improvement in cost, so best_local_cost unchanged.
                    
                    T *= cooling
                    continue
            # End if/else move type
            
        # End SA restart
        if best_local_cost < best_score:
            best_score = best_local_cost
            best_blocks = best_local_blocks
            # print(f"New best: {best_score}")
            
    # Final check and return
    # Ensure best_blocks is valid
    # It should be, but let's verify.
    cnt = compute_counts(best_blocks)
    if np.min(cnt) < 1:
        # Fallback to greedy
        return [list(all_blocks[bi]) for bi in greedy_cover()]
        
    return [list(all_blocks[bi]) for bi in best_blocks]
