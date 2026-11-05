"""Covering design C(19,7,4) - minimize number of 7-element blocks covering all 4-subsets."""
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    
    # Greedy initial solution
    def greedy_cover():
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered and time.time() < deadline - 5:
            # Pick the first uncovered t-subset as seed
            seed = min(uncovered)
            block = list(seed)
            # Greedily add points that maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain == 0:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            # Remove all t-subsets covered by this block
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
        return blocks
    
    blocks = greedy_cover()
    
    if not blocks:
        return [list(b) for b in blocks]
    
    # Check if we have a valid cover
    def is_valid_cover(blk_list):
        covered = set()
        for b in blk_list:
            covered.update(itertools.combinations(b, t))
        return covered == all_tsubsets_set
    
    if not is_valid_cover(blocks):
        # Fallback: return what we have (should be valid from greedy)
        return [list(b) for b in blocks]
    
    # Simulated annealing to reduce block count
    # State: list of blocks (tuples)
    # Move: remove a block, try to re-cover its unique t-subsets with modifications to other blocks
    # Or: swap a block with a new one that covers more uncovered
    
    def count_uncovered(blk_list):
        covered = set()
        for b in blk_list:
            covered.update(itertools.combinations(b, t))
        return len(all_tsubsets_set - covered)
    
    def energy(blk_list):
        """Energy = number of blocks + penalty for uncovered t-subsets"""
        unc = count_uncovered(blk_list)
        if unc > 0:
            return len(blk_list) + unc * 100  # heavy penalty for invalid
        return len(blk_list)
    
    best_blocks = list(blocks)
    best_energy = energy(best_blocks)
    current_blocks = list(blocks)
    current_energy = best_energy
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995  # cooling rate
    
    def try_improve(blk_list, max_steps=200):
        """Try to reduce blocks by finding redundant or swappable blocks"""
        current = list(blk_list)
        cur_energy = energy(current)
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            if len(current) <= 1:
                break
            
            # Try removing a random block and see if cover remains valid
            idx = random.randrange(len(current))
            candidate = current[:idx] + current[idx+1:]
            if is_valid_cover(candidate):
                # Valid! Accept
                current = candidate
                cur_energy = energy(current)
                continue
            
            # Try swapping: replace block at idx with a new block
            # Generate candidate block that covers some of the unique t-subsets of removed block
            removed = current[idx]
            # Find t-subsets only covered by this block
            covered_by_others = set()
            for j, b in enumerate(current):
                if j == idx:
                    continue
                covered_by_others.update(itertools.combinations(b, t))
            
            unique_to_removed = set(itertools.combinations(removed, t)) - covered_by_others
            
            if not unique_to_removed:
                # Block is redundant
                current = candidate
                cur_energy = energy(current)
                continue
            
            # Try to find a new block that covers many of unique_to_removed
            # Seed with a random t-subset from unique_to_removed
            if unique_to_removed:
                seed_ts = random.choice(list(unique_to_removed))
                new_block = list(seed_ts)
                # Greedily extend
                for _ in range(k - t):
                    best_x = None
                    best_gain = -1
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block, t-1):
                            ts = tuple(sorted(rest + (x,)))
                            if ts in unique_to_removed:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None or best_gain == 0:
                        break
                    new_block.append(best_x)
                new_block.sort()
                new_block = tuple(new_block)
                
                candidate = current[:idx] + [new_block] + current[idx+1:]
                cand_energy = energy(candidate)
                
                # Accept if better, or with probability exp(-(delta)/T)
                delta = cand_energy - cur_energy
                if delta <= 0:
                    current = candidate
                    cur_energy = cand_energy
                else:
                    # Use a fixed temperature for this inner loop
                    prob = min(1.0, max(0.0, 1.0 - delta / 5.0))
                    if random.random() < prob:
                        current = candidate
                        cur_energy = cand_energy
        
        return current, cur_energy
    
    # Main SA loop
    T = T_start
    iteration = 0
    while time.time() < deadline - 5 and T > T_end:
        iteration += 1
        
        # Perturb: try to improve
        perturbed, pert_energy = try_improve(current_blocks, max_steps=50)
        
        if pert_energy < current_energy:
            current_blocks = perturbed
            current_energy = pert_energy
            if current_energy < best_energy:
                best_blocks = list(current_blocks)
                best_energy = current_energy
        elif random.random() < min(1.0, max(0.0, 1.0 - (pert_energy - current_energy) / T)):
            current_blocks = perturbed
            current_energy = pert_energy
        
        T *= alpha
    
    # Final check
    if is_valid_cover(best_blocks):
        return [list(b) for b in best_blocks]
    else:
        # Fallback to original
        return [list(b) for b in blocks]
