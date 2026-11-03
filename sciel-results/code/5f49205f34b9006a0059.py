import itertools
import random
import time
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute the set of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsub_to_idx[ts])
        block_tsubs.append(covered)
    
    # Inverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, covered in enumerate(block_tsubs):
        for t_idx in covered:
            tsub_to_blocks[t_idx].append(b_idx)
    
    # Helper: Try to remove redundant blocks from a solution
    def remove_redundant(current_blocks):
        if not current_blocks:
            return current_blocks
        
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                if check_deadline():
                    return current_blocks
                b_idx = current_blocks[i]
                redundant = True
                for t_idx in block_tsubs[b_idx]:
                    if tsub_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    current_blocks.pop(i)
                    for t_idx in block_tsubs[b_idx]:
                        tsub_count[t_idx] -= 1
                    changed = True
                    break
        
        return current_blocks
    
    # Greedy construction with randomization
    def greedy_construction(seed_offset):
        if check_deadline():
            return None
        
        rng = random.Random(seed_offset)
        
        uncovered = set(range(n_tsubs))
        current_blocks = []
        
        while uncovered:
            if check_deadline():
                return current_blocks
            
            # Pick a random uncovered t-subset as target
            target_tsub_idx = rng.choice(list(uncovered))
            candidate_blocks = tsub_to_blocks[target_tsub_idx]
            
            if not candidate_blocks:
                break
            
            # Score each candidate by how many uncovered tsubs it covers
            best_b_idx = -1
            best_gain = -1
            best_rand = rng.random()
            
            for b_idx in candidate_blocks:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                
                rand_val = rng.random()
                if gain > best_gain or (gain == best_gain and rand_val > best_rand):
                    best_gain = gain
                    best_b_idx = b_idx
                    best_rand = rand_val
            
            if best_b_idx == -1:
                break
            
            current_blocks.append(best_b_idx)
            
            # Update uncovered
            for t_idx in block_tsubs[best_b_idx]:
                uncovered.discard(t_idx)
        
        # Remove redundant blocks
        current_blocks = remove_redundant(current_blocks)
        
        return current_blocks
    
    # Improved WalkSAT-style stochastic local search
    def walksat_sls(current_blocks, max_iterations, seed_offset):
        if not current_blocks or check_deadline():
            return current_blocks
        
        rng = random.Random(seed_offset)
        
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        current_size = len(current_blocks)
        best_size = current_size
        best_blocks = current_blocks.copy()
        
        no_improve_count = 0
        max_no_improve = 50
        
        for iteration in range(max_iterations):
            if check_deadline():
                break
            
            if no_improve_count > max_no_improve:
                break
            
            # Find fragile t-subsets (count == 1)
            fragile_tsubs = [t_idx for t_idx in range(n_tsubs) if tsub_count[t_idx] == 1]
            
            # Choose action
            action = rng.random()
            
            if action < 0.4 and len(current_blocks) > 0:
                # Try deletion or swap
                # Find blocks that can be safely removed (all their tsubs have count > 1)
                safe_to_remove = []
                for i, b_idx in enumerate(current_blocks):
                    safe = True
                    for t_idx in block_tsubs[b_idx]:
                        if tsub_count[t_idx] == 1:
                            safe = False
                            break
                    if safe:
                        safe_to_remove.append(i)
                
                if safe_to_remove:
                    # Remove a safe block
                    remove_idx = rng.choice(safe_to_remove)
                    b_remove = current_blocks[remove_idx]
                    current_blocks.pop(remove_idx)
                    for t_idx in block_tsubs[b_remove]:
                        tsub_count[t_idx] -= 1
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    no_improve_count = 0
                    continue
                
                # Try swap: remove a block and add one that covers all its fragile tsubs
                remove_idx = rng.randrange(len(current_blocks))
                b_remove = current_blocks[remove_idx]
                
                need_cover = [t_idx for t_idx in block_tsubs[b_remove] if tsub_count[t_idx] == 1]
                
                if not need_cover:
                    # Shouldn't happen, but handle it
                    current_blocks.pop(remove_idx)
                    for t_idx in block_tsubs[b_remove]:
                        tsub_count[t_idx] -= 1
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    no_improve_count = 0
                    continue
                
                # Find candidate blocks that cover all need_cover
                candidates = set(tsub_to_blocks[need_cover[0]])
                for t_idx in need_cover[1:]:
                    candidates &= set(tsub_to_blocks[t_idx])
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                # Pick best candidate (prefer those that cover more fragile tsubs overall)
                best_add = -1
                best_score = -float('inf')
                
                for b_add in candidates:
                    if b_add == b_remove:
                        continue
                    score = 0.0
                    for t_idx in block_tsubs[b_add]:
                        if tsub_count[t_idx] == 1:
                            score += 1.0
                    if score > best_score:
                        best_score = score
                        best_add = b_add
                
                if best_add == -1:
                    no_improve_count += 1
                    continue
                
                # Perform the swap
                current_blocks[remove_idx] = best_add
                for t_idx in block_tsubs[b_remove]:
                    tsub_count[t_idx] -= 1
                for t_idx in block_tsubs[best_add]:
                    tsub_count[t_idx] += 1
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                
            elif action < 0.7 and len(current_blocks) >= 2:
                # Pair-replacement: try to replace 2 blocks with 1 block
                i = rng.randrange(len(current_blocks))
                j = rng.randrange(len(current_blocks) - 1)
                if j >= i:
                    j += 1
                
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                # Find tsubs that would become uncovered
                need_cover = []
                for t_idx in block_tsubs[b1]:
                    if tsub_count[t_idx] == 1:
                        need_cover.append(t_idx)
                for t_idx in block_tsubs[b2]:
                    if tsub_count[t_idx] == 1 and t_idx not in need_cover:
                        need_cover.append(t_idx)
                
                if not need_cover:
                    # Both redundant
                    current_blocks.pop(j)
                    current_blocks.pop(i)
                    for t_idx in block_tsubs[b1]:
                        tsub_count[t_idx] -= 1
                    for t_idx in block_tsubs[b2]:
                        tsub_count[t_idx] -= 1
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    no_improve_count = 0
                    continue
                
                # Find a single block that covers all need_cover
                candidates = set(tsub_to_blocks[need_cover[0]])
                for t_idx in need_cover[1:]:
                    candidates &= set(tsub_to_blocks[t_idx])
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                # Pick best candidate
                best_add = -1
                best_score = -float('inf')
                
                for b_add in candidates:
                    if b_add in (b1, b2):
                        continue
                    score = 0.0
                    for t_idx in block_tsubs[b_add]:
                        if tsub_count[t_idx] == 1:
                            score += 1.0
                    if score > best_score:
                        best_score = score
                        best_add = b_add
                
                if best_add == -1:
                    no_improve_count += 1
                    continue
                
                # Perform the replacement
                current_blocks.pop(j)
                current_blocks.pop(i)
                for t_idx in block_tsubs[b1]:
                    tsub_count[t_idx] -= 1
                for t_idx in block_tsubs[b2]:
                    tsub_count[t_idx] -= 1
                current_blocks.append(best_add)
                for t_idx in block_tsubs[best_add]:
                    tsub_count[t_idx] += 1
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                
            else:
                # Random perturbation: add a random block, then try to remove redundant ones
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                
                # Add a random block
                b_add = rng.randrange(n_blocks)
                current_blocks.append(b_add)
                for t_idx in block_tsubs[b_add]:
                    tsub_count[t_idx] += 1
                
                # Try to remove redundant blocks
                changed = True
                while changed:
                    changed = False
                    for i in range(len(current_blocks)):
                        b_idx = current_blocks[i]
                        redundant = True
                        for t_idx in block_tsubs[b_idx]:
                            if tsub_count[t_idx] == 1:
                                redundant = False
                                break
                        if redundant:
                            current_blocks.pop(i)
                            for t_idx in block_tsubs[b_idx]:
                                tsub_count[t_idx] -= 1
                            changed = True
                            break
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        return best_blocks
    
    # Run multiple greedy constructions with different seeds
    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 25
    for restart in range(num_restarts):
        if check_deadline():
            break
        
        seed_offset = restart * 1000 + 42
        current = greedy_construction(seed_offset)
        
        if current is None:
            continue
        
        # WalkSAT SLS
        current = walksat_sls(current, 300, seed_offset + 1000)
        
        if len(current) < best_size:
            best_size = len(current)
            best_blocks = current.copy()
    
    # Final cleanup
    if best_blocks is None:
        return []
    
    best_blocks = remove_redundant(best_blocks)
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return final_blocks
