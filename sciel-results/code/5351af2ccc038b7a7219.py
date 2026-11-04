import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Pre-computation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
    
    # Map each triple to the list of block indices that cover it
    triple_to_blocks = defaultdict(list)
    for b_idx, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_to_blocks[tri_idx].append(b_idx)
    
    def greedy_construct():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            tri_idx = random.choice(list(uncovered))
            candidates = triple_to_blocks[tri_idx]
            
            best_block = None
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for tri in block_triples[b_idx]:
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if gain == len(block_triples[b_idx]):
                        break
            
            if best_block is not None:
                blocks.append(best_block)
                for tri in block_triples[best_block]:
                    if tri in uncovered:
                        uncovered.discard(tri)
            else:
                break
                
        return blocks
    
    def optimize_blocks(blocks):
        if not blocks:
            return []
            
        counts = [0] * n_triples
        for b_idx in blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
                
        active = [True] * len(blocks)
        
        def is_redundant(i):
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                if counts[tri] <= 1:
                    return False
            return True

        def remove_block(i):
            active[i] = False
            b_idx = blocks[i]
            for tri in block_triples[b_idx]:
                counts[tri] -= 1

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                if is_redundant(i):
                    remove_block(i)
                    changed = True
                    
        # Swap optimization
        no_improvement = 0
        
        while time.time() < deadline - 5 and no_improvement < 200:
            # Try removals again
            pass_rem = True
            while pass_rem:
                pass_rem = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if not active[i]:
                        continue
                    if is_redundant(i):
                        remove_block(i)
                        pass_rem = True
                        
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
                
            i_out = random.choice(active_indices)
            b_out = blocks[i_out]
            
            active[i_out] = False
            for tri in block_triples[b_out]:
                counts[tri] -= 1
                
            uncovered_tris = [i for i in range(n_triples) if counts[i] == 0]
            
            if len(uncovered_tris) > 5:
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                no_improvement += 1
                continue
                
            if not uncovered_tris:
                no_improvement = 0
                continue
            
            candidates = set(triple_to_blocks[uncovered_tris[0]])
            for tri in uncovered_tris[1:]:
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
                    
            if candidates:
                current_set = set(blocks[i] for i in range(len(blocks)) if active[i])
                valid = [b for b in candidates if b != b_out and b not in current_set]
                
                if valid:
                    b_in = random.choice(valid)
                    blocks.append(b_in)
                    i_in = len(blocks) - 1
                    active.append(True)
                    for tri in block_triples[b_in]:
                        counts[tri] += 1
                    no_improvement = 0
                    continue
                else:
                    active[i_out] = True
                    for tri in block_triples[b_out]:
                        counts[tri] += 1
                    no_improvement += 1
            else:
                active[i_out] = True
                for tri in block_triples[b_out]:
                    counts[tri] += 1
                no_improvement += 1

        final_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        return final_blocks

    def simulated_annealing(initial_blocks):
        current = list(initial_blocks)
        if not current:
            return []
            
        counts = [0] * n_triples
        for b_idx in current:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
        
        current_obj = len(current)
        best_obj = current_obj
        best = list(current)
        
        temp = 10.0
        cooling = 0.995
        
        sa_deadline = deadline - 5
        no_improve = 0
        
        while time.time() < sa_deadline and temp > 0.1:
            if len(current) == 0:
                break
                
            # Choose random operation
            op = random.random()
            
            if op < 0.5:
                # Try to remove a block
                i = random.randint(0, len(current) - 1)
                b_out = current[i]
                
                new_counts = counts[:]
                for tri in block_triples[b_out]:
                    new_counts[tri] -= 1
                    if new_counts[tri] < 0:
                        new_counts[tri] = 0
                        
                # Count how many triples become uncovered
                uncovered = 0
                for tri_idx in range(n_triples):
                    if counts[tri_idx] > 0 and new_counts[tri_idx] == 0:
                        uncovered += 1
                        
                if uncovered == 0:
                    # Accept: strict improvement or equal
                    current.pop(i)
                    counts = new_counts
                    current_obj -= 1
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best = list(current)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Penalty: uncovered triples
                    delta = -1 + 2 * uncovered
                    if random.random() < np.exp(-delta / max(temp, 0.01)):
                        current.pop(i)
                        counts = new_counts
                        current_obj += uncovered - 1
                        no_improve += 1
                    else:
                        no_improve += 1
                        
            elif op < 0.8:
                # Try to add a block
                # Pick a random uncovered or low-coverage triple
                low_cov = []
                for tri_idx in range(n_triples):
                    if counts[tri_idx] <= 1:
                        low_cov.append(tri_idx)
                        
                if low_cov:
                    tri = random.choice(low_cov)
                    candidates = triple_to_blocks[tri]
                    b_in = random.choice(candidates)
                    
                    current.append(b_in)
                    for tri2 in block_triples[b_in]:
                        counts[tri2] += 1
                    current_obj += 1
                    no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # Try to swap: remove one, add one
                if len(current) > 0:
                    i = random.randint(0, len(current) - 1)
                    b_out = current[i]
                    
                    temp_counts = counts[:]
                    for tri in block_triples[b_out]:
                        temp_counts[tri] -= 1
                        if temp_counts[tri] < 0:
                            temp_counts[tri] = 0
                        
                    uncovered = [ti for ti in range(n_triples) if counts[ti] > 0 and temp_counts[ti] == 0]
                    
                    if not uncovered:
                        # Just remove
                        current.pop(i)
                        counts = temp_counts
                        current_obj -= 1
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Try to find a block covering all uncovered
                        if len(uncovered) <= 3:
                            cands = set(triple_to_blocks[uncovered[0]])
                            for tri in uncovered[1:]:
                                cands.intersection_update(triple_to_blocks[tri])
                                if not cands:
                                    break
                            
                            if cands:
                                # Exclude current blocks
                                cur_set = set(current)
                                valid = [b for b in cands if b not in cur_set]
                                if valid:
                                    b_in = random.choice(valid)
                                    # Remove b_out, add b_in
                                    current.pop(i)
                                    for tri in block_triples[b_in]:
                                        temp_counts[tri] += 1
                                    counts = temp_counts
                                    current_obj = len(current)
                                    no_improve += 1
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                    
            temp *= cooling
            
            if no_improve > 1000:
                # Reset or break
                if random.random() < 0.1:
                    no_improve = 0
                else:
                    break
                    
        return best
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline - 20:
        init = greedy_construct()
        opt = optimize_blocks(init)
        
        if len(opt) < best_len:
            best_len = len(opt)
            best_blocks = list(opt)
            
        if best_len <= 69:
            break
            
        # If we have time, run SA on the best found so far
        if best_blocks and time.time() < deadline - 30:
            sa_result = simulated_annealing(best_blocks)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best_blocks = list(sa_result)
                
    if best_blocks is None:
        best_blocks = greedy_construct()
        best_blocks = optimize_blocks(best_blocks)
        
    # Final cleanup
    unique = []
    seen = set()
    for b in best_blocks:
        if b not in seen:
            seen.add(b)
            unique.append(b)
            
    return [list(all_blocks[b_idx]) for b_idx in unique]
