import itertools
import random
import time
import heapq
import math

def construct(v, k, t):
    # Precomputation
    triples = []
    triple_to_idx = {}
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        triple_to_idx[mask] = len(triples)
        triples.append(mask)
        
    n_triples = len(triples)
    
    blocks = []
    block_to_triples = []
    triple_to_blocks = [[] for _ in range(n_triples)]
    mask_to_block_idx = {}
    
    for comb in itertools.combinations(range(v), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        blocks.append(mask)
        mask_to_block_idx[mask] = len(blocks) - 1
        
        block_triple_indices = []
        for sub_comb in itertools.combinations(comb, t):
            sub_mask = 0
            for x in sub_comb:
                sub_mask |= (1 << x)
            idx = triple_to_idx[sub_mask]
            block_triple_indices.append(idx)
            triple_to_blocks[idx].append(len(blocks) - 1)
            
        block_to_triples.append(block_triple_indices)
        
    n_blocks = len(blocks)
    
    def greedy():
        remaining = [len(bt) for bt in block_to_triples]
        is_uncovered = [1] * n_triples
        solution = []
        
        for _ in range(n_triples):
            if not any(is_uncovered):
                break
            
            max_rem = -1
            best_block = -1
            for i in range(n_blocks):
                if remaining[i] > max_rem:
                    max_rem = remaining[i]
                    best_block = i
            
            if best_block == -1:
                break
                
            solution.append(best_block)
            
            for t_idx in block_to_triples[best_block]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = 0
                    for b_idx in triple_to_blocks[t_idx]:
                        remaining[b_idx] -= 1
                        
        return solution

    def local_search(initial_solution, time_limit):
        start_time = time.time()
        solution = initial_solution[:]
        in_solution = [False] * n_blocks
        for b in solution:
            in_solution[b] = True
            
        cover_count = [0] * n_triples
        for b in solution:
            for t_idx in block_to_triples[b]:
                cover_count[t_idx] += 1
                
        best_solution = solution[:]
        best_count = len(solution)
        
        def is_redundant(b_idx):
            for t_idx in block_to_triples[b_idx]:
                if cover_count[t_idx] == 1:
                    return False
            return True
            
        def remove_block(b_idx):
            for t_idx in block_to_triples[b_idx]:
                cover_count[t_idx] -= 1
            in_solution[b_idx] = False
            solution.remove(b_idx)
            
        def add_block(b_idx):
            for t_idx in block_to_triples[b_idx]:
                cover_count[t_idx] += 1
            in_solution[b_idx] = True
            solution.append(b_idx)
            
        while time.time() - start_time < time_limit:
            changed = True
            while changed:
                changed = False
                for b_idx in solution[:]:
                    if is_redundant(b_idx):
                        remove_block(b_idx)
                        changed = True
                        if len(solution) < best_count:
                            best_count = len(solution)
                            best_solution = solution[:]
                
            if not solution:
                break
                
            b_old_idx = random.choice(solution)
            
            critical_triples = []
            for t_idx in block_to_triples[b_old_idx]:
                if cover_count[t_idx] == 1:
                    critical_triples.append(t_idx)
                    
            if not critical_triples:
                remove_block(b_old_idx)
                continue
                
            union_mask = 0
            for t_idx in critical_triples:
                union_mask |= triples[t_idx]
                
            union_size = union_mask.bit_count()
            if union_size > k:
                continue
                
            available_points = [i for i in range(v) if not (union_mask & (1 << i))]
            
            for _ in range(5):
                if len(available_points) < k - union_size:
                    break
                extra = random.sample(available_points, k - union_size)
                new_mask = union_mask
                for p in extra:
                    new_mask |= (1 << p)
                    
                b_new_idx = mask_to_block_idx.get(new_mask)
                if b_new_idx is None or b_new_idx == b_old_idx or in_solution[b_new_idx]:
                    continue
                    
                # Swap
                for t_idx in block_to_triples[b_old_idx]:
                    cover_count[t_idx] -= 1
                in_solution[b_old_idx] = False
                solution.remove(b_old_idx)
                
                for t_idx in block_to_triples[b_new_idx]:
                    cover_count[t_idx] += 1
                in_solution[b_new_idx] = True
                solution.append(b_new_idx)
                
                # Check for removal
                found_removal = False
                for b_idx in solution[:]:
                    if b_idx == b_new_idx: continue
                    if is_redundant(b_idx):
                        remove_block(b_idx)
                        found_removal = True
                        if len(solution) < best_count:
                            best_count = len(solution)
                            best_solution = solution[:]
                        break
                        
                # If no removal, we keep the swap to explore
                # This might lead to a larger solution temporarily, but redundancy removal
                # at the start of the next loop will clean it up if possible.
                # However, if it doesn't lead to a removal, we might be drifting.
                # But since we always remove redundancies first, the solution size
                # is non-increasing on average? No, swap keeps size same.
                # So size is non-increasing.
                pass

        return best_solution

    init_sol = greedy()
    time_limit = 100
    best_blocks_idx = local_search(init_sol, time_limit)
    
    result = []
    for b_idx in best_blocks_idx:
        mask = blocks[b_idx]
        block = [i for i in range(v) if (mask & (1 << i))]
        result.append(block)
        
    return result
