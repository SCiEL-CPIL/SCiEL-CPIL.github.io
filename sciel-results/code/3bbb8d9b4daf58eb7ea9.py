import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    num_triples = len(all_triples)
    num_blocks = len(all_blocks)
    
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)
            
    def greedy_run(random_seed):
        random.seed(random_seed)
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        blocks_used = []
        
        while uncovered:
            max_gain = -1
            candidates = []
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    candidates = [i]
                elif gains[i] == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            b = random.choice(candidates)
            blocks_used.append(b)
            
            for t_idx in block_triples[b]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        
        return blocks_used

    def improve_solution(initial_blocks):
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in initial_blocks:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
        
        current_blocks = set(initial_blocks)
        
        def is_essential(b_idx):
            for t_idx in block_triples[b_idx]:
                if counts[t_idx] == 1:
                    return True
            return False

        improved = True
        while improved:
            improved = False
            
            removable = []
            for b_idx in current_blocks:
                if not is_essential(b_idx):
                    removable.append(b_idx)
            
            if removable:
                b_to_remove = random.choice(removable)
                current_blocks.remove(b_to_remove)
                for t_idx in block_triples[b_to_remove]:
                    counts[t_idx] -= 1
                improved = True
                continue
            
            candidates_to_swap = []
            for b_idx in current_blocks:
                unique_triples = [t_idx for t_idx in block_triples[b_idx] if counts[t_idx] == 1]
                if not unique_triples:
                    continue
                
                if len(unique_triples) == 1:
                    t0 = unique_triples[0]
                    valid_swaps = [b_prime for b_prime in triple_to_blocks[t0] if b_prime not in current_blocks]
                else:
                    common_blocks = set(triple_to_blocks[unique_triples[0]])
                    for t_idx in unique_triples[1:]:
                        common_blocks &= set(triple_to_blocks[t_idx])
                        if not common_blocks:
                            break
                    valid_swaps = [b_prime for b_prime in common_blocks if b_prime not in current_blocks]
                
                if valid_swaps:
                    candidates_to_swap.append((b_idx, valid_swaps))
            
            if candidates_to_swap:
                b_idx, valid_swaps = random.choice(candidates_to_swap)
                b_prime = random.choice(valid_swaps)
                
                for t_idx in block_triples[b_idx]:
                    counts[t_idx] -= 1
                for t_idx in block_triples[b_prime]:
                    counts[t_idx] += 1
                
                current_blocks.remove(b_idx)
                current_blocks.add(b_prime)
                improved = True
                continue
            
            break
            
        return list(current_blocks)

    start_time = time.time()
    max_time = 110
    
    best_solution = None
    best_size = float('inf')
    
    for i in range(20):
        sol_idx = greedy_run(i)
        improved_sol = improve_solution(sol_idx)
        
        if len(improved_sol) < best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol
            
        if time.time() - start_time > max_time:
            break
            
    if best_solution is None:
        sol_idx = greedy_run(0)
        best_solution = sol_idx
        
    result = [list(all_blocks[i]) for i in best_solution]
    return result
