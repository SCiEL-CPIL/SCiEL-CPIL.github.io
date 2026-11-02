import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def random_block_with_triple(tri):
        remaining = [x for x in range(v) if x not in tri]
        ext = random.sample(remaining, k - t)
        return tuple(sorted(tri + ext))

    def random_block():
        return tuple(sorted(random.sample(range(v), k)))

    # 1. Initial Solution: Randomized Greedy
    blocks = []
    uncovered_indices = set(range(n_triples))
    triple_counts = [0] * n_triples
    
    while uncovered_indices:
        tri_idx = random.choice(list(uncovered_indices))
        tri = all_triples[tri_idx]
        b = random_block_with_triple(tri)
        blocks.append(b)
        
        for tri_in_b in itertools.combinations(b, t):
            idx = triple_to_idx[tri_in_b]
            triple_counts[idx] += 1
            if triple_counts[idx] == 1:
                uncovered_indices.discard(idx)
                
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # 2. Simulated Annealing
    current_blocks = list(blocks)
    # triple_counts and uncovered_indices are current
    
    T = 10.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        r = random.random()
        if r < 0.3:
            # Delete
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks)-1)
            b = current_blocks[idx]
            
            old_len = len(current_blocks)
            old_uncovered_len = len(uncovered_indices)
            old_cost = old_len + 10 * old_uncovered_len
            
            # Apply Delete
            for tri in itertools.combinations(b, t):
                t_idx = triple_to_idx[tri]
                triple_counts[t_idx] -= 1
                if triple_counts[t_idx] == 0:
                    uncovered_indices.add(t_idx)
            
            new_len = old_len - 1
            new_uncovered_len = len(uncovered_indices)
            new_cost = new_len + 10 * new_uncovered_len
            
            # Accept?
            if new_cost < old_cost or (random.random() < pow(2.71828, -(new_cost - old_cost) / T)):
                current_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                if not uncovered_indices and new_len < best_len:
                    best_len = new_len
                    best_blocks = list(current_blocks)
            else:
                # Revert
                for tri in itertools.combinations(b, t):
                    t_idx = triple_to_idx[tri]
                    triple_counts[t_idx] += 1
                    if triple_counts[t_idx] == 1:
                        uncovered_indices.discard(t_idx)
                        
        elif r < 0.6:
            # Add
            if uncovered_indices:
                tri_idx = random.choice(list(uncovered_indices))
                tri = all_triples[tri_idx]
                new_b = random_block_with_triple(tri)
            else:
                new_b = random_block()
                
            old_len = len(current_blocks)
            old_uncovered_len = len(uncovered_indices)
            old_cost = old_len + 10 * old_uncovered_len
            
            # Apply Add
            for tri in itertools.combinations(new_b, t):
                t_idx = triple_to_idx[tri]
                triple_counts[t_idx] += 1
                if triple_counts[t_idx] == 1:
                    uncovered_indices.discard(t_idx)
                    
            new_len = old_len + 1
            new_uncovered_len = len(uncovered_indices)
            new_cost = new_len + 10 * new_uncovered_len
            
            # Accept?
            if new_cost < old_cost or (random.random() < pow(2.71828, -(new_cost - old_cost) / T)):
                current_blocks.append(new_b)
                if not uncovered_indices and new_len < best_len:
                    best_len = new_len
                    best_blocks = list(current_blocks)
            else:
                # Revert
                for tri in itertools.combinations(new_b, t):
                    t_idx = triple_to_idx[tri]
                    triple_counts[t_idx] -= 1
                    if triple_counts[t_idx] == 0:
                        uncovered_indices.add(t_idx)
                        
        else:
            # Swap
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks)-1)
            old_b = current_blocks[idx]
            
            if uncovered_indices:
                tri_idx = random.choice(list(uncovered_indices))
                tri = all_triples[tri_idx]
                new_b = random_block_with_triple(tri)
            else:
                new_b = random_block()
                
            old_len = len(current_blocks)
            old_uncovered_len = len(uncovered_indices)
            old_cost = old_len + 10 * old_uncovered_len
            
            # Apply Swap: Remove old, Add new
            for tri in itertools.combinations(old_b, t):
                t_idx = triple_to_idx[tri]
                triple_counts[t_idx] -= 1
                if triple_counts[t_idx] == 0:
                    uncovered_indices.add(t_idx)
                    
            for tri in itertools.combinations(new_b, t):
                t_idx = triple_to_idx[tri]
                triple_counts[t_idx] += 1
                if triple_counts[t_idx] == 1:
                    uncovered_indices.discard(t_idx)
                    
            new_len = old_len
            new_uncovered_len = len(uncovered_indices)
            new_cost = new_len + 10 * new_uncovered_len
            
            # Accept?
            if new_cost < old_cost or (random.random() < pow(2.71828, -(new_cost - old_cost) / T)):
                current_blocks[idx] = new_b
                if not uncovered_indices and new_len < best_len:
                    best_len = new_len
                    best_blocks = list(current_blocks)
            else:
                # Revert: Remove new, Add old
                for tri in itertools.combinations(new_b, t):
                    t_idx = triple_to_idx[tri]
                    triple_counts[t_idx] -= 1
                    if triple_counts[t_idx] == 0:
                        uncovered_indices.add(t_idx)
                        
                for tri in itertools.combinations(old_b, t):
                    t_idx = triple_to_idx[tri]
                    triple_counts[t_idx] += 1
                    if triple_counts[t_idx] == 1:
                        uncovered_indices.discard(t_idx)
                        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Final Cleanup: Greedy Deletion on best_blocks
    final_blocks = list(best_blocks)
    improved = True
    while improved:
        improved = False
        for i in range(len(final_blocks)):
            temp_blocks = final_blocks[:i] + final_blocks[i+1:]
            # Check if valid
            counts = [0] * n_triples
            for b in temp_blocks:
                for tri in itertools.combinations(b, t):
                    counts[triple_to_idx[tri]] += 1
            if all(c > 0 for c in counts):
                final_blocks = temp_blocks
                improved = True
                break
                
    return [list(b) for b in final_blocks]
