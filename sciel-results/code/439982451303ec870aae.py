import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
    
    # block_triples[i] is a sorted array of indices of triples covered by all_blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
        block_triples[i] = np.sort(block_triples[i])
    
    # Compute how many blocks cover each triple
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)
    
    # Precompute triple_to_blocks for efficient local search
    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1
    
    def greedy_cover(random_tiebreak=False, weight_power=1.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        # Precompute weighted triple values
        w = triple_weights ** weight_power if weight_power != 1.0 else triple_weights
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Weighted gain: sum of weights of uncovered triples in block
            gains = np.sum(uncovered[block_triples].astype(np.float64) * w[block_triples], axis=1)
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        
        return result
    
    def local_search(block_indices, max_iterations=100):
        """Try to replace blocks with alternatives that maintain coverage."""
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # Check for redundancy
                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break
                    
                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue
                
                critical_triples = block_triples_bi[critical_mask]
                
                # Find candidate blocks that cover ALL critical_triples
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                
                other_critical = critical_triples[critical_triples != min_ct]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        if np.all(np.isin(other_critical, block_triples[cand])):
                            valid_candidates.append(cand)
                
                if valid_candidates:
                    best_cand = valid_candidates[0]
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break
            
            if not improved:
                break
                
        return current
    
    def local_search_remove_two(block_indices, max_iterations=30):
        """Try to remove two blocks at once if their union is still covered."""
        if len(block_indices) < 2:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current)):
                    bi1 = current[i]
                    bi2 = current[j]
                    # Check if removing both leaves all triples covered
                    # A triple is critical if it's covered only by these two blocks
                    both_triples = np.union1d(block_triples[bi1], block_triples[bi2])
                    if np.all(coverage[both_triples] > 2):
                        # Safe to remove both
                        coverage[both_triples] -= 2
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return current
    
    # First greedy pass with different weight powers
    best_solution = None
    best_size = float('inf')
    
    for wp in [1.0, 1.5, 2.0]:
        if time.time() > deadline - 20.0:
            break
        selected = greedy_cover(random_tiebreak=False, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=30)
        selected = local_search_remove_two(selected, max_iterations=10)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]
    
    # Multiple random restarts
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 25
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        wp = random.choice([1.0, 1.5, 2.0, 2.5])
        selected = greedy_cover(random_tiebreak=True, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=20)
        selected = local_search_remove_two(selected, max_iterations=5)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]
    
    # Final polish on best solution
    if best_solution is not None:
        best_indices = []
        for b in best_solution:
            best_indices.append(all_blocks.index(tuple(b)))
        best_indices = local_search(best_indices, max_iterations=50)
        best_indices = remove_redundant(best_indices)
        best_indices = local_search_remove_two(best_indices, max_iterations=10)
        best_solution = [list(all_blocks[i]) for i in best_indices]
        best_size = len(best_solution)
    
    return best_solution
