import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None):
        """Run greedy set cover with optional weighting and random tie-breaking."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak and rng is not None:
                candidates = np.where(gains == max_gain)[0]
                best_idx = rng.choice(candidates)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected
    
    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution using vectorized checks."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        keep = []
        for i, bidx in enumerate(block_indices):
            my_triples = block_triples[bidx]
            # Check if any triple in this block has coverage <= 1
            redundant = np.all(coverage_count[my_triples] > 1)
            if redundant:
                coverage_count[my_triples] -= 1
            else:
                keep.append(bidx)
        
        return keep
    
    def simulated_annealing(block_indices, rng, max_iters=5000, T_start=1.0, T_end=0.01, cooling=0.9995):
        """Enhanced simulated annealing with diverse moves to minimize block count."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        T = T_start
        iter_count = 0
        while time.time() < deadline - 5.0 and iter_count < max_iters and T > T_end:
            iter_count += 1
            T *= cooling
            
            if len(current_blocks) == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.25:
                # Move 1: Try to remove a redundant block
                i = rng.randrange(len(current_blocks))
                bidx = current_blocks[i]
                my_triples = block_triples[bidx]
                redundant = np.all(coverage_count[my_triples] > 1)
                
                if redundant:
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            elif move_type < 0.55:
                # Move 2: Swap one block with another (maintain size)
                i = rng.randrange(len(current_blocks))
                old_bidx = current_blocks[i]
                old_triples = block_triples[old_bidx]
                
                critical_mask = (coverage_count == 1)
                critical_triples = old_triples[critical_mask[old_triples]]
                
                if len(critical_triples) == 0:
                    continue
                
                # Find candidate blocks that cover all critical triples
                best_candidate = -1
                best_gain = -1
                
                n_candidates = min(200, n_blocks)
                cand_indices = rng.sample(range(n_blocks), n_candidates)
                
                for cand_idx in cand_indices:
                    if cand_idx == old_bidx:
                        continue
                    cand_triples = block_triples[cand_idx]
                    
                    covers_all = True
                    for ct in critical_triples:
                        if not np.isin(ct, cand_triples):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    new_coverage = 0
                    for ct in cand_triples:
                        if coverage_count[ct] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_candidate = cand_idx
                
                if best_candidate >= 0:
                    old_triples_list = old_triples.tolist()
                    cand_triples_list = block_triples[best_candidate].tolist()
                    
                    for tr in old_triples_list:
                        coverage_count[tr] -= 1
                    for tr in cand_triples_list:
                        coverage_count[tr] += 1
                    
                    current_blocks[i] = best_candidate
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            elif move_type < 0.75:
                # Move 3: Add a block, then try to remove two blocks (net -1)
                if len(current_blocks) > 1:
                    # Pick a random block to add
                    add_idx = rng.randrange(n_blocks)
                    if add_idx not in current_blocks:
                        coverage_count[block_triples[add_idx]] += 1
                        current_blocks.append(add_idx)
                        
                        # Now try to remove two blocks
                        removed_count = 0
                        for _ in range(2):
                            if not current_blocks:
                                break
                            i = rng.randrange(len(current_blocks))
                            bidx = current_blocks[i]
                            my_triples = block_triples[bidx]
                            redundant = np.all(coverage_count[my_triples] > 1)
                            if redundant:
                                coverage_count[my_triples] -= 1
                                current_blocks.pop(i)
                                removed_count += 1
                        
                        if removed_count >= 1:
                            if len(current_blocks) < best_size:
                                best_size = len(current_blocks)
                                best_blocks = list(current_blocks)
            
            else:
                # Move 4: Perturbation - remove random block and repair greedily
                if len(current_blocks) > 1:
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    coverage_count[block_triples[removed_bidx]] -= 1
                    current_blocks.pop(j)
                    
                    uncovered_mask = (coverage_count == 0)
                    while np.any(uncovered_mask):
                        if time.time() > deadline - 5.0:
                            break
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        
        return best_blocks
    
    def final_polish(block_indices, rng, time_budget=10.0):
        """High-precision polish: intensive local search on the best solution."""
        if not block_indices:
            return list(block_indices)
        
        start = time.time()
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Intensive iterative improvement
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i in range(len(current_blocks)):
                if time.time() - start > time_budget:
                    break
                bidx = current_blocks[i]
                my_triples = block_triples[bidx]
                if np.all(coverage_count[my_triples] > 1):
                    coverage_count[my_triples] -= 1
                    current_blocks.pop(i)
                    i -= 1
                    improved = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        
        # Run SA with tighter parameters
        if time.time() - start < time_budget:
            remaining = time_budget - (time.time() - start)
            sa_result = simulated_annealing(best_blocks, rng, max_iters=3000, 
                                            T_start=0.5, T_end=0.05, cooling=0.9997)
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_blocks = list(sa_result)
        
        return best_blocks
    
    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Strategy 3+: Random restarts with weighted greedy + simulated annealing
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        use_weighted = (r % 2 == 0)
        sel = greedy_cover(use_weighted=use_weighted, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply simulated annealing
            if time.time() < deadline - 15.0:
                sa_result = simulated_annealing(peeled, rng, max_iters=2000)
                sa_size = len(sa_result)
                if sa_size < size:
                    size = sa_size
                    peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
    
    # Final polish: intensive local search on the best solution found
    if best_solution is not None and time.time() < deadline - 10.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for b in best_solution:
            t = tuple(b)
            if t in block_to_idx:
                current_indices.append(block_to_idx[t])
        
        if current_indices:
            # Remove duplicates
            seen = set()
            unique_indices = []
            for idx in current_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            polish_result = final_polish(unique_indices, rng, time_budget=8.0)
            if len(polish_result) < len(best_solution):
                best_solution = [list(all_blocks[i]) for i in polish_result]
                best_size = len(polish_result)
    
    # Final deduplication
    if best_solution:
        seen = set()
        unique_blocks = []
        for b in best_solution:
            t = tuple(b)
            if t not in seen:
                seen.add(t)
                unique_blocks.append(b)
        best_solution = unique_blocks
        best_size = len(best_solution)
    
    if best_solution is None:
        best_solution = []
    
    return best_solution
