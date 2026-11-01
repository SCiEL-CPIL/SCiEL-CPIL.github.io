<<<<<<< SEARCH
import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Precompute for each point x and each (t-1)-subset, which triples are formed
    # For efficiency: for a given block of size b, we need to compute gain for each candidate x
    # gain(x) = number of (t-1)-subsets P of block such that (x, P) forms an uncovered triple
    
    # Precompute: for each point x (0..v-1) and each (t-1)-subset s, the triple index
    # This is too large, so we'll compute on the fly but vectorize
    
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        # uncovered[i] is 1 if triple i is uncovered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        # Precompute all (t-1)-subsets for quick reference
        all_t_minus_1 = list(itertools.combinations(all_points, t - 1))
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Pick the most constrained uncovered triple (fewest candidate blocks)
            # For speed, pick randomly but weight toward uncovered
            seed_idx = None
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices.tolist())
            
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            
            # Extend block to size k using vectorized gain computation
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                
                block_pairs = list(itertools.combinations(block, t - 1))
                candidates = [x for x in all_points if x not in block_set]
                
                if not candidates:
                    break
                
                # Compute gain for each candidate x
                # For each x, gain = sum over (t-1)-subsets P of block of uncovered[triple(x, P)]
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for x_i, x in enumerate(candidates):
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx = triple_to_idx[triple]
                        gains[x_i] += uncovered[idx]
                
                # Pick the candidate with highest gain, break ties randomly
                max_gain = np.max(gains)
                if max_gain <= 0:
                    # No gain, pick randomly
                    block.append(random.choice(candidates))
                else:
                    best_indices = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_indices.tolist())
                    block.append(candidates[chosen])
                
                block_set.add(block[-1])
            
            if len(block) < k:
                # Fill with random points if we ran out of time
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(random.choice(available))
                    block_set.add(block[-1])
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # triple_count[idx] = number of blocks covering triple idx
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                # Update counts
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
                
        return blocks
    
    def local_search_swap(blocks, triple_count):
        """Try to remove a block and replace it with a better one, or just remove if possible."""
        if len(blocks) < 2:
            return blocks
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for i in range(len(blocks)):
                if time.time() > deadline - 10.0:
                    break
                
                # Check if block i is redundant
                is_redundant = True
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
        
        return blocks
    
    # Main loop
    while time.time() < deadline - 5.0:
        blocks = random_greedy_cover()
        if not blocks:
            continue
        
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
    return best_blocks if best_blocks is not None else []
=======
import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Precompute triple indices for all C(k,t) triples within each possible block
    # For gain computation: for each point x and each (t-1)-subset s, precompute the triple index
    # This is v * C(v, t-1) entries which is manageable for v=24, t=3
    # triple_idx[x][s_idx] where s is a (t-1)-subset
    all_t_minus_1 = list(itertools.combinations(all_points, t - 1))
    t_minus_1_to_idx = {s: i for i, s in enumerate(all_t_minus_1)}
    num_t_minus_1 = len(all_t_minus_1)
    
    # triple_idx[x][s_idx] = index of triple (x, s) in all_t_subsets
    triple_idx = np.zeros((v, num_t_minus_1), dtype=np.int32)
    for x in all_points:
        for s_idx, s in enumerate(all_t_minus_1):
            triple = tuple(sorted((x,) + s))
            triple_idx[x, s_idx] = triple_to_idx[triple]
    
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        while True:
            if not np.any(uncovered):
                break
            
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices.tolist())
            
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                
                block_pairs = list(itertools.combinations(block, t - 1))
                candidates = [x for x in all_points if x not in block_set]
                
                if not candidates:
                    break
                
                # Compute gain for each candidate x using precomputed triple_idx
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for x_i, x in enumerate(candidates):
                    for pair in block_pairs:
                        s_idx = t_minus_1_to_idx[tuple(sorted(pair))]
                        idx = triple_idx[x, s_idx]
                        gains[x_i] += uncovered[idx]
                
                max_gain = np.max(gains)
                if max_gain <= 0:
                    block.append(random.choice(candidates))
                else:
                    best_indices = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_indices.tolist())
                    block.append(candidates[chosen])
                
                block_set.add(block[-1])
            
            if len(block) < k:
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(random.choice(available))
                    block_set.add(block[-1])
            
            block.sort()
            blocks.append(block)
            
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
                
        return blocks
    
    def local_search_replace(blocks, time_budget):
        """Try to replace blocks with better ones to reduce total count."""
        if len(blocks) < 2 or time.time() > time_budget:
            return blocks
        
        # Compute triple counts
        triple_count = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            
            for i in range(len(blocks)):
                if time.time() > time_budget:
                    break
                
                # Find triples covered only by block i (critical triples)
                critical = []
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        critical.append(idx)
                
                if not critical:
                    # Block i is fully redundant, just remove it
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all critical triples
                # and maximizes additional coverage
                # Critical triples define constraints: replacement must contain all points in each critical triple
                # Since critical triples are subsets of block i, and block i has k points,
                # the union of all points in critical triples must be in the replacement
                critical_points = set()
                for idx in critical:
                    critical_points.update(all_t_subsets[idx])
                
                if len(critical_points) > k:
                    continue  # Cannot fit all critical points in one block
                
                # Build candidate: must include critical_points, fill rest to size k
                # Try several random completions
                best_gain = 0
                best_block = None
                
                remaining_slots = k - len(critical_points)
                other_points = [x for x in all_points if x not in critical_points]
                
                # Try a few random completions
                for attempt in range(min(5, len(other_points))):
                    if remaining_slots <= 0:
                        candidate = list(critical_points)
                    else:
                        extra = random.sample(other_points, min(remaining_slots, len(other_points)))
                        candidate = list(critical_points) + extra
                    
                    if len(candidate) != k:
                        continue
                    
                    candidate.sort()
                    cand_set = set(candidate)
                    
                    # Check this isn't already in blocks
                    if candidate in [b for b in blocks]:
                        continue
                    
                    # Compute gain: how many new triples does this cover beyond what's already covered?
                    # Specifically, how many triples would become covered that aren't by block i?
                    new_triples = 0
                    for triple in itertools.combinations(candidate, t):
                        idx = triple_to_idx[tuple(triple)]
                        if triple_count[idx] <= 1:
                            # If triple_count[idx] == 1, it's covered by block i (being replaced)
                            # If triple_count[idx] == 0, it's uncovered
                            new_triples += 1
                    
                    # We need to cover all critical triples
                    all_critical_covered = True
                    for idx in critical:
                        if triple_to_idx[all_t_subsets[idx]] is None:
                            pass
                        # Check if critical triple is in candidate
                        crit_triple = all_t_subsets[idx]
                        if not all(p in cand_set for p in crit_triple):
                            all_critical_covered = False
                            break
                    
                    if not all_critical_covered:
                        continue
                    
                    if new_triples > best_gain:
                        best_gain = new_triples
                        best_block = candidate
                
                if best_block is not None:
                    # Replace block i with best_block
                    # Update triple counts
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                    
                    blocks[i] = best_block
                    
                    for triple in itertools.combinations(best_block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] += 1
                    
                    improved = True
                    break
        
        return blocks
    
    def final_polish(blocks, time_budget):
        """High-precision polish: aggressive redundant removal + replacement."""
        if not blocks or time.time() > time_budget:
            return blocks
        
        # First pass: aggressive redundant removal
        blocks = local_search_remove_redundant(blocks)
        
        # Second pass: replacement search with more time
        blocks = local_search_replace(blocks, time_budget - 10.0)
        
        # Third pass: another round of redundant removal after replacements
        blocks = local_search_remove_redundant(blocks)
        
        # Fourth pass: more replacement
        if time.time() < time_budget:
            blocks = local_search_replace(blocks, time_budget)
        
        # Final redundant removal
        blocks = local_search_remove_redundant(blocks)
        
        return blocks
    
    # Main loop
    restarts = 0
    while time.time() < deadline - 15.0:
        blocks = random_greedy_cover()
        if not blocks:
            continue
        
        blocks = local_search_remove_redundant(blocks)
        
        # Try replacement local search if we have time
        if time.time() < deadline - 15.0 and len(blocks) <= best_len:
            blocks = local_search_replace(blocks, deadline - 10.0)
            blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
        
        restarts += 1
    
    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        polished = final_polish([b[:] for b in best_blocks], deadline - 3.0)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished
    
    return best_blocks if best_blocks is not None else []
>>>>>>> REPLACE
