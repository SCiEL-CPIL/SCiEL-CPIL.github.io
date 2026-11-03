import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            if a == b: continue
            for c in range(v):
                if a == c or b == c: continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
    
    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]
    
    def greedy_cover(uncovered_mask=None, random_state=0, aggressive=False):
        rng = random.Random(random_state)
        if uncovered_mask is None:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            num_uncovered = num_triples
        else:
            uncovered = uncovered_mask.copy()
            num_uncovered = int(np.sum(uncovered))
            
        blocks = []
        points = list(range(v))
        
        while num_uncovered > 0:
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break
                    
                rng.shuffle(candidates)
                
                best_gain = -1
                best_point = -1
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        if gain == len(block_pairs):
                            break
                            
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                for (p1, p2) in block_pairs:
                    idx = get_triple_idx(best_point, p1, p2)
                    if uncovered[idx]:
                        uncovered[idx] = 0
                        num_uncovered -= 1
                        
            blocks.append(tuple(sorted(block)))
            
        return blocks, uncovered, num_uncovered
    
    def trim_blocks(blocks):
        if not blocks:
            return blocks
            
        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        
        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)
            
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
    
    def local_search_trim(blocks, max_iters=50):
        """Try to swap blocks to reduce count while maintaining coverage."""
        if not blocks:
            return blocks
            
        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)
            
        active = [True] * len(blocks)
        best_len = len(blocks)
        
        for _ in range(max_iters):
            improved = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                # Try removing block i
                temp_counts = counts.copy()
                for idx in block_triples[i]:
                    temp_counts[idx] -= 1
                if np.any(temp_counts < 0):
                    continue
                # Check if all triples still covered
                if np.all(temp_counts >= 0):
                    # Try to find a replacement block that covers the now-uncovered triples
                    uncovered_after = np.where(temp_counts == 0)[0]
                    if len(uncovered_after) == 0:
                        # Can remove without replacement
                        active[i] = False
                        counts = temp_counts
                        best_len -= 1
                        improved = True
                        break
                    else:
                        # Look for a block that covers all uncovered_after triples
                        for j in range(len(blocks)):
                            if not active[j] or j == i:
                                continue
                            covers_all = True
                            for uidx in uncovered_after:
                                if uidx not in block_triples[j]:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Swap: remove i, keep j (j was already active)
                                active[i] = False
                                counts = temp_counts
                                best_len -= 1
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]
    
    def is_valid(blocks):
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                covered[idx] = 1
        return np.all(covered == 1)
    
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 10:
        # Phase 1: Fresh greedy
        blocks, uncovered, num_uncovered = greedy_cover(random_state=seed_counter)
        trimmed = trim_blocks(blocks)
        
        if len(trimmed) < best_len:
            best_len = len(trimmed)
            best_blocks = list(trimmed)
        
        if best_len <= 69:
            break
            
        # Phase 2: Perturbation and repair
        if time.time() < deadline - 5:
            rng_pert = random.Random(seed_counter + 1000)
            
            # Multiple perturbation strategies
            strategies = []
            
            # Strategy 1: Remove 1-3 random blocks
            if len(trimmed) > 5:
                num_kicks = rng_pert.randint(1, min(3, len(trimmed) - 1))
                kick_indices = rng_pert.sample(range(len(trimmed)), num_kicks)
                remaining = [b for i, b in enumerate(trimmed) if i not in kick_indices]
                strategies.append(remaining)
            
            # Strategy 2: Remove blocks with lowest "essentiality" (most redundant)
            if len(trimmed) > 5:
                counts = np.zeros(num_triples, dtype=np.int32)
                for block in trimmed:
                    for triple in itertools.combinations(block, t):
                        counts[triple_to_idx[triple]] += 1
                # Score blocks by how many triples they cover that are covered by only 1 block
                block_scores = []
                for i, block in enumerate(trimmed):
                    score = 0
                    for triple in itertools.combinations(block, t):
                        if counts[triple_to_idx[triple]] == 1:
                            score += 1
                    block_scores.append(score)
                # Remove blocks with lowest scores (most redundant)
                sorted_idx = sorted(range(len(trimmed)), key=lambda x: block_scores[x])
                num_remove = min(2, len(trimmed) - 1)
                remove_idx = set(sorted_idx[:num_remove])
                remaining = [b for i, b in enumerate(trimmed) if i not in remove_idx]
                strategies.append(remaining)
            
            # Strategy 3: Swap a block with a random alternative
            if len(trimmed) > 3:
                swap_idx = rng_pert.randint(0, len(trimmed) - 1)
                # Generate a random block that shares some points
                orig_block = list(trimmed[swap_idx])
                new_block = list(rng_pert.sample(range(v), k))
                remaining = trimmed.copy()
                remaining[swap_idx] = tuple(sorted(new_block))
                strategies.append(remaining)
            
            for remaining_blocks in strategies:
                if time.time() >= deadline - 5:
                    break
                    
                covered_mask = np.zeros(num_triples, dtype=np.uint8)
                for block in remaining_blocks:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        covered_mask[idx] = 1
                
                uncovered_mask = 1 - covered_mask
                num_uncovered = int(np.sum(uncovered_mask))
                
                if num_uncovered > 0:
                    repaired_blocks, _, _ = greedy_cover(uncovered_mask=uncovered_mask, random_state=seed_counter + 200)
                    combined = remaining_blocks + repaired_blocks
                else:
                    combined = remaining_blocks
                    
                trimmed_combined = trim_blocks(combined)
                trimmed_combined = local_search_trim(trimmed_combined, max_iters=20)
                
                if len(trimmed_combined) < best_len:
                    best_len = len(trimmed_combined)
                    best_blocks = list(trimmed_combined)
                    
        seed_counter += 1
    
    # Final polish: local search on best solution
    if best_blocks and time.time() < deadline - 2:
        best_blocks = local_search_trim(best_blocks, max_iters=100)
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
