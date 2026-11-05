import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks and their covered triplet IDs
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_triplets = []
    for block in all_blocks:
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        block_triplets.append(tids)
    
    # Map each triplet to the list of block indices that cover it
    triple_blocks = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            triple_blocks[tid].append(b_idx)

    def greedy_construct(seed=0, bias=0.0):
        """Greedy set cover with optional randomization bias."""
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=np.bool_)
        gain = np.array([len(ts) for ts in block_triplets], dtype=np.int32)
        used_blocks = set()
        selected = []
        
        while is_uncovered.any():
            if time.time() > deadline - 30:
                break
            
            # Find max gain among unused blocks
            max_gain = -1
            best_idx = -1
            candidates = []
            
            for b_idx in range(n_blocks):
                if b_idx in used_blocks:
                    continue
                s = gain[b_idx]
                if s > max_gain:
                    max_gain = s
                    best_idx = b_idx
                    candidates = [b_idx]
                elif s == max_gain:
                    candidates.append(b_idx)
            
            if best_idx == -1 or max_gain == 0:
                break
            
            # Pick from candidates with optional randomization
            if bias > 0 and len(candidates) > 1:
                # Weighted random selection favoring higher gain
                weights = [gain[c] + bias * rng.random() for c in candidates]
                total = sum(weights)
                r = rng.random() * total
                cumulative = 0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
                b_idx = chosen
            else:
                b_idx = rng.choice(candidates)
            
            selected.append(b_idx)
            used_blocks.add(b_idx)
            
            # Update gain and uncovered status
            for tid in block_triplets[b_idx]:
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    for other_b in triple_blocks[tid]:
                        gain[other_b] -= 1
        
        return selected

    def reduce_cover(selected_indices):
        """Remove redundant blocks from a cover."""
        cover_count = np.zeros(n_triplets, dtype=np.int32)
        for b_idx in selected_indices:
            for tid in block_triplets[b_idx]:
                cover_count[tid] += 1
        
        active = [True] * len(selected_indices)
        
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_indices):
                if not active[i]:
                    continue
                redundant = True
                for tid in block_triplets[b_idx]:
                    if cover_count[tid] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for tid in block_triplets[b_idx]:
                        cover_count[tid] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_indices) if active[i]]

    def try_swap_2_for_1(blocks_indices, cover_count):
        """Try to replace two blocks with one that covers all critical triplets."""
        n = len(blocks_indices)
        if n < 2:
            return blocks_indices, cover_count, False
        
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks_indices, cover_count, False
            b1 = blocks_indices[i]
            ids_b1 = set(block_triplets[b1])
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks_indices, cover_count, False
                b2 = blocks_indices[j]
                ids_b2 = set(block_triplets[b2])
                
                critical_points = set()
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = cover_count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                
                # Check b2's triplets (only those not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = cover_count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    for tid in block_triplets[b1]:
                        cover_count[tid] -= 1
                    for tid in block_triplets[b2]:
                        cover_count[tid] -= 1
                    new_blocks = [b for idx, b in enumerate(blocks_indices) if idx != i and idx != j]
                    return new_blocks, cover_count, True
                
                if len(critical_points) <= k:
                    # Find a block that contains all critical points
                    cp_set = set(critical_points)
                    new_block_idx = -1
                    for b_idx in range(n_blocks):
                        if all(p in set(all_blocks[b_idx]) for p in cp_set):
                            new_block_idx = b_idx
                            break
                    
                    if new_block_idx != -1:
                        # Verify validity: all previously covered triplets remain covered
                        valid = True
                        for tid in ids_b1 | ids_b2:
                            if cover_count[tid] == 1 and tid not in set(block_triplets[new_block_idx]):
                                valid = False
                                break
                        
                        if valid:
                            for tid in block_triplets[b1]:
                                cover_count[tid] -= 1
                            for tid in block_triplets[b2]:
                                cover_count[tid] -= 1
                            for tid in block_triplets[new_block_idx]:
                                cover_count[tid] += 1
                            new_blocks = [b for idx, b in enumerate(blocks_indices) if idx != i and idx != j]
                            new_blocks.append(new_block_idx)
                            return new_blocks, cover_count, True
        
        return blocks_indices, cover_count, False

    # Phase 1: Initial constructions with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # Pure greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        selected = greedy_construct(seed, bias=0.0)
        reduced = reduce_cover(selected)
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_blocks = reduced
    
    # Randomized greedy
    for seed in range(15):
        if time.time() > deadline - 30:
            break
        bias = 0.05 + 0.02 * (seed % 5)
        selected = greedy_construct(seed, bias=bias)
        reduced = reduce_cover(selected)
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_blocks = reduced

    if best_blocks is None:
        best_blocks = reduce_cover(greedy_construct(0, bias=0.0))

    # Phase 2: Local search optimization
    current_blocks = list(best_blocks)
    cover_count = np.zeros(n_triplets, dtype=np.int32)
    for b_idx in current_blocks:
        for tid in block_triplets[b_idx]:
            cover_count[tid] += 1
    
    progress = True
    while progress and time.time() < deadline - 2:
        progress = False
        
        # Remove redundants
        new_blocks = reduce_cover(current_blocks)
        if len(new_blocks) < len(current_blocks):
            # Rebuild cover_count
            cover_count = np.zeros(n_triplets, dtype=np.int32)
            for b_idx in new_blocks:
                for tid in block_triplets[b_idx]:
                    cover_count[tid] += 1
            current_blocks = new_blocks
            progress = True
        
        # Try 2-for-1 swaps
        if not progress:
            current_blocks, cover_count, swapped = try_swap_2_for_1(current_blocks, cover_count)
            if swapped:
                progress = True
    
    # Final reduction
    current_blocks = reduce_cover(current_blocks)
    
    return [list(all_blocks[b_idx]) for b_idx in current_blocks]

if __name__ == "__main__":
    pass
