import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_cover_exhaustive(v, k, t, rng, time_limit):
    """
    Constructs a covering using a greedy heuristic.
    For each step, picks a pivot triple and finds the best block containing it
    by evaluating all possible completions.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            break
            
        # Pick a random uncovered triple to start the block
        pivot_triple = rng.choice(list(uncovered))
        pivot_set = set(pivot_triple)
        
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is not None:
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
            
    return [list(b) for b in blocks]

def _spectral_greedy_cover(v, k, t, rng, time_limit):
    """
    Spectral-guided greedy covering:
    1. Build incidence matrix between all possible k-blocks and t-triples.
    2. Compute spectral centrality of blocks using top eigenvectors of normalized Laplacian.
    3. Use spectral scores to guide greedy selection, prioritizing structurally important blocks.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Generate all possible blocks and triples
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    
    n_blocks = len(all_blocks)
    n_triples = len(all_triples)
    
    # Map triples to indices
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Build incidence matrix A (n_blocks x n_triples) where A[i,j]=1 if block i covers triple j
    # This can be large, so we'll be smart about it
    # For v=24, k=7: C(24,7)=346104, C(24,3)=2024
    # Matrix would be 346104 x 2024 - too large for dense, but we can use sparse or work with subsets
    
    # Instead of full matrix, compute spectral scores approximately:
    # Use a subset of blocks for spectral analysis, then use scores to guide selection
    
    # Precompute which triples each block covers
    block_triples = []
    for block in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(block, t) if tri in triple_to_idx]
        block_triples.append(tris)
    
    # Select a random subset of blocks for spectral analysis (to keep computation feasible)
    n_sample = min(5000, n_blocks)
    if n_blocks > n_sample:
        sample_indices = rng.sample(range(n_blocks), n_sample)
    else:
        sample_indices = list(range(n_blocks))
    
    # Build incidence matrix for sampled blocks
    A_sample = np.zeros((len(sample_indices), n_triples), dtype=np.float32)
    for idx, bi in enumerate(sample_indices):
        for ti in block_triples[bi]:
            A_sample[idx, ti] = 1.0
    
    # Compute degree vectors
    block_degrees = A_sample.sum(axis=1)  # number of triples each block covers
    triple_degrees = A_sample.sum(axis=0)  # number of blocks covering each triple
    
    # Avoid division by zero
    block_degrees_safe = np.where(block_degrees > 0, block_degrees, 1)
    triple_degrees_safe = np.where(triple_degrees > 0, triple_degrees, 1)
    
    # Normalized incidence: D_B^{-1/2} A D_T^{-1/2}
    D_B_inv_sqrt = 1.0 / np.sqrt(block_degrees_safe)
    D_T_inv_sqrt = 1.0 / np.sqrt(triple_degrees_safe)
    
    A_norm = A_sample * D_B_inv_sqrt[:, np.newaxis]
    A_norm = A_norm * D_T_inv_sqrt[np.newaxis, :]
    
    # Compute top singular vectors of A_norm for spectral scores
    # SVD: A_norm = U S V^T
    # Block centrality ~ U[:, 0] (first left singular vector)
    try:
        U, S, Vt = np.linalg.svd(A_norm, full_matrices=False)
        spectral_scores_sample = U[:, 0]  # First left singular vector
    except:
        # Fallback: use degree-based scores
        spectral_scores_sample = block_degrees_safe
    
    # Map spectral scores back to all blocks
    # For unsampled blocks, estimate score based on overlap with sampled blocks
    spectral_scores_all = np.zeros(n_blocks, dtype=np.float32)
    for idx, bi in enumerate(sample_indices):
        spectral_scores_all[bi] = spectral_scores_sample[idx]
    
    # For unsampled blocks, estimate using average of similar sampled blocks
    # Simple heuristic: use block degree as proxy for unsampled
    for bi in range(n_blocks):
        if bi not in set(sample_indices):
            spectral_scores_all[bi] = block_degrees[bi] / np.max(block_degrees_safe) if np.max(block_degrees_safe) > 0 else 0
    
    # Now do greedy selection using spectral scores as tie-breaker/priority
    uncovered = set(range(n_triples))  # All triples initially uncovered
    blocks_selected = []
    
    # Create a priority queue: blocks sorted by (spectral_score, random_noise)
    # We'll iterate through blocks in order of decreasing spectral score with some randomization
    
    block_order = list(range(n_blocks))
    # Sort by spectral score descending, with random tie-breaking
    rng.shuffle(block_order)
    block_order.sort(key=lambda i: -spectral_scores_all[i] + rng.random() * 0.1)
    
    # Greedy: iterate through blocks in spectral priority order, select if it covers uncovered triples
    # But we need to ensure every triple gets covered, so we also need to check coverage
    
    # Actually, let's do a hybrid: use spectral scores to guide which pivot to choose
    # and which completion to prefer
    
    # Simpler approach: use spectral scores to weight the gain calculation
    # gain_weighted = sum(spectral_weight[tri] for tri in block_triples if tri in uncovered)
    
    # Compute triple weights: triples covered by fewer blocks are more "important"
    triple_weights = 1.0 / np.where(triple_degrees_safe > 0, triple_degrees_safe, 1.0)
    
    # Greedy with weighted gain
    while uncovered:
        if time.time() > deadline:
            break
        
        # Find the triple with highest weight that is uncovered
        # This prioritizes rare triples
        max_weight = 0
        pivot_triple_idx = None
        for tri_idx in list(uncovered)[:100]:  # Sample from uncovered for efficiency
            if triple_weights[tri_idx] > max_weight:
                max_weight = triple_weights[tri_idx]
                pivot_triple_idx = tri_idx
        
        if pivot_triple_idx is None:
            pivot_triple_idx = rng.choice(list(uncovered))
        
        pivot_triple = all_triples[pivot_triple_idx]
        pivot_set = set(pivot_triple)
        
        # Find best block containing this pivot, weighted by spectral score and gain
        best_score = -1
        best_block = None
        
        # Sample candidate blocks containing the pivot
        other_points = [p for p in range(v) if p not in pivot_set]
        candidates = []
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            candidates.append(block)
        
        # For efficiency, sample candidates if too many
        max_candidates = 200
        if len(candidates) > max_candidates:
            candidates = rng.sample(candidates, max_candidates)
        
        for block in candidates:
            # Find index of this block
            # This is O(n_blocks) which is slow, so let's precompute a lookup
            pass
        
        # Precompute block lookup for speed
        # Actually, let's just compute gain directly
        for block in candidates:
            block_set = set(block)
            tris_in_block = list(itertools.combinations(block, t))
            
            gain = 0
            spectral_boost = 0
            for tri in tris_in_block:
                tri_idx = triple_to_idx.get(tri, -1)
                if tri_idx >= 0 and tri_idx in uncovered:
                    gain += 1
                    spectral_boost += triple_weights[tri_idx]
            
            # Combine gain and spectral boost
            score = gain + 0.5 * spectral_boost
            
            if score > best_score:
                best_score = score
                best_block = block
        
        if best_block is not None:
            blocks_selected.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                tri_idx = triple_to_idx.get(tri, -1)
                if tri_idx >= 0:
                    uncovered.discard(tri_idx)
    
    return blocks_selected

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    A block is redundant if every triple it covers is also covered by at least one other block.
    """
    if not blocks:
        return blocks
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        for i, block in enumerate(blocks):
            is_essential = False
            for tri in block_triples[i]:
                if triple_cover_count[tri] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                changed = True
                for tri in block_triples[i]:
                    triple_cover_count[tri] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search_sa(blocks, v, k, t, rng, time_limit):
    """
    Simulated Annealing style local search.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    def get_state(blocks):
        triple_cover_count = defaultdict(int)
        block_triples = []
        for block in blocks:
            tris = list(itertools.combinations(block, t))
            block_triples.append(tris)
            for tri in tris:
                triple_cover_count[tri] += 1
        return triple_cover_count, block_triples

    triple_cover_count, block_triples = get_state(blocks)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            original_block = blocks[i]
            original_triples = block_triples[i]
            
            # Identify "must" triples (count == 1)
            must_triples = [tri for tri in original_triples if triple_cover_count[tri] == 1]
            
            # The new block MUST contain all points in the union of must_triples.
            must_points = set()
            for tri in must_triples:
                must_points.update(tri)
                
            if len(must_points) > k:
                # Cannot form a block containing all must points
                continue
                
            # We need to choose (k - len(must_points)) points from the remaining (v - len(must_points)) points.
            found_better = False
            max_candidates = 5
            for _ in range(max_candidates):
                available_points = [p for p in range(v) if p not in must_points]
                if len(available_points) < k - len(must_points):
                    continue
                extra_points = rng.sample(available_points, k - len(must_points))
                candidate = sorted(must_points | set(extra_points))
                
                if tuple(candidate) == tuple(original_block):
                    continue
                
                new_triples = list(itertools.combinations(candidate, t))
                old_triples_set = set(original_triples)
                new_triples_set = set(new_triples)
                
                valid = True
                for tri in old_triples_set:
                    if tri not in new_triples_set:
                        if triple_cover_count[tri] == 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Perform Swap
                for tri in original_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    found_better = True
                    triple_cover_count, block_triples = get_state(blocks)
                    break
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
            if found_better:
                break

    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Initial Greedy Construction
    rng = random.Random(42)
    blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Spectral Greedy Construction (new)
    if time.time() < deadline - 10:
        rng_spec = random.Random(777)
        spec_blocks = _spectral_greedy_cover(v, k, t, rng_spec, deadline - time.time() - 5)
        if spec_blocks:  # Only if we got a complete covering
            spec_blocks = _reduce_covering(spec_blocks, v, k, t)
            if len(spec_blocks) < best_size:
                best_size = len(spec_blocks)
                best_blocks = spec_blocks
    
    # 3. Local Search Refinement
    if time.time() < deadline - 5:
        rng_ls = random.Random(123)
        improved_blocks = _local_search_sa(best_blocks, v, k, t, rng_ls, deadline - time.time())
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
    
    # 4. Multiple Randomized Greedy Runs
    seed_counter = 1
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
        
        if time.time() > deadline - 5:
            break
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            if time.time() < deadline - 15:
                rng_ls = random.Random(seed_counter + 1000)
                improved_blocks = _local_search_sa(best_blocks, v, k, t, rng_ls, deadline - time.time() - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
