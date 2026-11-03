"""Dual-guided heuristic for covering design C(v,k,t).

Formulates the problem as a minimum-weight vertex cover on a bipartite graph
between t-subsets and k-blocks, then uses a primal-dual style heuristic:
- Maintain dual variables (weights) for uncovered t-subsets.
- Score each candidate block by the sum of dual weights of t-subsets it covers.
- Pick the highest-scoring block, add it, and update duals.
- Use randomization and restarts to escape local optima.
"""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using a dual-guided heuristic.
    
    Args:
        v: number of points
        k: block size
        t: subset size to cover
    
    Returns:
        list of blocks, each a sorted list of k distinct integers in 0..v-1
    """
    # Precompute all t-subsets and k-blocks
    all_t_subs = list(itertools.combinations(range(v), t))
    all_k_blocks = list(itertools.combinations(range(v), k))
    
    n_t = len(all_t_subs)
    n_k = len(all_k_blocks)
    
    # Map t-subsets to indices
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Build coverage matrix: for each k-block, which t-subsets it covers
    # For efficiency, we'll work with bitmasks or sparse representation
    # Since n_t can be large (C(19,5)=11628), use a list of lists
    
    # For each block, compute the set of t-subset indices it covers
    block_covers = []
    for block in all_k_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_sub_to_idx[ts])
        block_covers.append(covered)
    
    # Precompute for each t-subset, which blocks cover it (reverse index)
    t_to_blocks = [set() for _ in range(n_t)]
    for bi, covered in enumerate(block_covers):
        for ti in covered:
            t_to_blocks[ti].add(bi)
    
    # Dual-guided heuristic with randomization
    best_blocks = None
    best_count = float('inf')
    
    # Time budget: leave some margin under 120s
    start_time = time.time()
    time_budget = 100.0  # seconds
    
    # Multiple passes with different random seeds and strategies
    rng = random.Random(42)
    
    for attempt in range(50):  # up to 50 attempts, but break if time is up
        if time.time() - start_time > time_budget:
            break
        
        # Initialize dual variables: start with uniform 1.0
        dual = np.ones(n_t, dtype=np.float64)
        
        # Uncovered t-subsets
        uncovered = set(range(n_t))
        selected_blocks = []
        
        # Greedy selection with dual weights
        while uncovered:
            if time.time() - start_time > time_budget:
                break
            
            # Compute score for each candidate block
            # Score = sum of dual weights of uncovered t-subsets it covers
            # Only consider blocks that cover at least one uncovered t-subset
            
            best_block = -1
            best_score = -1
            
            # To speed up, only consider blocks that cover at least one uncovered subset
            # For each uncovered t-subset, look at blocks covering it
            candidate_blocks = set()
            # Sample a subset of uncovered for efficiency if too many
            if len(uncovered) > 500:
                sample = rng.sample(list(uncovered), 500)
            else:
                sample = list(uncovered)
            
            for ti in sample:
                for bi in t_to_blocks[ti]:
                    candidate_blocks.add(bi)
            
            if not candidate_blocks:
                # Fallback: pick any block
                bi = rng.choice(range(n_k))
                best_block = bi
            else:
                for bi in candidate_blocks:
                    # Compute score
                    score = 0.0
                    for ti in block_covers[bi]:
                        if ti in uncovered:
                            score += dual[ti]
                    if score > best_score:
                        best_score = score
                        best_block = bi
            
            if best_block == -1:
                # Should not happen, but safety
                best_block = rng.choice(range(n_k))
            
            # Select the block
            selected_blocks.append(all_k_blocks[best_block])
            
            # Update uncovered
            for ti in block_covers[best_block]:
                if ti in uncovered:
                    uncovered.discard(ti)
            
            # Update dual variables: increase dual for uncovered t-subsets
            # Primal-dual: increase duals of uncovered constraints
            for ti in uncovered:
                dual[ti] *= 1.1  # multiplicative increase
            # Decay duals for covered ones slightly
            # Actually, keep duals only for uncovered; reset covered to 0 conceptually
        
        if not uncovered and len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = list(selected_blocks)
        
        # Early stop if we hit a good number
        if best_count <= 300:
            break
    
    # If no valid solution found within time, fall back to simple greedy
    if best_blocks is None:
        # Simple greedy fallback
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            # Pick block covering most uncovered
            best_block = -1
            best_cover = -1
            for bi in range(n_k):
                cover_count = sum(1 for ti in block_covers[bi] if ti in uncovered)
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_block = bi
            if best_block == -1:
                break
            blocks.append(all_k_blocks[best_block])
            for ti in block_covers[best_block]:
                uncovered.discard(ti)
        best_blocks = blocks
    
    # Convert to list of lists (sorted)
    result = [sorted(b) for b in best_blocks]
    
    # Remove duplicates (shouldn't happen but safety)
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
    
    return unique_result
