"""Frequency-biased greedy set cover for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(v, k, t):
    # All candidate k-subsets of {0..v-1}
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute points for each t-subset for fast bias update
    tsub_points = [np.array(ts, dtype=np.int64) for ts in all_tsubs]

    # Create incidence matrix M: M[b, s] = 1 if block b covers t-subset s
    # Using bool to save memory
    M = np.zeros((n_blocks, n_tsubs), dtype=bool)
    
    # Precompute block_points for bias calculation (matrix P)
    # P[b, p] = 1 if point p is in block b
    P = np.zeros((n_blocks, v), dtype=bool)
    
    for b, blk in enumerate(all_blocks):
        # Set points in P
        for p in blk:
            P[b, p] = True
        # Set t-subsets in M
        for ts in itertools.combinations(blk, t):
            s_idx = tsub_to_idx[ts]
            M[b, s_idx] = True

    # uncovered: boolean array over t-subsets
    uncovered = np.ones(n_tsubs, dtype=bool)

    # point_tsub_count[p] = number of uncovered t-subsets containing point p
    # Initialize
    point_tsub_count = np.zeros(v, dtype=np.int64)
    for ts in all_tsubs:
        for p in ts:
            point_tsub_count[p] += 1

    blocks = []
    used = np.zeros(n_blocks, dtype=bool)
    
    # Convert uncovered to int8 for dot product
    uncovered_int = uncovered.astype(np.int8)

    while uncovered.any():
        # Score each unused block: number of uncovered t-subsets it covers
        # scores = M @ uncovered_int
        scores = M.dot(uncovered_int).astype(np.float64)

        # Bias: add a small weighted term based on point under-coverage
        # under_cover = 1 - (point_tsub_count / max_count)
        max_pt = point_tsub_count.max()
        if max_pt > 0:
            under_cover = 1.0 - (point_tsub_count / max_pt).astype(np.float64)
        else:
            under_cover = np.ones(v, dtype=np.float64)
            
        # bias_scores = P @ under_cover
        bias_scores = P.dot(under_cover)
        
        # Combine scores
        # We want to prioritize covering more t-subsets.
        # The bias is very small, just to break ties or guide towards under-covered points.
        # Scale bias so it's less than 1 (so it doesn't override the main score)
        bias_scale = 0.001
        total_scores = scores + bias_scale * bias_scores

        # Mask used blocks
        total_scores[used] = -1.0
        
        best_idx = int(np.argmax(total_scores))
        best_score = total_scores[best_idx]
        
        if best_score <= 0:
            break

        used[best_idx] = True
        blk = all_blocks[best_idx]
        blocks.append(list(blk))
        
        # Identify newly covered t-subsets
        # M[best_idx, :] gives all t-subsets covered by this block
        # We need those that were previously uncovered
        covered_by_block = M[best_idx, :]
        newly_covered_idx = np.where(covered_by_block & uncovered)[0]
        
        # Update uncovered
        uncovered[covered_by_block] = False
        uncovered_int[covered_by_block] = 0
        
        # Update point_tsub_count
        # For each newly covered t-subset, decrement count for its points
        for s_idx in newly_covered_idx:
            pts = tsub_points[s_idx]
            point_tsub_count[pts] -= 1

    return blocks
# EVOLVE-BLOCK-END
