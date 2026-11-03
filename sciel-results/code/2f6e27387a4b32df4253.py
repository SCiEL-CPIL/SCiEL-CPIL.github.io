"""Rewritten program for covering design C(19,8,5) using ILP-style exclusion with lazy constraints."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design using an ILP-inspired exclusion approach:
    1. Build a candidate pool of k-blocks via greedy heuristics with multiple seeds.
    2. Formulate as: minimize number of blocks kept, subject to every t-subset covered.
       Equivalent to: maximize excluded blocks, subject to coverage.
    3. Solve via custom branch-and-bound / cutting-plane heuristic:
       - Start with all blocks included.
       - Iteratively try to exclude blocks; if excluding a block leaves some t-subset
         uncovered, add a constraint that forces at least one of the blocks covering
         that t-subset to be kept.
    4. Use local search to improve.
    """
    all_points = list(range(v))
    
    # Precompute all t-subsets
    all_t_subsets = [frozenset(comb) for comb in itertools.combinations(all_points, t)]
    total_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each t-subset, precompute which points it contains (as a frozenset for fast ops)
    
    def generate_greedy_blocks(seed, max_blocks=None):
        """Greedy set cover: repeatedly pick block covering most uncovered t-subsets."""
        rng = random.Random(seed)
        uncovered = set(range(total_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset as anchor
            anchor_idx = rng.choice(list(uncovered))
            anchor = all_t_subsets[anchor_idx]
            
            # Generate candidate k-subsets containing anchor
            remaining_points = [p for p in all_points if p not in anchor]
            need = k - t
            
            best_block = None
            best_count = -1
            
            for extra in itertools.combinations(remaining_points, need):
                block = anchor | set(extra)
                # Count how many uncovered t-subsets this block covers
                count = 0
                for ts in itertools.combinations(sorted(block), t):
                    idx = t_subset_to_idx[frozenset(ts)]
                    if idx in uncovered:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
            
            block_frozen = frozenset(best_block)
            blocks.append(sorted(best_block))
            
            # Update uncovered
            for ts in itertools.combinations(sorted(best_block), t):
                uncovered.discard(t_subset_to_idx[frozenset(ts)])
            
            if max_blocks and len(blocks) >= max_blocks:
                break
        
        return blocks
    
    def build_candidate_pool(num_seeds=12, max_per_seed=350):
        """Build a diverse pool of candidate blocks."""
        pool = set()
        for seed in range(num_seeds):
            blocks = generate_greedy_blocks(seed, max_blocks=max_per_seed)
            for b in blocks:
                pool.add(frozenset(b))
        return list(pool)
    
    def solve_exclusion(candidate_blocks, time_limit=105.0):
        """
        ILP-style exclusion: start with all blocks kept, try to exclude as many as possible.
        Constraint: for each t-subset, at least one kept block must contain it.
        
        Approach:
        - For each t-subset, precompute which candidate blocks cover it.
        - Use a greedy deletion heuristic with backtracking-like local search.
        """
        start_time = time.time()
        n_blocks = len(candidate_blocks)
        
        # Precompute: for each t-subset, which blocks cover it
        t_subset_to_blocks = defaultdict(list)
        for i, block in enumerate(candidate_blocks):
            for ts in itertools.combinations(sorted(block), t):
                idx = t_subset_to_idx[frozenset(ts)]
                t_subset_to_blocks[idx].append(i)
        
        # Initial state: all blocks kept
        kept = set(range(n_blocks))
        
        # Try to exclude blocks in random order
        order = list(range(n_blocks))
        random.Random(42).shuffle(order)
        
        for block_idx in order:
            if time.time() - start_time > time_limit - 5:
                break
            
            if block_idx not in kept:
                continue
            
            # Try excluding this block
            kept.discard(block_idx)
            
            # Check if any t-subset is now uncovered
            uncovered_found = False
            for ts_idx, covering_blocks in t_subset_to_blocks.items():
                if not any(b in kept for b in covering_blocks):
                    # This t-subset is uncovered; we need to keep at least one
                    # Restore block_idx or find another
                    kept.add(block_idx)
                    uncovered_found = True
                    break
            
            if uncovered_found:
                # Could not exclude this block
                pass
        
        # Local search: try swapping - exclude one kept block, include one excluded
        excluded = set(range(n_blocks)) - kept
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for block_idx in list(kept):
                if time.time() - start_time > time_limit - 5:
                    break
                # Try excluding block_idx
                kept.discard(block_idx)
                
                # Find uncovered t-subsets
                uncovered_ts = []
                for ts_idx, covering_blocks in t_subset_to_blocks.items():
                    if not any(b in kept for b in covering_blocks):
                        uncovered_ts.append(ts_idx)
                
                if not uncovered_ts:
                    # Successfully excluded
                    improved = True
                    continue
                else:
                    # Need to add back some block to cover uncovered_ts
                    # Find an excluded block that covers the most uncovered t-subsets
                    best_add = None
                    best_cover = 0
                    for add_idx in excluded:
                        if add_idx == block_idx:
                            continue
                        cover_count = 0
                        for ts_idx in uncovered_ts:
                            if add_idx in t_subset_to_blocks[ts_idx]:
                                cover_count += 1
                        if cover_count > best_cover:
                            best_cover = cover_count
                            best_add = add_idx
                    
                    if best_add is not None and best_cover == len(uncovered_ts):
                        # Can swap: exclude block_idx, include best_add
                        kept.add(best_add)
                        excluded.discard(best_add)
                        excluded.add(block_idx)
                        improved = True
                    else:
                        # Cannot exclude block_idx
                        kept.add(block_idx)
        
        return list(kept)
    
    # Main Algorithm
    start_time = time.time()
    
    # Build candidate pool
    candidate_pool = build_candidate_pool(num_seeds=15, max_per_seed=400)
    
    # Solve ILP exclusion
    solution_indices = solve_exclusion(candidate_pool, time_limit=110.0)
    
    # Convert to actual blocks
    result_blocks = [list(candidate_pool[i]) for i in solution_indices]
    
    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in result_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    # Verify coverage
    if not unique_blocks:
        # Fallback to simple greedy
        unique_blocks = generate_greedy_blocks(42)
    
    # Final verification
    covered = set()
    for b in unique_blocks:
        for ts in itertools.combinations(sorted(b), t):
            covered.add(frozenset(ts))
    
    if len(covered) != total_t_subsets:
        # Coverage failed, fall back to greedy
        unique_blocks = generate_greedy_blocks(42)
    
    return unique_blocks
# EVOLVE-BLOCK-END
