"""Modular arithmetic block construction for covering design C(19,7,4)."""
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using affine transformations of base blocks
    with good difference properties, followed by greedy completion.
    """
    if v < k or t > k:
        return []
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    
    # Map t-subset to index for fast lookup
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # uncovered is a boolean array, True if not covered
    uncovered = np.ones(total_tsubsets, dtype=bool)
    blocks = []
    used_blocks = set()
    
    # --- Phase 1: Generate structured blocks via Affine Transformations ---
    # We generate a pool of candidate blocks by applying affine transformations
    # x -> a*x + b (mod v) to a set of base blocks.
    # Since v=19 is prime, a can be any non-zero element.
    
    # Step 1: Select good base blocks.
    # A base block is "good" if its difference multiset is as uniform as possible.
    # For v=19, k=7, ideal difference count for each non-zero element is (7*6)/(19-1) = 42/18 = 2.33.
    # So we want differences to appear 2 or 3 times.
    
    def score_base_block(base):
        """Score a base block based on difference uniformity."""
        diffs = [0] * v
        b_set = set(base)
        for x in b_set:
            for y in b_set:
                if x != y:
                    diffs[(x - y) % v] += 1
        # Exclude 0
        non_zero_diffs = [d for i, d in enumerate(diffs) if i != 0]
        if not non_zero_diffs:
            return 0
        mean = np.mean(non_zero_diffs)
        variance = np.var(non_zero_diffs)
        return -variance # Lower variance is better
    
    # Generate candidate base blocks
    base_candidates = []
    
    # 1. Consecutive
    base_candidates.append(list(range(k)))
    
    # 2. Arithmetic Progressions (multiples of step)
    for step in range(1, v):
        if np.gcd(step, v) == 1:
            ap = sorted((i * step) % v for i in range(k))
            # Normalize: sort and rotate so min is 0? 
            # Affine transformations will handle normalization, but for uniqueness in base selection:
            # We just add it. Duplicates in base list don't hurt, just waste time.
            base_candidates.append(ap)
            
    # 3. Random-ish structured (Quadratic Residues subset, etc.)
    # Let's add a few random ones seeded for reproducibility to find better bases
    np.random.seed(42)
    for _ in range(50):
        # Pick 7 random distinct points
        rand_block = list(np.random.choice(v, k, replace=False))
        base_candidates.append(rand_block)
        
    # Evaluate and select top 5 base blocks
    scored_bases = []
    seen_bases = set()
    for b in base_candidates:
        key = tuple(sorted(b))
        if key in seen_bases:
            continue
        seen_bases.add(key)
        s = score_base_block(b)
        scored_bases.append((s, b))
        
    scored_bases.sort(key=lambda x: x[0], reverse=True)
    top_bases = [b for s, b in scored_bases[:5]]
    
    # Step 2: Generate Affine Orbits for these bases
    # For each base block B, generate { (a*x + b) % v for x in B } for all a in [1, v-1], b in [0, v-1]
    affine_pool = []
    seen_affine = set()
    
    for base in top_bases:
        base_set = set(base)
        for a in range(1, v):
            for b in range(v):
                transformed = tuple(sorted((a*x + b) % v for x in base_set))
                if transformed not in seen_affine:
                    seen_affine.add(transformed)
                    affine_pool.append(transformed)
                    
    # Precompute the t-subsets covered by each block in the pool
    # block_tsubs[i] = list of indices of t-subsets covered by affine_pool[i]
    block_tsubs = []
    for block in affine_pool:
        tsubs = []
        for c in itertools.combinations(block, t):
            idx = tsub_idx[c]
            tsubs.append(idx)
        block_tsubs.append(tsubs)
        
    # --- Phase 2: Greedy Selection from Affine Pool ---
    # We want to pick blocks from affine_pool that cover the most currently uncovered t-subsets.
    # To speed this up, we can use a simple greedy loop.
    # Given the size of the pool (max 5 bases * 18 * 19 = 1710 blocks) and total t-subsets (3876),
    # a standard greedy is feasible.
    
    # Optimization: Pre-compute coverage counts? No, uncovered changes.
    # We will iterate.
    
    max_iterations = len(affine_pool)
    for _ in range(max_iterations):
        best_block_idx = -1
        best_gain = 0
        
        for i, tsubs in enumerate(block_tsubs):
            if not tsubs:
                continue
            # Count how many of these are currently uncovered
            # Using numpy for speed if possible, but simple loop is okay for 3876 items
            # Let's use a sum with generator
            gain = sum(1 for idx in tsubs if uncovered[idx])
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1 or best_gain == 0:
            break
            
        # Add the block
        block = list(affine_pool[best_block_idx])
        blocks.append(block)
        
        # Update uncovered
        for idx in block_tsubs[best_block_idx]:
            uncovered[idx] = False
            
        # Remove from pool to avoid re-considering (or just mark as used)
        # Since we iterate all each time, we should mark as used to skip
        # But simply breaking when gain is 0 is sufficient if we don't remove.
        # However, re-evaluating used blocks is wasteful.
        # Let's set block_tsubs[best_block_idx] = [] to effectively remove it.
        block_tsubs[best_block_idx] = []
        
        # Early stopping if all covered
        if not uncovered.any():
            break
            
    # --- Phase 3: Greedy Completion for Remaining Uncovered ---
    # If there are still uncovered t-subsets, we need to construct new blocks.
    # This part is heuristic but must be fast.
    
    while uncovered.any():
        # Find an uncovered t-subset
        # We need to pick one that is "hard" or just the first one.
        # Picking the first one is fine.
        first_uncovered_idx = np.argmax(uncovered)
        first_uncovered = all_tsubsets[first_uncovered_idx]
        
        # Start with this t-subset
        current_block = list(first_uncovered)
        current_block_set = set(current_block)
        
        # Greedily add points to maximize coverage of other uncovered t-subsets
        while len(current_block) < k:
            best_x = None
            best_gain = 0
            
            for x in range(v):
                if x in current_block_set:
                    continue
                
                # Calculate gain: how many new uncovered t-subsets are covered
                # A t-subset is covered if it is a subset of current_block + {x}
                # Specifically, we are adding x to current_block.
                # The new t-subsets covered are those that contain x and t-1 elements from current_block.
                gain = 0
                # Iterate over all (t-1)-subsets of current_block
                for rest in itertools.combinations(current_block, t - 1):
                    # Create the t-subset
                    new_tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_idx[new_tsub]
                    if uncovered[idx]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x is None:
                # No gain? Pick arbitrary
                for x in range(v):
                    if x not in current_block_set:
                        best_x = x
                        break
                if best_x is None:
                    break
            else:
                current_block.append(best_x)
                current_block_set.add(best_x)
                
        # If we didn't reach k, fill arbitrarily
        if len(current_block) < k:
            for x in range(v):
                if len(current_block) >= k:
                    break
                if x not in current_block_set:
                    current_block.append(x)
                    current_block_set.add(x)
                    
        current_block.sort()
        blocks.append(current_block)
        
        # Update uncovered
        for c in itertools.combinations(current_block, t):
            idx = tsub_idx[c]
            uncovered[idx] = False
            
    return blocks
