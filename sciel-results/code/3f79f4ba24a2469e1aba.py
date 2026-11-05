"""Improved covering design C(24,7,3) via randomized greedy with time management."""
import itertools
import time
import random
import sys


def _greedy_construct(v, k, t, seed=0, time_limit=115.0):
    """
    Randomized greedy set cover for covering t-subsets with k-subsets.
    
    Strategy:
    1. Start with all C(v,t) t-subsets as uncovered.
    2. Repeatedly:
       a. Pick the uncovered t-subset with the fewest candidate blocks (most constrained).
       b. Among blocks containing that subset, pick one that covers the most uncovered t-subsets.
       c. Add that block, remove all t-subsets it covers.
    3. Use randomization to break ties and explore different orderings.
    
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Precompute all t-subsets as tuples for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to its index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets contain it
    # For each pair of points, precompute which t-subsets contain both (for k=7, t=3, 
    # a block covers C(7,3)=35 triples; we need to track which triples are covered)
    
    # We'll use a bitset-like approach with a set of uncovered triple indices
    uncovered = set(range(n_t))
    
    # For efficiency, precompute for each point x, the list of triple indices containing x
    triples_containing = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            triples_containing[x].append(i)
    
    # For each pair (x,y), precompute triple indices containing both
    # This helps when building a block incrementally
    pairs_to_triples = {}
    for i, ts in enumerate(all_t_subsets):
        for a in range(t):
            for b in range(a+1, t):
                key = (ts[a], ts[b])
                if key not in pairs_to_triples:
                    pairs_to_triples[key] = []
                pairs_to_triples[key].append(i)
    
    blocks = []
    
    def get_gain(block_so_far, candidate_x):
        """Calculate how many uncovered triples would be covered if we add x to block_so_far."""
        # A new triple is formed by x + (t-1) elements from block_so_far
        gain = 0
        for rest in itertools.combinations(block_so_far, t - 1):
            triple = tuple(sorted(rest + (candidate_x,)))
            idx = t_subset_to_idx.get(triple)
            if idx is not None and idx in uncovered:
                gain += 1
        return gain
    
    def complete_block(block_so_far):
        """Complete a block by greedily adding points that maximize uncovered triple coverage."""
        block = list(block_so_far)
        while len(block) < k:
            # Find the point not in block that maximizes gain
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = get_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        block.sort()
        return block
    
    def add_block(block):
        """Add a block and update uncovered set."""
        nonlocal uncovered
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if idx in uncovered:
                uncovered.discard(idx)
    
    # Main loop: repeatedly find the most constrained uncovered triple and cover it
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        # Find the uncovered triple with the fewest possible completing blocks
        # For efficiency, sample a few uncovered triples and pick the most constrained
        if len(uncovered) > 1000:
            # Sample for speed
            sample = rng.sample(list(uncovered), min(100, len(uncovered)))
        else:
            sample = list(uncovered)
        
        best_triple_idx = sample[0]
        best_constraint = float('inf')
        
        for idx in sample:
            ts = all_t_subsets[idx]
            # Count how many points could complete this into a block
            # A block containing ts needs (k-t) more points from the remaining (v-t) points
            # The constraint is how "rare" this triple is - we approximate by counting
            # available candidates
            # For simplicity, use the number of uncovered triples that share at least 2 points with ts
            # as a proxy for constraint
            shared_count = 0
            for x in ts:
                for tri_idx in triples_containing[x]:
                    if tri_idx != idx and tri_idx in uncovered:
                        # Check if they share at least one other point
                        tri = all_t_subsets[tri_idx]
                        if any(y in ts for y in tri if y != x):
                            shared_count += 1
                            break
            if shared_count < best_constraint:
                best_constraint = shared_count
                best_triple_idx = idx
        
        # Start a block with this triple
        seed_triple = all_t_subsets[best_triple_idx]
        
        # Try a few random completions and pick the best
        best_block = None
        best_coverage = -1
        
        n_attempts = 3 if len(uncovered) > 500 else 5
        
        for attempt in range(n_attempts):
            # Start with the seed triple, then add random points first, then optimize
            block = list(seed_triple)
            
            # Add some random points to diversify
            remaining_points = [x for x in range(v) if x not in block]
            n_random = min(2, len(remaining_points), k - t - 1)
            if n_random > 0:
                random_points = rng.sample(remaining_points, n_random)
                block.extend(random_points)
            
            # Complete greedily
            block = complete_block(block)
            
            # Count how many uncovered triples this block covers
            coverage = 0
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if idx in uncovered:
                    coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_block = block
        
        if best_block is None:
            break
            
        add_block(best_block)
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design: choose as few k-element blocks of v points as possible
    so that every t-element subset lies in some block.
    
    Uses randomized greedy with multiple restarts and time management.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Special case handling
    if k <= t:
        # Every k-subset is also a t-subset (if k==t) or we need to think differently
        if k == t:
            return [list(range(v))] if v == k else list(itertools.combinations(range(v), k))
        else:
            # k < t: impossible to cover t-subsets with k-subsets if k < t
            # But the problem guarantees valid parameters
            return []
    
    # Run multiple randomized restarts, keeping the best result
    best_blocks = None
    best_len = float('inf')
    
    # First run: deterministic-ish (seed=0)
    blocks = _greedy_construct(v, k, t, seed=0, time_limit=time_limit - (time.time() - start_time))
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks
    
    # Use remaining time for more restarts with different seeds
    while time.time() - start_time < time_limit - 5.0:
        seed = int(time.time() * 1000) % (2**31)
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10.0:
            break
        blocks = _greedy_construct(v, k, t, seed=seed, time_limit=remaining - 2.0)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Final verification and cleanup
    if best_blocks is None:
        # Fallback: should not happen, but provide a valid (possibly large) covering
        best_blocks = _greedy_construct(v, k, t, seed=42, time_limit=110.0)
    
    # Remove duplicate blocks and verify
    seen = set()
    unique_blocks = []
    for block in best_blocks:
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(key))
    
    return unique_blocks


if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    blocks = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Number of blocks: {len(blocks)}")
    print(f"Time: {elapsed:.2f}s")
    
    # Verify
    all_triples = set(itertools.combinations(range(24), 3))
    covered = set()
    for block in blocks:
        for triple in itertools.combinations(block, 3):
            covered.add(triple)
    missing = all_triples - covered
    print(f"Missing triples: {len(missing)}")
