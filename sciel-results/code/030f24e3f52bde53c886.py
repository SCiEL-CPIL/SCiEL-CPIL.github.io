"""Covering design C(19,8,5) - optimized multi-start greedy with vectorized gains."""
import numpy as np
import itertools
import time
from math import comb


def _combinatorial_index(comb_tuple, v, t):
    """Compute the index of a combination in lexicographic order."""
    # Use the "combinadic" / colex index
    idx = 0
    for i, x in enumerate(comb_tuple):
        # Number of combinations with first element < x
        # Actually, use standard combinatorial number system
        pass
    # Simpler: precompute or use a mapping
    # For small v, we can just use a dict
    raise NotImplementedError


def construct(v, k, t):
    """
    Multi-start randomized greedy covering design constructor.
    Uses vectorized gain computation via precomputed t-subset membership.
    """
    start_time = time.time()
    time_limit = 110  # leave 10s buffer
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Create a mapping from t-subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # We'll track coverage as a boolean array
    def compute_block_coverage(block):
        """Return set of t-subset indices covered by block."""
        covered = set()
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[tuple(sub)]
            covered.add(idx)
        return covered
    
    def greedy_construct(seed_offset=0):
        """Run one greedy construction pass."""
        uncovered = np.ones(n_t_subs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Pick the first uncovered t-subset as seed for new block
            # Randomize by rotating the search
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Choose a seed subset - use the one that appears in fewest potential blocks
            # For speed, just pick the first uncovered one with some randomization
            seed_idx = uncovered_indices[(seed_offset + len(blocks)) % len(uncovered_indices)]
            seed_sub = all_t_subs[seed_idx]
            block = list(seed_sub)
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Compute gain: number of uncovered t-subsets that would be covered
                    # by adding x to block
                    # A t-subset S is covered by block+x if S ⊆ block ∪ {x}
                    # and S contains x (since it wasn't covered by block alone)
                    # So we need: S ⊆ (block ∪ {x}) and x ∈ S
                    # i.e., S\{x} ⊆ block and S is uncovered
                    
                    gain = 0
                    # For each (t-1)-subset of block, check if adding x makes a covered t-subset
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in sub_to_idx:
                            idx = sub_to_idx[candidate]
                            if uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1 or best_gain == 0:
                    # No improvement possible, pick random remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = remaining[(seed_offset + len(blocks) + len(block)) % len(remaining)]
                    else:
                        break
                
                block.append(best_x)
            
            block.sort()
            
            # Check if this block is new
            block_tuple = tuple(block)
            if block_tuple not in [tuple(b) for b in blocks]:
                blocks.append(block)
                # Update uncovered
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx[tuple(sub)]
                    uncovered[idx] = False
            
            # Time check
            if time.time() - start_time > time_limit:
                break
        
        return blocks
    
    # Try multiple random restarts with different parameters
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Different randomization strategies
        seed_offset = r * 7 + 1
        
        blocks = greedy_construct(seed_offset)
        
        # Validate
        if len(blocks) < best_len:
            # Verify coverage
            covered = set()
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    covered.add(tuple(sub))
            if len(covered) == n_t_subs:
                best_blocks = blocks
                best_len = len(blocks)
    
    if best_blocks is None:
        # Fallback: run one more greedy if we ran out of time
        blocks = greedy_construct(0)
        best_blocks = blocks
    
    return best_blocks
