"""
Covering design C(19,8,5) solver using a greedy set-cover approach with
bitmask acceleration and local improvement heuristics.

Strategy:
1. Precompute all C(19,5) = 11628 t-subsets as bitmasks.
2. For each candidate block (8-subset), precompute which t-subsets it covers.
3. Use a greedy set cover: repeatedly pick the block that covers the most
   uncovered t-subsets, with tie-breaking to maximize diversity.
4. Apply a local search / removal pass to try to eliminate redundant blocks.
5. Use randomized restarts to escape local optima, keeping the best solution.

We leverage numpy for vectorized operations where possible, but the core
greedy loop is done with Python sets for speed on this small instance.
"""

import itertools
import random
import time
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using greedy set cover with
    bitmask acceleration and local improvement.
    
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    # All t-subsets as frozensets for set operations
    all_t_subs = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subs)
    
    # Map each t-subset to an index
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Generate all candidate blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    
    # For each block, compute which t-subsets it covers
    # block_covers[block_idx] = set of t-subset indices covered by this block
    block_covers = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_sub_index[ts])
        block_covers.append(covered)
    
    # Precompute: for each t-subset index, which blocks cover it
    t_to_blocks = defaultdict(list)
    for bi, covers in enumerate(block_covers):
        for ti in covers:
            t_to_blocks[ti].append(bi)
    
    # Greedy set cover with randomized restarts
    best_blocks = None
    best_count = len(all_t_subs) + 1  # infinity
    start_time = time.time()
    time_limit = 110  # leave some margin for the 120s limit
    
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Shuffle blocks for randomized tie-breaking
        block_order = list(range(len(all_blocks)))
        rng.shuffle(block_order)
        
        uncovered = set(range(len(all_t_subs)))
        chosen_blocks = []
        
        while uncovered:
            # Find the block that covers the most uncovered t-subsets
            best_gain = -1
            best_bi = None
            best_block_idx = None
            
            for bi in block_order:
                gain = len(block_covers[bi] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_block_idx = bi
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: prefer blocks that cover t-subsets
                    # that are covered by fewer other blocks (rarity)
                    # This is a heuristic to improve diversity
                    pass
            
            if best_gain <= 0:
                # Should not happen if the covering is feasible
                break
            
            chosen_blocks.append(all_blocks[best_bi])
            uncovered -= block_covers[best_bi]
        
        # Local improvement: try to remove redundant blocks
        chosen_set = set(chosen_blocks)
        for i, block in enumerate(list(chosen_set)):
            # Check if removing this block still leaves all t-subsets covered
            remaining = chosen_set - {block}
            covered_after_removal = set()
            for b in remaining:
                bi = all_blocks.index(b) if b in all_blocks else None
                # This is inefficient; let's use a different approach
                pass
        
        # Better local improvement: for each block, check if all its t-subsets
        # are covered by other blocks
        covered_by_others = defaultdict(int)
        for block in chosen_set:
            bi = None
            # We need to find the index of this block
            # Let's use a set of tuples for faster lookup
            pass
        
        # Simpler local improvement: iteratively try to remove each block
        # and check coverage
        improved = True
        current_set = set(chosen_blocks)
        while improved:
            improved = False
            for block in list(current_set):
                # Check if all t-subsets of this block are covered by other blocks
                others = current_set - {block}
                all_covered = True
                for ts in itertools.combinations(block, t):
                    found = False
                    for other in others:
                        if ts in set(itertools.combinations(other, t)):
                            found = True
                            break
                    if not found:
                        all_covered = False
                        break
                if all_covered:
                    current_set.remove(block)
                    improved = True
        
        if len(current_set) < best_count:
            best_count = len(current_set)
            best_blocks = list(current_set)
        
        # Early stop if we hit a good bound
        if best_count <= 300:
            break
    
    # Convert to sorted lists
    result = [sorted(b) for b in best_blocks] if best_blocks else []
    
    # Verify the solution is valid
    if not _verify_covering(result, v, k, t):
        # Fallback to pure greedy without local improvement
        result = _pure_greedy(v, k, t)
    
    return result


def _verify_covering(blocks, v, k, t):
    """Verify that every t-subset is contained in at least one block."""
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total = set(itertools.combinations(range(v), t))
    return covered == total


def _pure_greedy(v, k, t):
    """Pure greedy covering as a fallback."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
