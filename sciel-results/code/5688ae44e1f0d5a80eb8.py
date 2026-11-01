"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _combo_index(combo, v, t):
    """Convert a t-combination of range(v) to a unique integer index."""
    # Use combinatorial number system
    idx = 0
    for i in range(t):
        # C(combo[i], i)
        c = 1
        for j in range(i):
            c = c * (combo[i] - j) // (j + 1)
        idx += c
    return idx


def _build_index_map(v, t):
    """Build mapping from combo tuple to index, and list of all combos."""
    combos = list(itertools.combinations(range(v), t))
    idx_map = {}
    for i, c in enumerate(combos):
        idx_map[c] = i
    return combos, idx_map


def _greedy_cover_fast(v, k, t, max_blocks=None, seed=42):
    """Fast greedy set cover using bitsets for coverage tracking."""
    rng = random.Random(seed)
    total_combos = len(list(itertools.combinations(range(v), t)))
    
    # Precompute all C(v,t) combos and their indices
    all_combos = list(itertools.combinations(range(v), t))
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}
    
    # uncovered: boolean array
    uncovered = np.ones(total_combos, dtype=bool)
    uncovered_count = total_combos
    
    blocks = []
    deadline = time.time() + 100  # Leave margin for other operations
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        if max_blocks and len(blocks) >= max_blocks:
            break
            
        # Pick the first uncovered combo to start a block
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        start_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
        start_combo = all_combos[start_idx]
        
        block = list(start_combo)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_combo = tuple(sorted(rest + (x,)))
                    if new_combo in combo_to_idx:
                        ci = combo_to_idx[new_combo]
                        if uncovered[ci]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and best_x != -1:
                    # Tie-breaking: prefer point that appears in more uncovered combos
                    pass
                    
            if best_x == -1 or best_gain <= 0:
                # No improvement possible, pick random
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            
            block.append(best_x)
        
        if len(block) != k:
            continue
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for combo in itertools.combinations(block, t):
            ci = combo_to_idx.get(combo)
            if ci is not None and uncovered[ci]:
                uncovered[ci] = False
                uncovered_count -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # For each block, compute which t-subsets it covers
    block_subsets = []
    for b in blocks:
        subs = list(itertools.combinations(b, t))
        block_subsets.append(subs)
    
    # Count coverage of each subset
    from collections import defaultdict
    subset_to_blocks = defaultdict(list)
    for bi, subs in enumerate(block_subsets):
        for s in subs:
            subset_to_blocks[s].append(bi)
    
    # A block is redundant if for every subset it covers, there's another block covering it
    keep = [True] * len(blocks)
    for bi in range(len(blocks)):
        if not keep[bi]:
            continue
        redundant = True
        for s in block_subsets[bi]:
            if len(subset_to_blocks[s]) == 1:
                redundant = False
                break
        if redundant:
            keep[bi] = False
    
    result = [blocks[i] for i in range(len(blocks)) if keep[i]]
    return result


def _local_search(blocks, v, k, t, deadline, seed=42):
    """Try to reduce block count by replacing pairs of blocks with one."""
    rng = random.Random(seed)
    blocks = list(blocks)
    
    # Build subset-to-block-index mapping
    from collections import defaultdict
    subset_to_blocks = defaultdict(list)
    for bi, b in enumerate(blocks):
        for s in itertools.combinations(b, t):
            subset_to_blocks[s].append(bi)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each block and see if coverage is maintained
        for bi in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if block bi is redundant
            redundant = True
            for s in itertools.combinations(blocks[bi], t):
                others = [x for x in subset_to_blocks[s] if x != bi]
                if len(others) == 0:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                for s in itertools.combinations(blocks[bi], t):
                    subset_to_blocks[s].remove(bi)
                del blocks[bi]
                # Reindex
                for s in subset_to_blocks:
                    subset_to_blocks[s] = sorted([x if x < bi else x - 1 for x in subset_to_blocks[s]])
                improved = True
    
    return blocks


def _verify_covering(blocks, v, k, t):
    """Check if all t-subsets are covered."""
    covered = set()
    for b in blocks:
        for s in itertools.combinations(b, t):
            covered.add(s)
    total = len(list(itertools.combinations(range(v), t)))
    return len(covered) == total


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for verification and return
    
    # Multiple seeds for diversity
    best_blocks = None
    best_count = float('inf')
    
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        blocks = _greedy_cover_fast(v, k, t, max_blocks=150, seed=seed)
        
        if not blocks:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        
        # Local search to remove more redundant blocks
        blocks = _local_search(blocks, v, k, t, deadline, seed=seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Early stop if we get a good result
        if best_count <= 78:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover_fast(v, k, t, max_blocks=200, seed=42)
    
    # Final verification
    if not _verify_covering(best_blocks, v, k, t):
        # If invalid, try again with more blocks
        fallback = _greedy_cover_fast(v, k, t, max_blocks=300, seed=999)
        if _verify_covering(fallback, v, k, t):
            best_blocks = fallback
        else:
            # Last resort: brute force greedy without limits
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
            best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
