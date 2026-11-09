import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute for each point, the list of triples containing it?
    # Or for each pair, the list of triples?
    # For greedy block construction, we need to check triples formed by (pair from block, new cand).
    # So for a pair (a,b), we need to quickly check if (a,b,c) is covered.
    # triple_to_idx is O(1).

    # Count of how many times each triple is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []

    def add_block(block):
        blocks.append(tuple(block))
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] += 1

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            count[triple_to_idx[tr]] -= 1

    def get_uncovered_indices():
        return np.where(count == 0)[0]

    def build_block_greedy(start_triple, count_arr, rng_obj):
        """Build a block greedily starting from a given triple."""
        block = list(start_triple)
        remaining = set(range(v))
        for p in start_triple:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            # Get pairs from current block
            # t=3, so we need pairs (t-1=2)
            pairs = list(itertools.combinations(block, t-1))
            
            # Iterate candidates
            # Optimization: if remaining is large, we might want to sample? No, v=24 is small.
            for cand in remaining:
                s = 0
                for p1, p2 in pairs:
                    # Create triple and check coverage
                    # Since block is sorted and cand is not in block, we can sort the triple quickly
                    # But tuple(sorted(...)) is fine for small numbers.
                    tr = tuple(sorted((p1, p2, cand)))
                    if count_arr[triple_to_idx[tr]] == 0:
                        s += 1
                if s > best_s:
                    best_s = s
                    best_cand = cand
            
            if best_cand is None:
                # Fallback: pick first remaining
                best_cand = next(iter(remaining))
            
            block.append(best_cand)
            remaining.discard(best_cand)
        
        return tuple(sorted(block))

    # Phase 1: Greedy construction
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break

        # Sample uncovered triples to find best starting point
        sample_size = min(30, len(uncovered_indices))
        sample_idx = rng.choice(uncovered_indices, size=sample_size, replace=False)

        best_block = None
        best_gain = -1

        for idx in sample_idx:
            tr = all_triples[idx]
            block = build_block_greedy(tr, count, rng)
            # Compute gain
            gain = 0
            for tr2 in itertools.combinations(block, t):
                if count[triple_to_idx[tr2]] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        add_block(best_block)

    # Phase 2: Local improvement - try replacing blocks
    # We can try to remove a block and see if we can cover the newly uncovered triples with a *better* block
    # Or, we can try to swap a block with another that allows removing a different block later?
    # Simple swap: Remove block i, find a block that covers the lost triples.
    # If the new block is "better" in some sense (e.g., covers more other triples), keep it.
    # But how to define better? Maybe just keep if it covers all lost triples.
    # Actually, if we remove a block, we MUST cover its triples.
    # So we need a block that contains all the triples of the removed block? No, just the ones that become uncovered.
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            removed_block = blocks[bi]
            # Identify which triples will become uncovered if we remove this block
            # These are triples in removed_block with count == 1
            critical_triples = []
            for tr in itertools.combinations(removed_block, t):
                if count[triple_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            if not critical_triples:
                # Can remove this block safely
                remove_block(removed_block)
                blocks.pop(bi)
                improved = True
                break
            
            # We need a new block that covers all critical_triples
            # This is a hitting set problem for the critical triples.
            # We can try to build a block greedily that covers them.
            # Start with one of the critical triples?
            # Or try to find a block that contains all of them?
            # Since k=7 and t=3, a block can contain at most 7 points.
            # If critical_triples share many points, we can build a block around them.
            
            # Strategy: Pick the critical triple with the most "overlap" potential?
            # Or just pick the first one and build a block, then check if it covers all criticals.
            # If not, try another.
            
            best_repl = None
            best_score = -1 # Score could be number of *other* uncovered triples covered
            
            # Try building a block from each critical triple
            # To limit time, sample if too many criticals?
            # Max criticals is 35.
            
            for start_tr in critical_triples:
                # Build a block that MUST contain start_tr
                # And ideally contains points from other criticals
                block = build_block_greedy(start_tr, count, rng)
                
                # Check if this block covers all critical_triples
                valid = True
                for tr in critical_triples:
                    if tr not in itertools.combinations(block, t):
                        valid = False
                        break
                
                if valid:
                    # Calculate "quality": how many OTHER uncovered triples does it cover?
                    gain = 0
                    for tr2 in itertools.combinations(block, t):
                        if count[triple_to_idx[tr2]] == 0:
                            gain += 1
                    if gain > best_score:
                        best_score = gain
                        best_repl = block
            
            if best_repl is not None:
                # Replace
                remove_block(removed_block)
                blocks[bi] = best_repl
                add_block(best_repl)
                improved = True
                break
            else:
                # Cannot replace, do nothing (block stays)
                pass

        if not improved:
            break

    # Phase 3: Ensure full coverage
    while time.time() < deadline - 2:
        uncovered_indices = get_uncovered_indices()
        if len(uncovered_indices) == 0:
            break
        tr = all_triples[uncovered_indices[0]]
        block = build_block_greedy(tr, count, rng)
        add_block(block)

    # Final deduplication
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
