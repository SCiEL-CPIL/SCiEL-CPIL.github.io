"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Covering design C(v, k, t) using a multi-strategy approach:
    1. Fast greedy to get an initial covering
    2. Local search (swap/drop) to reduce block count
    3. If time permits, multi-restart with different random seeds
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Helper: count how many uncovered t-subsets a block covers
    def block_covers(block, uncovered_mask):
        """Return list of indices of t-subsets covered by this block."""
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and not uncovered_mask[idx]:
                covered.append(idx)
        return covered
    
    # Helper: get coverage count for a block given uncovered set
    def block_gain(block, uncovered_set):
        """Count how many uncovered t-subsets this block would cover."""
        count = 0
        for ts in itertools.combinations(block, t):
            if ts in uncovered_set:
                count += 1
        return count
    
    # Strategy 1: Greedy covering
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(all_tsubs)
        blocks = []
        
        while uncovered:
            # Pick the first uncovered t-subset (or random for variation)
            if seed == 0:
                first_ts = min(uncovered)
            else:
                first_ts = rng.choice(list(uncovered))
            
            block = list(first_ts)
            
            while len(block) < k:
                # Find point that maximizes coverage of uncovered t-subsets
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        if new_ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
            
            if time.time() > deadline:
                break
        
        return blocks, uncovered
    
    # Strategy 2: Local search to improve a covering
    def local_search(blocks, max_iters=5000):
        """Try to reduce block count by dropping or swapping blocks."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        
        # Build coverage map: for each t-subset, which blocks cover it
        # For efficiency, track coverage counts
        coverage_count = [0] * n_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
        
        uncovered = [i for i in range(n_tsubs) if coverage_count[i] == 0]
        
        if not uncovered:
            return blocks
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try dropping each block
            for i in range(n):
                # Check if block i is redundant
                # i.e., all t-subsets it covers are covered by other blocks
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[ts]
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    n -= 1
                    break
            
            # Try swapping: replace one block with another that covers more uncovered
            if not improved and n > 0:
                # Find uncovered t-subsets
                uncovered_set = set(i for i in range(n_tsubs) if coverage_count[i] == 0)
                if uncovered_set:
                    # Try adding a new block and removing one
                    # This is expensive, so do limited attempts
                    if iter_count % 10 == 0:
                        for i in range(min(n, 10)):
                            # Try to find a block that covers uncovered t-subsets
                            # and then see if we can drop block i
                            # For simplicity, just try greedy addition
                            pass
        
        return blocks
    
    # Strategy 3: Multi-start local search
    def multi_start(num_starts=5):
        best_blocks = None
        best_len = float('inf')
        
        for seed in range(num_starts):
            if time.time() > deadline:
                break
            
            blocks, uncovered = greedy_cover(seed)
            
            if not uncovered:
                # We have a valid covering, try to improve
                blocks = local_search(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks
            
            if time.time() > deadline:
                break
        
        return best_blocks
    
    # Main execution
    # First, get a quick greedy solution
    blocks, uncovered = greedy_cover(0)
    
    if time.time() > deadline:
        # Out of time, return what we have
        return blocks
    
    # If greedy gave us a valid covering, try to improve
    if not uncovered:
        # Apply local search
        blocks = local_search(blocks, max_iters=10000)
        
        # Try multi-start if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            best = multi_start(num_starts=3)
            if best is not None and len(best) < len(blocks):
                blocks = best
    
    # Final check: ensure we return a valid covering
    # If for some reason we don't have a valid covering, do one more greedy pass
    if time.time() < deadline:
        # Verify coverage
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        
        if len(covered) < n_tsubs:
            # Not all t-subsets covered, need more blocks
            uncovered = set(all_tsubs) - covered
            while uncovered and time.time() < deadline:
                first_ts = min(uncovered)
                block = list(first_ts)
                while len(block) < k:
                    best_point = -1
                    best_gain = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            new_ts = tuple(sorted(rest + (x,)))
                            if new_ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
    
    return blocks
# EVOLVE-BLOCK-END
