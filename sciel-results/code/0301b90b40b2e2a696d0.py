"""Covering design C(19,8,5): Local search with block swaps.
Strategy:
1. Generate an initial valid covering using a fast greedy heuristic.
2. Perform local search: repeatedly try to replace a block with another 
   block that allows removing more redundant blocks or simply reduces the set size.
   Specifically, we look for a block B in the current solution such that 
   removing B leaves some t-subsets uncovered. We then try to find a new block B' 
   that covers those uncovered subsets and potentially allows removing other blocks.
   A simpler and effective heuristic for this specific small problem:
   - Start with a valid solution.
   - Try to remove any redundant block.
   - If no redundant block, try to swap a block with another block that results in 
     a smaller total count after redundancy removal.
"""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    # This is memory intensive but v=19, k=8 -> C(19,8)=75582 blocks.
    # Each block covers C(8,5)=56 t-subsets.
    # Total entries: 75582 * 56 ~ 4.2 million. This is manageable in memory if stored efficiently.
    # However, computing this upfront might take time. Let's do it on the fly or cache.
    
    # Better approach: 
    # 1. Get a valid solution quickly (Greedy).
    # 2. Use a set of covered t-subsets to check validity.
    # 3. Local search: 
    #    - Identify redundant blocks (those whose coverage is fully provided by others). Remove them.
    #    - If no redundant blocks, try to "swap" a block. 
    #      Actually, a common heuristic is:
    #      Pick a block B. Remove it. See what becomes uncovered.
    #      Find a block B' that covers the most of those uncovered subsets.
    #      Add B'. Check if this allows removing other blocks.
    #      If total size decreases or stays same but we break tie, accept.
    
    def get_initial_solution():
        """Fast greedy to get a valid starting point."""
        uncovered = set(all_t)
        blocks = []
        while uncovered:
            # Pick the t-subset with highest frequency? Or just min?
            # To be fast, just pick min.
            seed_t = min(uncovered)
            block = list(seed_t)
            # Greedily add points to cover more
            while len(block) < k:
                best_gain = -1
                best_pts = []
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pts = [x]
                    elif gain == best_gain:
                        best_pts.append(x)
                if not best_pts:
                    break
                block.append(random.choice(best_pts))
            if len(block) < k:
                # Fill with random points
                remaining = [x for x in range(v) if x not in block]
                random.shuffle(remaining)
                block.extend(remaining[:k - len(block)])
            block.sort()
            blocks.append(block)
            # Update uncovered
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
        return blocks

    def is_redundant(block_idx, blocks, cover_counts):
        """Check if block at block_idx is redundant given cover_counts."""
        b = blocks[block_idx]
        for s in itertools.combinations(b, t):
            i = idx[s]
            if cover_counts[i] == 1:
                return False
        return True

    def calculate_cover_counts(blocks):
        counts = np.zeros(N, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                counts[idx[s]] += 1
        return counts

    def try_remove_redundant(blocks):
        """Iteratively remove redundant blocks. Returns new list of blocks."""
        changed = True
        while changed:
            changed = False
            counts = calculate_cover_counts(blocks)
            for i in range(len(blocks)):
                if is_redundant(i, blocks, counts):
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def local_search(blocks, deadline):
        """Perform local search by swapping blocks."""
        # Convert to list of tuples for hashing? No, lists are fine.
        current_blocks = list(blocks)
        current_size = len(current_blocks)
        
        # Precompute coverage for current solution
        counts = calculate_cover_counts(current_blocks)
        
        # Map block -> index in list?
        # We need to efficiently find which blocks cover a specific t-subset.
        # Let's maintain a list of lists: coverers[t_idx] = list of block indices covering it.
        
        def build_coverers(blocks):
            coverers = [[] for _ in range(N)]
            for i, b in enumerate(blocks):
                for s in itertools.combinations(b, t):
                    coverers[idx[s]].append(i)
            return coverers

        coverers = build_coverers(current_blocks)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Try to find a redundant block
            redundant_found = False
            for i in range(len(current_blocks)):
                if is_redundant(i, current_blocks, counts):
                    # Remove it
                    b_to_remove = current_blocks[i]
                    for s in itertools.combinations(b_to_remove, t):
                        si = idx[s]
                        counts[si] -= 1
                        # Update coverers
                        if i in coverers[si]:
                            coverers[si].remove(i)
                    current_blocks.pop(i)
                    # Shift indices in coverers for blocks after i
                    for si in range(N):
                        for j in range(len(coverers[si])):
                            if coverers[si][j] > i:
                                coverers[si][j] -= 1
                    redundant_found = True
                    no_improve_count = 0
                    break
            
            if redundant_found:
                continue
                
            # No redundant blocks. Try a swap.
            # Pick a random block to remove.
            if len(current_blocks) == 0:
                break
                
            i = random.randrange(len(current_blocks))
            b_orig = current_blocks[i]
            
            # Simulate removing b_orig
            uncovered_by_removal = []
            for s in itertools.combinations(b_orig, t):
                si = idx[s]
                if counts[si] == 1:
                    uncovered_by_removal.append(si)
            
            if not uncovered_by_removal:
                # This block was redundant? Should have been caught above.
                # Maybe counts were stale?
                continue

            # Find a block b_new that covers the most of uncovered_by_removal
            # and is not already in current_blocks.
            # We can't iterate all 75k blocks every time, it's too slow.
            # Heuristic: Pick a random t-subset from uncovered_by_removal, 
            # and find random blocks containing it.
            
            # Pick a random uncovered t-subset
            target_si = random.choice(uncovered_by_removal)
            target_t = all_t[target_si]
            
            # Find all blocks containing target_t? 
            # Precomputation: For each t-subset, list of blocks containing it.
            # This is expensive to build (75k * 56 checks).
            # Alternative: Just try random blocks from the pool?
            # Or, generate a block greedily to cover uncovered_by_removal.
            
            # Let's try a "repair" strategy:
            # Start with b_orig, remove one point, add a different point.
            # This is a 1-swap local search.
            
            best_swap = None
            best_score = len(current_blocks) # We want to reduce size, or keep same if it leads to future reduction
            
            # Try swapping one point in b_orig
            for j in range(k):
                p_out = b_orig[j]
                b_partial = [x for x in b_orig if x != p_out]
                
                for p_in in range(v):
                    if p_in in b_partial:
                        continue
                    b_new = tuple(sorted(b_partial + [p_in]))
                    if b_new == b_orig:
                        continue
                    if b_new in current_blocks:
                        continue
                    
                    # Calculate gain: How many new t-subsets does b_new cover that were not covered by others?
                    # Specifically, how many of the uncovered_by_removal does it cover?
                    # And does it cover any other subsets that might allow removing other blocks?
                    
                    # Simple heuristic: Count how many of the specific uncovered_by_removal it covers.
                    covered_count = 0
                    for si in uncovered_by_removal:
                        s = all_t[si]
                        if all(x in b_new for x in s):
                            covered_count += 1
                    
                    # If it covers all, it's a candidate.
                    if covered_count == len(uncovered_by_removal):
                        # Check if this swap allows removing other blocks.
                        # This requires a full simulation which is expensive.
                        # Let's just accept it if it covers all, and rely on the next iteration's redundancy check.
                        best_swap = b_new
                        break
                if best_swap:
                    break
            
            if best_swap:
                # Perform the swap
                # Remove b_orig
                for s in itertools.combinations(b_orig, t):
                    si = idx[s]
                    counts[si] -= 1
                    if i in coverers[si]:
                        coverers[si].remove(i)
                
                # Add best_swap
                new_idx = len(current_blocks)
                current_blocks[i] = best_swap
                for s in itertools.combinations(best_swap, t):
                    si = idx[s]
                    counts[si] += 1
                    coverers[si].append(new_idx)
                
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return current_blocks

    # Main Logic
    best_blocks = None
    best_size = float('inf')
    
    # Run a few greedy starts
    for seed in range(10):
        if time.time() > deadline:
            break
        random.seed(seed)
        try:
            blocks = get_initial_solution()
            # Clean up redundant
            blocks = try_remove_redundant(blocks)
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
        except Exception:
            pass

    if best_blocks is None:
        # Fallback
        return []

    # Local search on the best found so far
    if time.time() < deadline - 10:
        try:
            ls_blocks = local_search(best_blocks, deadline - 5)
            # Verify validity of ls_blocks
            counts = calculate_cover_counts(ls_blocks)
            if np.all(counts > 0):
                if len(ls_blocks) < best_size:
                    best_blocks = ls_blocks
                    best_size = len(ls_blocks)
        except Exception:
            pass

    return best_blocks
