import itertools
import random
import time
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    n_subs = len(all_t_subs)
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    
    # Map each block to the set of t-subsets it covers (as a frozenset of indices for fast lookup)
    # Using bitmasks might be faster for intersection, but frozensets are easier to reason about for CSP domains
    # Given n_subs = C(19,4) = 3876, a bitmask is a single Python int (arbitrary precision), which is very fast.
    block_masks = {}
    for b in all_k_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << sub_to_idx[sub]
        block_masks[b] = mask
        
    # Precompute for each t-subset, the list of blocks that cover it
    # This is crucial for forward checking: if we need to cover subset S, we must pick one of these blocks.
    sub_to_blocks = [[] for _ in range(n_subs)]
    for b in all_k_blocks:
        mask = block_masks[b]
        # Iterate through bits set in mask
        # Efficient way to iterate bits:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            sub_to_blocks[idx].append(b)
            m ^= lsb
            
    # Sort blocks for each subset by how "rare" they are? 
    # Actually, for CSP, we want to order the *choices* (blocks) for a variable (uncovered subset).
    # A good heuristic: prefer blocks that cover many *other* currently uncovered subsets.
    # But that depends on state. 
    # Static heuristic: prefer blocks with higher degree? 
    # Let's stick to dynamic ordering in the solver.

    best_blocks = None
    best_size = float('inf')
    
    # Symmetry Breaking: Fix the first block to be (0, 1, ..., k-1)
    # Since the problem is symmetric under permutation of points, we can assume WLOG that 
    # the block containing point 0 is (0, 1, ..., k-1) IF such a block exists in the optimal solution.
    # Wait, it's not guaranteed that (0,1,...,k-1) is in the optimal solution.
    # However, we can fix that the block containing the lexicographically smallest uncovered 4-subset 
    # is the one we pick. 
    # A stronger symmetry break: 
    # 1. The first block selected must contain point 0.
    # 2. Among blocks containing 0, we can fix the first one to be the lexicographically smallest 
    #    block containing 0 that appears in the solution? No, that's hard to enforce in backtracking.
    
    # Standard symmetry break for covering designs:
    # Fix the first block to be {0, 1, ..., k-1}. 
    # Is this valid? 
    # Any covering can be permuted so that a specific block is {0, 1, ..., k-1}.
    # But we don't know which block to permute. 
    # However, we can argue: Let B be the lexicographically smallest block in the solution.
    # By permuting points, we can map B to {0, 1, ..., k-1}.
    # So there exists an optimal solution containing {0, 1, ..., k-1}.
    # Therefore, we can safely restrict our search to solutions that contain {0, 1, ..., k-1}.
    
    initial_block = tuple(range(k))
    initial_mask = block_masks[initial_block]
    
    # We need to cover all subsets not in initial_mask.
    # Let's define the state as (uncovered_mask, current_blocks).
    
    def solve(uncovered_mask, current_blocks, depth):
        nonlocal best_blocks, best_size, deadline, start_time
        
        if time.time() > deadline:
            return False
            
        if not uncovered_mask:
            # Found a solution
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
            return True
            
        # Pruning: If current size + lower_bound > best_size, prune
        # Lower bound: max over all uncovered subsets s of (1 / max_blocks_covering_s) ? No.
        # Simple LB: number of uncovered subsets / max coverage per block (k choose t)
        # But blocks overlap. 
        # A better LB: For each uncovered subset, we need at least 1 block.
        # If we pick a block, it covers at most C(k,t) subsets.
        # So LB = ceil(popcount(uncovered_mask) / C(k,t)).
        # This is weak.
        # Better LB: For each point, count how many uncovered subsets contain it?
        # Let's use a simple greedy lower bound: 
        # Pick the subset with fewest covering blocks remaining.
        # If that number is 0, fail.
        # LB = 1 + solve(uncovered_mask without those covered by best block)? Too expensive.
        
        # Heuristic LB: 
        # Let's just check if we've exceeded best_size
        if len(current_blocks) >= best_size:
            return False
            
        # Variable Selection: Pick the uncovered subset with the fewest available covering blocks.
        # This is "Minimum Remaining Values" (MRV) heuristic.
        # We need to know, for each uncovered subset, how many blocks in `all_k_blocks` cover it 
        # AND are not already "ruled out"? 
        # In this formulation, we haven't ruled out blocks, we just haven't picked them yet.
        # But if we pick a block, we don't rule out others, we just cover subsets.
        # So the "domain" of a subset S is the set of all blocks covering S.
        # This domain is static! It doesn't shrink as we pick blocks, unless we enforce that 
        # we can't pick the same block twice (which we don't).
        # Wait, if we pick block B, it covers S. S is no longer uncovered.
        # So the "hard" subset is the one with the fewest total blocks covering it in the entire universe.
        # This is static.
        
        # So, find the uncovered subset with the minimum number of covering blocks.
        # Precompute static degrees:
        # degree[s] = len(sub_to_blocks[s])
        
        # To find the min degree uncovered subset efficiently:
        # We can iterate through uncovered bits.
        # n_subs = 3876. This is fast enough.
        
        min_deg = float('inf')
        target_sub_idx = -1
        candidates = []
        
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = len(sub_to_blocks[idx])
            if deg < min_deg:
                min_deg = deg
                target_sub_idx = idx
                candidates = [idx]
            elif deg == min_deg:
                candidates.append(idx)
            m ^= lsb
            
        if target_sub_idx == -1:
            return True # Should not happen if uncovered_mask was non-zero
            
        # If min_deg is 0, we are stuck
        if min_deg == 0:
            return False
            
        # Value Ordering: For the chosen subset, order its covering blocks by how many 
        # OTHER uncovered subsets they cover.
        # We want to pick the block that covers the most uncovered stuff.
        
        best_block = None
        best_gain = -1
        
        for b in sub_to_blocks[target_sub_idx]:
            # Calculate gain: how many bits in block_masks[b] are set in uncovered_mask
            gain = (block_masks[b] & uncovered_mask).bit_count()
            
            # Tie-breaking: prefer blocks with lower lexicographic order? 
            # Or random to avoid local minima?
            # Let's add a bit of randomness to break ties and explore different branches.
            if gain > best_gain:
                best_gain = gain
                best_block = b
            elif gain == best_gain:
                # Random tie break
                if random.random() < 0.5:
                    best_block = b
                    
        if best_block is None:
            return False
            
        new_uncovered = uncovered_mask & ~block_masks[best_block]
        new_blocks = current_blocks + [best_block]
        
        if solve(new_uncovered, new_blocks, depth + 1):
            return True
            
        # If the best choice failed, try other choices?
        # Backtracking search usually tries all values.
        # But for covering designs, the "best" greedy choice is often part of the optimal solution.
        # However, to be complete, we must try others if the first fails.
        # But this can explode.
        # Strategy: Try top-N blocks by gain.
        
        # Let's collect top blocks
        sorted_blocks = sorted(sub_to_blocks[target_sub_idx], 
                               key=lambda b: (block_masks[b] & uncovered_mask).bit_count(), 
                               reverse=True)
        
        # Try top 5 blocks
        for b in sorted_blocks[:5]:
            if b == best_block:
                continue
            new_uncovered = uncovered_mask & ~block_masks[b]
            new_blocks = current_blocks + [b]
            if solve(new_uncovered, new_blocks, depth + 1):
                return True
                
        return False

    # Initial state
    # We fix the first block to be (0, 1, ..., k-1)
    # This block is valid.
    # Uncovered mask is all ones except those covered by initial_block.
    initial_uncovered = ((1 << n_subs) - 1) & ~initial_mask
    
    # Start search
    # We need to ensure the search doesn't run away.
    # The deadline check is inside solve.
    
    # To increase chances of finding a good solution, we can run multiple searches 
    # with different random seeds or slight variations in tie-breaking.
    # But CSP with backtracking is deterministic if tie-breaking is fixed.
    # Let's run it once. If it finds a solution, great.
    # If it times out, we return the best found (which might be None if it didn't complete).
    # If it doesn't find a solution within time, we fall back to a greedy heuristic to ensure we return *something*.
    
    # Actually, the task requires a valid covering. 
    # If CSP fails to find one in time, we must provide a valid one.
    # So we should keep a fallback greedy solution.
    
    # Let's run a quick greedy first to have a baseline best_size and best_blocks.
    # This also serves as a fallback.
    
    def greedy_cover():
        uncovered = ((1 << n_subs) - 1) & ~initial_mask
        blocks = [initial_block]
        while uncovered:
            if time.time() > deadline:
                break
            # Pick subset with min degree
            min_deg = float('inf')
            target_idx = -1
            m = uncovered
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = len(sub_to_blocks[idx])
                if deg < min_deg:
                    min_deg = deg
                    target_idx = idx
                m ^= lsb
                
            if target_idx == -1:
                break
                
            # Pick block with max gain
            best_b = None
            best_gain = -1
            for b in sub_to_blocks[target_idx]:
                gain = (block_masks[b] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
                    
            if best_b is None:
                break
                
            blocks.append(best_b)
            uncovered &= ~block_masks[best_b]
            
        return blocks

    # Run greedy to get a baseline
    greedy_res = greedy_cover()
    if greedy_res:
        best_size = len(greedy_res)
        best_blocks = greedy_res
        
    # Now run CSP search trying to beat the greedy solution
    # Note: The CSP search starts with initial_block fixed.
    # It will try to find a solution smaller than best_size.
    
    # To make CSP more robust, we can allow it to run for a portion of the time,
    # and if it doesn't beat the greedy, we return the greedy.
    
    # We modify solve to stop if it finds a solution better than best_size.
    # It already does that (updates best_blocks).
    
    # Let's run the CSP search.
    # We need to be careful with recursion depth and time.
    
    try:
        # The search might be very slow. 
        # We rely on the deadline check inside solve.
        solve(initial_uncovered, [initial_block], 0)
    except RecursionError:
        pass
        
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]

# Test locally
if __name__ == "__main__":
    # Test with small parameters to ensure it works
    # C(19, 7, 4) is the target.
    # Let's just run the construct function.
    res = construct(19, 7, 4)
    print(f"Found {len(res)} blocks")
    # Verify
    v, k, t = 19, 7, 4
    all_subs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in res:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    if all_subs == covered:
        print("Valid covering")
    else:
        print("Invalid covering")
