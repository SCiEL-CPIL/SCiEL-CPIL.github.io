import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # For fast lookup of triplet index (generalized for t=3, but we keep it general via dict)
    # For t=3, a 3D array is fastest. For general t, dict is safer.
    # Given the task is C(24,7,3), t=3 is fixed in evaluation, but let's write robust code.
    # We'll use a dict for general t, but optimize for t=3 if possible.
    # Actually, let's just use the dict for simplicity and correctness across t, 
    # but precompute the specific triplets for each block candidate to speed up gain.
    
    # Precompute triplets for each possible block? No, too many blocks.
    # Precompute for each pair (if t=3) the list of triplets? 
    # Let's stick to the efficient approach from the current program but refined.
    
    # Using a numpy array for coverage counts is faster for updates.
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    covered_mask = 0  # Bitmask is hard for large n_subsets. C(24,3) = 2024. 2^2024 is too big for int.
    # So we must rely on cover_count and check for zeros.
    
    # To check if covered, we need to know if any count is 0.
    # We can maintain a variable `uncovered_count`.
    uncovered_count = n_subsets

    def add_block(bl):
        nonlocal uncovered_count
        for s in itertools.combinations(bl, t):
            idx = subset_to_index[s]
            if cover_count[idx] == 0:
                uncovered_count -= 1
            cover_count[idx] += 1

    def remove_block(bl):
        nonlocal uncovered_count
        for s in itertools.combinations(bl, t):
            idx = subset_to_index[s]
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                uncovered_count += 1

    def is_covered():
        return uncovered_count == 0

    # Greedy construction with multiple seeds
    def greedy_construct(seed):
        rng = random.Random(seed)
        blocks = []
        # We need a fast way to find an uncovered subset to start with.
        # Scanning all subsets is O(n_subsets). n_subsets=2024. This is fast.
        
        while uncovered_count > 0:
            if time.time() > deadline - 5:
                break
            
            # Find first uncovered subset
            seed_idx = -1
            # Optimization: instead of scanning all, we could keep a list of uncovered indices?
            # But removals are rare in greedy. Scanning is fine.
            for i in range(n_subsets):
                if cover_count[i] == 0:
                    seed_idx = i
                    break
            
            if seed_idx == -1:
                break # Should not happen if uncovered_count > 0
                
            seed_sub = all_subsets[seed_idx]
            B = list(seed_sub)
            B_set = set(B)
            
            while len(B) < k:
                best_x = -1
                best_gain = -1
                
                # Precompute pairs in B for t=3 gain calculation
                # gain(x) = number of pairs (u,v) in B such that {x,u,v} is uncovered
                pairs_in_B = list(itertools.combinations(B, t-1)) if t > 1 else [()]
                
                for x in range(v):
                    if x in B_set:
                        continue
                    
                    gain = 0
                    for pair in pairs_in_B:
                        # Form the t-subset
                        # For t=3, pair is (u,v). subset is sorted((x,u,v))
                        # We need to map this to an index.
                        # Creating a tuple and looking up in dict is O(t).
                        # For t=3, we can optimize.
                        if t == 3:
                            u, w = pair
                            # Ensure sorted order for dict key
                            a, b, c = sorted((x, u, w))
                            idx = subset_to_index[(a, b, c)]
                        else:
                            # General case
                            sub = sorted(pair + (x,))
                            idx = subset_to_index[tuple(sub)]
                        
                        if cover_count[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    # No gain, pick random
                    candidates = [x for x in range(v) if x not in B_set]
                    if not candidates:
                        break
                    best_x = rng.choice(candidates)
                
                B.append(best_x)
                B_set.add(best_x)
            
            bl = sorted(B)
            blocks.append(bl)
            add_block(bl)
        
        return blocks

    # Initial construction with multiple seeds
    best_blocks = None
    best_count = float('inf')
    
    # Reserve time for optimization
    opt_time = 60.0
    construct_deadline = start_time + (deadline - start_time - opt_time)
    
    num_seeds = 10
    for seed in range(num_seeds):
        if time.time() > construct_deadline:
            break
        
        # Reset state for new greedy run
        cover_count[:] = 0
        uncovered_count = n_subsets
        
        blocks = greedy_construct(seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            # Store the cover_count state? No, we recompute for optimization.
            # But we need to restore cover_count for the best blocks if we proceed.
            # Actually, greedy_construct modifies the global cover_count.
            # We need to snapshot the best cover_count or recompute it.
            # Recomputing is O(B * C(k,t)). 76 * 35 = 2660 ops. Fast.
            
    if best_blocks is None:
        # Fallback
        cover_count[:] = 0
        uncovered_count = n_subsets
        best_blocks = greedy_construct(0)

    # Reset state for optimization phase
    cover_count[:] = 0
    uncovered_count = n_subsets
    for b in best_blocks:
        add_block(b)
    
    # Local Search
    def try_remove_redundant():
        nonlocal blocks, cover_count, uncovered_count
        improved = True
        while improved:
            if time.time() > deadline - 2:
                return
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2:
                    return
                bl = blocks[i]
                redundant = True
                for s in itertools.combinations(bl, t):
                    idx = subset_to_index[s]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(bl)
                    blocks.pop(i)
                    improved = True

    def try_swaps():
        nonlocal blocks, cover_count, uncovered_count
        improved = True
        while improved:
            if time.time() > deadline - 2:
                return
            improved = False
            n = len(blocks)
            # Try 2-for-1 swap: remove two blocks, add one block that covers all uniquely covered triplets
            # This is powerful but expensive.
            # Try 1-for-1 swap: replace one block with another that covers its unique triplets and reduces total count?
            # Actually, 1-for-1 doesn't reduce count. It might improve structure for future removals.
            # The inspiration had 2-for-1. Let's implement a simplified 2-for-1.
            
            # Iterate over pairs
            for i in range(n):
                if time.time() > deadline - 2:
                    return
                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        return
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    # Find triplets uniquely covered by b1 or b2
                    # Unique to b1: count==1 and in b1
                    # Unique to b2: count==1 and in b2
                    # If we remove both, these become uncovered.
                    # We need a new block that covers ALL of them.
                    
                    unique_to_b1 = []
                    unique_to_b2 = []
                    
                    for s in itertools.combinations(b1, t):
                        idx = subset_to_index[s]
                        if cover_count[idx] == 1:
                            unique_to_b1.append(s)
                    
                    for s in itertools.combinations(b2, t):
                        idx = subset_to_index[s]
                        if cover_count[idx] == 1:
                            unique_to_b2.append(s)
                    
                    # If any triplet is covered by both b1 and b2 with count 1? Impossible, count would be >=2.
                    # So unique_to_b1 and unique_to_b2 are disjoint in terms of "uniqueness" relative to the set of blocks.
                    # However, a triplet could be in both b1 and b2. If count is 1, it must be in only one.
                    
                    all_unique = unique_to_b1 + unique_to_b2
                    
                    if not all_unique:
                        # Both are redundant?
                        # We can remove one, then check the other.
                        # Or remove both and add nothing? No, we need to cover their shared non-unique parts?
                        # If they are both redundant, removing one makes the other potentially redundant too.
                        # Let's just remove one.
                        remove_block(b1)
                        blocks.pop(i)
                        improved = True
                        break
                        # Note: indices shift. Restarting the loop is safer.
                    
                    # Check if all_unique points fit in a block of size k
                    points_needed = set()
                    for s in all_unique:
                        points_needed.update(s)
                    
                    if len(points_needed) > k:
                        continue
                    
                    # Construct new block
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        # Add random point
                        p = random.randint(0, v-1)
                        if p not in new_block:
                            new_block.append(p)
                    
                    new_block.sort()
                    
                    # Check if this new block is valid (distinct points)
                    if len(set(new_block)) != k:
                        continue
                        
                    # Check if it actually covers all unique triplets
                    # It should, by construction of points_needed.
                    # But we need to ensure it doesn't introduce issues?
                    # We are replacing 2 blocks with 1.
                    # We need to ensure that all triplets in b1 and b2 that are NOT unique to them
                    # are still covered by other blocks.
                    # Since they were not unique, count >= 2. Removing b1 and b2 reduces count by 2 (if in both) or 1 (if in one).
                    # If count was 2 and in both, it becomes 0 -> BAD.
                    # If count was 2 and in one, it becomes 1 -> OK.
                    
                    # So we must check: for every triplet in b1 or b2, if it is NOT in new_block,
                    # its count must remain >= 1 after removing b1 and b2.
                    
                    # Let's simulate removal
                    # Save counts
                    # Actually, let's just check the condition directly.
                    
                    valid = True
                    # For each triplet in b1
                    for s in itertools.combinations(b1, t):
                        idx = subset_to_index[s]
                        # How many times is s covered by b1 and b2?
                        in_b1 = 1
                        in_b2 = 1 if s in itertools.combinations(b2, t) else 0
                        # Wait, s is a tuple. Checking membership in combinations is slow.
                        # Better: precompute set of triplets for b2?
                        # Or just check points.
                        # s is a tuple of t points.
                        # s is in b2 if all points of s are in b2.
                        in_b2 = 1 if all(p in b2 for p in s) else 0
                        
                        # Is s in new_block?
                        in_new = 1 if all(p in new_block for p in s) else 0
                        
                        # New count = old_count - in_b1 - in_b2 + in_new
                        # We need new_count >= 1
                        # old_count - in_b1 - in_b2 + in_new >= 1
                        # old_count >= 1 + in_b1 + in_b2 - in_new
                        
                        if cover_count[idx] < 1 + in_b1 + in_b2 - in_new:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Perform swap
                    remove_block(b1)
                    remove_block(b2)
                    add_block(new_block)
                    
                    # Update blocks list
                    # Remove j first if j > i, then i
                    del blocks[j]
                    del blocks[i]
                    blocks.append(new_block)
                    
                    improved = True
                    break
                if improved:
                    break

    # Run optimization loop
    while time.time() < deadline - 5:
        try_remove_redundant()
        if is_covered():
            try_swaps()
            if is_covered():
                try_remove_redundant()
            else:
                # Swaps broke coverage? Should not happen if logic is correct.
                # If it does, we need to revert. 
                # For safety, if coverage is lost, we stop swapping.
                pass
        else:
            # Coverage lost during removal? 
            # try_remove_redundant should only remove redundant blocks, so coverage shouldn't be lost.
            # Unless there's a bug.
            pass

    return best_blocks

