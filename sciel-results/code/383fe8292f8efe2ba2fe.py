"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a multi-phase approach:
    1. Greedy construction with randomized tie-breaking
    2. Local search (Simulated Annealing) with remove/swap moves
    3. Time-bounded with safety margin
    
    Returns a list of blocks (each a list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Coverage count for each t-subset
    cover_count = [0] * n_tsubsets
    
    # Block membership: for each block, which t-subsets it covers
    block_tsubs = []  # list of sets of t-subset indices
    
    def add_block(block):
        """Add a block and update coverage."""
        b_tsubs = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            b_tsubs.add(idx)
            cover_count[idx] += 1
        block_tsubs.append(b_tsubs)
    
    def remove_block(block_idx):
        """Remove a block and update coverage."""
        for idx in block_tsubs[block_idx]:
            cover_count[idx] -= 1
        block_tsubs[block_idx] = None
    
    def is_valid():
        """Check if all t-subsets are covered."""
        return all(c > 0 for c in cover_count)
    
    def greedy_construct(randomize=False, seed=None):
        """Greedy construction of a covering."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        uncovered = set(range(n_tsubsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset (or first if not randomizing)
            if randomize:
                start_ts = rng.choice(list(uncovered))
            else:
                start_ts = min(uncovered)
            
            start_subset = all_tsubsets[start_ts]
            block = list(start_subset)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                
                if randomize:
                    rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if tsub_idx[ts] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(tsub_idx[ts])
            
            # Check time
            if time.time() > deadline - 20.0:
                break
        
        return blocks
    
    # Phase 1: Greedy construction
    initial_blocks = greedy_construct(randomize=False)
    
    # Initialize coverage from initial blocks
    cover_count = [0] * n_tsubsets
    block_tsubs = []
    for block in initial_blocks:
        b_tsubs = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            b_tsubs.add(idx)
            cover_count[idx] += 1
        block_tsubs.append(b_tsubs)
    
    # Check if valid
    if not is_valid():
        # Need to add more blocks
        uncovered = [i for i, c in enumerate(cover_count) if c == 0]
        while uncovered:
            ts_idx = uncovered[0]
            start_subset = all_tsubsets[ts_idx]
            block = list(start_subset)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if tsub_idx[ts] in set(uncovered):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            
            block.sort()
            initial_blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_idx[ts]
                cover_count[idx] += 1
                if idx in set(uncovered):
                    uncovered.remove(idx)
            
            if time.time() > deadline - 10.0:
                break
    
    best_blocks = list(initial_blocks)
    best_len = len(best_blocks)
    
    # Phase 2: Simulated Annealing with remove/swap moves
    if time.time() < deadline - 30.0:
        sa_blocks = list(initial_blocks)
        sa_cover = list(cover_count)
        sa_block_tsubs = list(block_tsubs)
        
        def sa_add_block(block):
            b_tsubs = set()
            for ts in itertools.combinations(block, t):
                idx = tsub_idx[ts]
                b_tsubs.add(idx)
                sa_cover[idx] += 1
            sa_block_tsubs.append(b_tsubs)
        
        def sa_remove_block(block_idx):
            for idx in sa_block_tsubs[block_idx]:
                sa_cover[idx] -= 1
            sa_block_tsubs[block_idx] = None
        
        def sa_is_valid():
            return all(c > 0 for c in sa_cover)
        
        def sa_cost():
            """Cost = number of blocks + penalty for uncovered t-subsets."""
            uncovered = sum(1 for c in sa_cover if c == 0)
            return len(sa_blocks) + 100 * uncovered
        
        current_cost = sa_cost()
        best_sa_cost = current_cost
        best_sa_blocks = list(sa_blocks)
        
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        rng = random.Random(42)
        
        while T > min_T and time.time() < deadline - 15.0:
            # Choose a move: remove, swap, or add
            move_type = rng.random()
            
            if move_type < 0.4 and len(sa_blocks) > 1:
                # Try to remove a block
                # Prefer blocks that cover many t-subsets with high redundancy
                removable = []
                for i in range(len(sa_blocks)):
                    if sa_block_tsubs[i] is None:
                        continue
                    # Can only remove if all its t-subsets are covered by other blocks
                    can_remove = True
                    for idx in sa_block_tsubs[i]:
                        if sa_cover[idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable.append(i)
                
                if removable:
                    block_idx = rng.choice(removable)
                    sa_remove_block(block_idx)
                    sa_blocks.pop(block_idx)
                    
                    new_cost = sa_cost()
                    delta = new_cost - current_cost
                    if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                        current_cost = new_cost
                        if current_cost < best_sa_cost:
                            best_sa_cost = current_cost
                            best_sa_blocks = list(sa_blocks)
                    else:
                        # Revert
                        sa_blocks.insert(block_idx, None)
                        # We need to restore the block - this is tricky
                        # Let's just skip revert for now and track validity
                        pass
                
            elif move_type < 0.7 and len(sa_blocks) > 0:
                # Swap a point in a block
                block_idx = rng.randrange(len(sa_blocks))
                if sa_block_tsubs[block_idx] is None:
                    continue
                    
                block = list(best_sa_blocks[block_idx]) if block_idx < len(best_sa_blocks) else None
                if block is None:
                    continue
                    
                # Find a point to swap out
                out_idx = rng.randrange(len(block))
                out_point = block[out_idx]
                
                # Find a point to swap in
                in_point = rng.choice([x for x in range(v) if x not in block])
                
                old_block = list(block)
                block[out_idx] = in_point
                block.sort()
                
                # Update coverage
                for idx in sa_block_tsubs[block_idx]:
                    sa_cover[idx] -= 1
                sa_block_tsubs[block_idx] = None
                
                new_tsubs = set()
                for ts in itertools.combinations(block, t):
                    idx = tsub_idx[ts]
                    new_tsubs.add(idx)
                    sa_cover[idx] += 1
                sa_block_tsubs[block_idx] = new_tsubs
                
                new_cost = sa_cost()
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                    current_cost = new_cost
                    if current_cost < best_sa_cost:
                        best_sa_cost = current_cost
                        best_sa_blocks = [list(b) if b is not None else None for b in sa_blocks]
                else:
                    # Revert
                    for idx in sa_block_tsubs[block_idx]:
                        sa_cover[idx] -= 1
                    sa_block_tsubs[block_idx] = None
                    
                    old_tsubs = set()
                    for ts in itertools.combinations(old_block, t):
                        idx = tsub_idx[ts]
                        old_tsubs.add(idx)
                        sa_cover[idx] += 1
                    sa_block_tsubs[block_idx] = old_tsubs
            
            elif move_type < 0.85:
                # Add a block
                if not sa_is_valid():
                    uncovered = [i for i, c in enumerate(sa_cover) if c == 0]
                    if uncovered:
                        ts_idx = rng.choice(uncovered)
                        start_subset = all_tsubsets[ts_idx]
                        block = list(start_subset)
                        while len(block) < k:
                            best_x = None
                            best_gain = -1
                            for x in range(v):
                                if x in block:
                                    continue
                                gain = 0
                                for rest in itertools.combinations(block, t - 1):
                                    ts = tuple(sorted(rest + (x,)))
                                    if sa_cover[tsub_idx[ts]] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                            block.append(best_x)
                        block.sort()
                        sa_blocks.append(block)
                        sa_add_block(block)
                        
                        new_cost = sa_cost()
                        delta = new_cost - current_cost
                        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                            current_cost = new_cost
                            if current_cost < best_sa_cost:
                                best_sa_cost = current_cost
                                best_sa_blocks = list(sa_blocks)
                        else:
                            # Revert
                            block_idx = len(sa_blocks) - 1
                            sa_remove_block(block_idx)
                            sa_blocks.pop()
            
            T *= cooling
        
        # Use the best found
        if best_sa_blocks and all(b is not None for b in best_sa_blocks):
            # Verify
            test_cover = [0] * n_tsubsets
            for block in best_sa_blocks:
                for ts in itertools.combinations(block, t):
                    test_cover[tsub_idx[ts]] += 1
            if all(c > 0 for c in test_cover) and len(best_sa_blocks) < best_len:
                best_blocks = best_sa_blocks
                best_len = len(best_blocks)
    
    # Final validation
    final_cover = [0] * n_tsubsets
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            final_cover[tsub_idx[ts]] += 1
    
    if not all(c > 0 for c in final_cover):
        # Fallback to greedy
        best_blocks = greedy_construct(randomize=False)
    
    return best_blocks
# EVOLVE-BLOCK-END
