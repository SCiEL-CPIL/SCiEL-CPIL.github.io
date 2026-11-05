import itertools
import random
import time

def construct(v, k, t):
    """
    Covering design C(v, k, t) via Max-SAT / WalkSAT-style stochastic local search.
    
    Each block from a pre-generated candidate pool is a Boolean variable.
    Each t-subset is a clause that must be satisfied.
    We use a soft Max-SAT objective: minimize sum of selected blocks + large penalty for uncovered triples.
    
    The approach:
    1. Generate a pool of candidate blocks via randomized greedy + diversity sampling.
    2. Initialize with a valid greedy cover.
    3. Run WalkSAT-style search: flip variables (add/remove blocks) to minimize cost.
       Cost = num_selected_blocks + BIG_PENALTY * num_uncovered_triples.
    4. Periodically try to remove redundant blocks (greedy deletion).
    5. Return best valid solution found.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin
    
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def get_block_mask(block):
        """Return bitset of t-subset indices covered by this block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tsub_to_idx[ts])
        return mask
    
    # --- Step 1: Generate candidate pool of blocks ---
    # We need enough diverse blocks for the search to work.
    # Generate ~2000-4000 candidate blocks.
    
    # First, a deterministic greedy construction to get an initial valid cover
    # This also gives us some "good" blocks for the pool.
    
    def greedy_init():
        """Standard greedy: repeatedly pick the block covering the most uncovered t-subsets."""
        uncovered = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered:
            best_gain = 0
            best_block = None
            
            # Try a structured approach: sample blocks and also try some deterministic candidates
            candidates = set()
            
            # Random sampling
            for _ in range(3000):
                candidates.add(tuple(sorted(random.sample(range(v), k))))
            
            # Also add blocks based on each uncovered triple
            unc_triples = []
            m = uncovered
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                unc_triples.append(all_tsubs[idx])
                m ^= lsb
            
            for ts in unc_triples[:200]:  # Limit to 200 for speed
                remaining = [x for x in range(v) if x not in ts]
                for _ in range(10):
                    extra = random.sample(remaining, k - t)
                    candidates.add(tuple(sorted(list(ts) + extra)))
            
            # Evaluate candidates
            for cand in candidates:
                mask = get_block_mask(cand)
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = cand
                if gain >= 15:  # Good enough, break early
                    break
            
            if best_block is None:
                best_block = tuple(sorted(random.sample(range(v), k)))
            
            blocks.append(best_block)
            uncovered &= ~get_block_mask(best_block)
            
        return blocks
    
    # Generate initial greedy cover (with time limit)
    init_start = time.time()
    greedy_blocks = greedy_init()
    num_greedy = len(greedy_blocks)
    # print(f"Initial greedy: {num_greedy} blocks")
    
    # Build candidate pool: include greedy blocks + many random/diverse blocks
    pool = list(set(greedy_blocks))
    pool_set = set(pool)
    
    # Add more diverse random blocks
    target_pool_size = min(3000, len(list(itertools.combinations(range(v), k))))
    num_added = 0
    attempts = 0
    while len(pool) < target_pool_size and attempts < target_pool_size * 3:
        attempts += 1
        b = tuple(sorted(random.sample(range(v), k)))
        if b not in pool_set:
            pool.append(b)
            pool_set.add(b)
    
    # Also add blocks specifically designed to cover random triples
    for _ in range(500):
        ts = random.choice(all_tsubs)
        remaining = [x for x in range(v) if x not in ts]
        extra = random.sample(remaining, k - t)
        b = tuple(sorted(list(ts) + extra))
        if b not in pool_set:
            pool.append(b)
            pool_set.add(b)
    
    # Pre-compute masks for all pool blocks
    pool_masks = [get_block_mask(b) for b in pool]
    num_pool = len(pool)
    # print(f"Pool size: {num_pool}")
    
    # --- Step 2: Max-SAT / WalkSAT Search ---
    # State: active[] - boolean array for pool blocks
    # active[i] = True means pool[i] is selected
    
    # Initialize: select the greedy blocks (they form a valid cover)
    greedy_block_set = set(greedy_blocks)
    active = [b in greedy_block_set for b in pool]
    
    # Compute initial cost
    # coverage_count[j] = number of active blocks covering t-subset j
    # We use a bitset sum approach for speed
    
    def compute_coverage_count(active):
        """Return list of coverage counts for each t-subset."""
        counts = [0] * num_tsubs
        for i in range(num_pool):
            if active[i]:
                mask = pool_masks[i]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
        return counts
    
    def compute_cost(active, counts):
        """
        Cost = num_selected + BIG_PENALTY * num_uncovered
        Uncovered = number of t-subs with count == 0
        """
        num_selected = sum(1 for a in active if a)
        uncovered = sum(1 for c in counts if c == 0)
        return num_selected + 10000 * uncovered
    
    BIG_PENALTY = 10000
    
    # Initialize coverage counts
    cur_counts = compute_coverage_count(active)
    cur_cost = compute_cost(active, cur_counts)
    best_cost = cur_cost
    best_active = active[:]
    
    # WalkSAT parameters
    num_flips = 500000  # Total flips to attempt
    restart_every = 5000
    flip_p = 0.01  # Probability of picking a variable that doesn't directly improve cost
    
    # Helper: get affected t-subsets for a block
    def get_block_tsub_indices(block_idx):
        mask = pool_masks[block_idx]
        indices = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        return indices
    
    # Pre-compute t-subset indices for each pool block
    pool_tsub_indices = [get_block_tsub_indices(i) for i in range(num_pool)]
    
    # WalkSAT main loop
    elapsed = time.time() - start_time
    flips_done = 0
    no_improve_streak = 0
    max_no_improve = 5000
    
    while flips_done < num_flips and (time.time() - start_time) < deadline:
        # Check for restart
        if flips_done % restart_every == 0 and flips_done > 0:
            # Restart with some noise: flip a few random variables from best state
            # Or restart from best and continue
            # Let's do: with 30% probability, restart from best; otherwise continue
            if random.random() < 0.3:
                active = best_active[:]
                cur_counts = compute_coverage_count(active)
                cur_cost = compute_cost(active, cur_counts)
            # Reset no-improve counter
            no_improve_streak = 0
        
        # Choose a variable to flip
        # Strategy: prefer variables that would improve cost
        # Cost change from flipping block i:
        #   If turning ON: cost changes by +1 - BIG_PENALTY * (num t-subs newly covered that were uncovered)
        #     Actually: for each t-sub in block i with count 0, it becomes 1 (uncovered decreases by 1)
        #     So cost change = +1 - BIG_PENALTY * num_newly_covered
        #   If turning OFF: cost changes by -1 + BIG_PENALTY * (num t-subs that would become uncovered)
        #     For each t-sub in block i with count 1, it becomes 0 (uncovered increases by 1)
        #     So cost change = -1 + BIG_PENALTY * num_becoming_uncovered
        
        # For efficiency, don't evaluate all variables. Sample a few.
        
        # Determine if current state is valid (no uncovered t-subs)
        has_uncovered = any(c == 0 for c in cur_counts)
        
        if has_uncovered:
            # Prioritize turning ON blocks that cover uncovered t-subs
            # Find uncovered t-subs
            uncovered_indices = [i for i, c in enumerate(cur_counts) if c == 0]
            
            # Pick a random uncovered t-sub
            target_tsub = random.choice(uncovered_indices)
            
            # Find blocks in pool that cover this t-sub
            covering_blocks = []
            for i in range(num_pool):
                if not active[i]:
                    if cur_counts[target_tsub] == 0:  # This t-sub is uncovered
                        # Check if block i covers it
                        # Check if target_tsub is in pool_tsub_indices[i]
                        # This is O(num_pool * avg_tsubs_per_block) which is slow
                        # Alternative: use bitsets
                        pass
            
            # Better: for the uncovered t-sub, find pool blocks containing all its elements
            # t-sub = all_tsubs[target_tsub]
            ts_elements = set(all_tsubs[target_tsub])
            
            covering_blocks = []
            for i in range(num_pool):
                if not active[i]:
                    block = pool[i]
                    if ts_elements.issubset(set(block)):
                        covering_blocks.append(i)
            
            if covering_blocks:
                var = random.choice(covering_blocks)
            else:
                # Shouldn't happen if pool is rich enough
                var = random.randint(0, num_pool - 1)
            
        else:
            # State is valid. Prioritize turning OFF blocks (to reduce count)
            # or turning OFF blocks that don't break validity
            # Find blocks that can be turned off (all their t-subs have count > 1)
            
            # Sample some active blocks
            active_indices = [i for i in range(num_pool) if active[i]]
            
            if not active_indices:
                # No blocks selected, add one
                var = random.randint(0, num_pool - 1)
            else:
                # With some probability, try to turn off a block
                if random.random() < 0.6:
                    # Try to find a removable block
                    candidate_removable = []
                    for i in active_indices[:min(200, len(active_indices))]:
                        can_remove = True
                        for ts_idx in pool_tsub_indices[i]:
                            if cur_counts[ts_idx] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            candidate_removable.append(i)
                    
                    if candidate_removable:
                        var = random.choice(candidate_removable)
                    else:
                        # No removable blocks found in sample. Pick a random active block to turn off
                        # (will temporarily create uncovered, but search will fix it)
                        var = random.choice(active_indices)
                else:
                    # Turn on a random inactive block (explore)
                    inactive_indices = [i for i in range(num_pool) if not active[i]]
                    if inactive_indices:
                        var = random.choice(inactive_indices)
                    else:
                        var = random.choice(active_indices)
        
        # Flip the variable
        was_active = active[var]
        active[var] = not was_active
        
        # Update counts incrementally
        ts_indices = pool_tsub_indices[var]
        if not was_active:
            # Turning ON
            for ts_idx in ts_indices:
                cur_counts[ts_idx] += 1
        else:
            # Turning OFF
            for ts_idx in ts_indices:
                cur_counts[ts_idx] -= 1
        
        # Compute new cost
        new_cost = compute_cost(active, cur_counts)
        flips_done += 1
        
        if new_cost < cur_cost:
            # Accept the flip
            cur_cost = new_cost
            no_improve_streak = 0
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_active = active[:]
        else:
            # With some probability, accept worsening moves (to escape local minima)
            # Or revert
            if random.random() < 0.05:
                # Accept anyway (stochastic)
                cur_cost = new_cost
                no_improve_streak += 1
            else:
                # Revert
                active[var] = was_active
                for ts_idx in ts_indices:
                    if not was_active:
                        cur_counts[ts_idx] -= 1
                    else:
                        cur_counts[ts_idx] += 1
                no_improve_streak += 1
        
        # Early termination check
        if no_improve_streak > max_no_improve and flips_done % restart_every == 0:
            # Restart from best
            active = best_active[:]
            cur_counts = compute_coverage_count(active)
            cur_cost = compute_cost(active, cur_cost) if cur_cost < best_cost else best_cost
            cur_cost = compute_cost(active, cur_counts)
            no_improve_streak = 0
    
    # --- Step 3: Post-processing: greedy removal from best state ---
    # Ensure we have a valid solution
    active_final = best_active[:]
    counts_final = compute_coverage_count(active_final)
    
    # Greedy removal: repeatedly try to remove blocks
    changed = True
    while changed:
        changed = False
        for i in range(num_pool - 1, -1, -1):  # Reverse order to remove "later" blocks first
            if not active_final[i]:
                continue
            # Check if block i is removable
            can_remove = True
            for ts_idx in pool_tsub_indices[i]:
                if counts_final[ts_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                active_final[i] = False
                for ts_idx in pool_tsub_indices[i]:
                    counts_final[ts_idx] -= 1
                changed = True
    
    # Verify validity
    if any(c == 0 for c in counts_final):
        # Fallback: use greedy blocks (they were valid)
        # Rebuild from greedy
        active_final = [b in set(greedy_blocks) for b in pool]
        # Apply greedy removal
        counts_final = compute_coverage_count(active_final)
        changed = True
        while changed:
            changed = False
            for i in range(num_pool - 1, -1, -1):
                if not active_final[i]:
                    continue
                can_remove = True
                for ts_idx in pool_tsub_indices[i]:
                    if counts_final[ts_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_final[i] = False
                    for ts_idx in pool_tsub_indices[i]:
                        counts_final[ts_idx] -= 1
                    changed = True
    
    # Build final result
    final_blocks = []
    for i in range(num_pool):
        if active_final[i]:
            final_blocks.append(list(pool[i]))
    
    # Final verification
    verify_counts = [0] * num_tsubs
    for b in final_blocks:
        for ts in itertools.combinations(b, t):
            verify_counts[tsub_to_idx[ts]] += 1
    
    if any(c == 0 for c in verify_counts):
        # Last resort: return greedy blocks
        return [list(b) for b in greedy_blocks]
    
    return final_blocks
