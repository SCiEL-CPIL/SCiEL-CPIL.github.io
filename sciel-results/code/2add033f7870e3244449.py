import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map t-subset to index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def get_block_triples(block):
        """Return list of indices of t-subsets covered by this block."""
        return [tsub_to_idx[tuple(sorted(ts))] for ts in itertools.combinations(block, t)]

    # Phase 1: Greedy Initialization
    # Start with empty coverage
    coverage = [0] * num_tsubs
    blocks = []
    uncovered_count = num_tsubs
    
    # Greedy: repeatedly pick block covering most uncovered triples
    # We can speed this up by not checking all C(v,k) blocks every time,
    # but for v=24, k=7, C(24,7)=346104. 346k * 35 is ~12M ops.
    # Doing this repeatedly might be slow if we do many steps.
    # Better: Randomized Greedy Construction
    random.seed(42)
    
    # Create a random initial covering using a heuristic
    # Pick random points to form blocks, ensuring coverage
    # Simpler: Start with random blocks until covered? No, that's huge.
    # Let's use a standard greedy set cover approximation.
    
    # To avoid the O(C(v,k)) scan in every greedy step, we can use a randomized approach:
    # 1. Maintain a set of uncovered triples.
    # 2. Pick a random uncovered triple.
    # 3. Find a block containing it that covers the most *other* uncovered triples.
    #    We can sample some blocks containing that triple.
    
    uncovered_set = set(range(num_tsubs))
    
    while uncovered_set and (time.time() - start_time < time_limit * 0.2):
        # Pick a random uncovered triple
        target_idx = random.choice(list(uncovered_set))
        target_ts = all_tsubs[target_idx]
        
        # Find blocks containing this triple
        # A block containing target_ts must contain the 3 points of target_ts.
        # The other k-3 points are chosen from the remaining v-3 points.
        remaining_points = [p for p in range(v) if p not in target_ts]
        
        best_block = None
        best_gain = -1
        
        # Sample some combinations of the remaining points
        # If v-3 is large, sampling is better than exhaustive.
        # C(21, 4) = 5985. This is small enough to check all?
        # 5985 * 35 (triples per block) ~ 200k ops per greedy step.
        # If we need ~70 blocks, 70 * 200k = 14M ops. Feasible.
        
        for other_points in itertools.combinations(remaining_points, k - t):
            block = list(target_ts) + list(other_points)
            block.sort()
            
            # Calculate gain: how many uncovered triples does this block cover?
            gain = 0
            b_triples = get_block_triples(block)
            for idx in b_triples:
                if idx in uncovered_set:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
            
        blocks.append(best_block)
        b_triples = get_block_triples(best_block)
        for idx in b_triples:
            coverage[idx] += 1
            if coverage[idx] == 1:
                uncovered_set.remove(idx)
                
    # If greedy didn't finish in time, finish it quickly with random blocks for uncovered
    if uncovered_set:
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            target_ts = all_tsubs[target_idx]
            remaining_points = [p for p in range(v) if p not in target_ts]
            # Just pick random other points
            other_points = random.sample(remaining_points, k - t)
            block = list(target_ts) + other_points
            block.sort()
            blocks.append(block)
            b_triples = get_block_triples(block)
            for idx in b_triples:
                coverage[idx] += 1
                if coverage[idx] == 1:
                    uncovered_set.remove(idx)
                    
    # Ensure we have a valid covering
    # If there are still uncovered triples (shouldn't happen if loop finished), add more
    while uncovered_set:
        target_idx = random.choice(list(uncovered_set))
        target_ts = all_tsubs[target_idx]
        remaining_points = [p for p in range(v) if p not in target_ts]
        other_points = random.sample(remaining_points, k - t)
        block = list(target_ts) + other_points
        block.sort()
        blocks.append(block)
        b_triples = get_block_triples(block)
        for idx in b_triples:
            coverage[idx] += 1
            if coverage[idx] == 1:
                uncovered_set.remove(idx)

    # Phase 2: Simulated Annealing
    # Cost = len(blocks) + penalty * sum(1 for c in coverage if c == 0)
    # We want to minimize cost.
    
    current_blocks = list(blocks)
    current_coverage = list(coverage)
    current_uncovered = sum(1 for c in current_coverage if c == 0)
    current_cost = len(current_blocks) + 100 * current_uncovered
    
    best_blocks = list(current_blocks)
    best_coverage = list(current_coverage)
    best_uncovered = current_uncovered
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time < time_limit):
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to remove a block
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            block = current_blocks[idx]
            b_triples = get_block_triples(block)
            
            # Calculate new cost if we remove it
            # Uncovered triples will increase for those with coverage == 1
            new_uncovered = current_uncovered
            for idx_t in b_triples:
                if current_coverage[idx_t] == 1:
                    new_uncovered += 1
            
            new_len = len(current_blocks) - 1
            new_cost = new_len + 100 * new_uncovered
            
            if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
                # Accept move
                for idx_t in b_triples:
                    current_coverage[idx_t] -= 1
                current_blocks.pop(idx)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                    best_coverage = list(current_coverage)
                    best_uncovered = current_uncovered
                    
        elif move_type < 0.6:
            # Try to add a block
            # Pick a random block or one covering many uncovered
            # Random block is simple
            block = sorted(random.sample(range(v), k))
            b_triples = get_block_triples(block)
            
            new_uncovered = current_uncovered
            for idx_t in b_triples:
                if current_coverage[idx_t] == 0:
                    new_uncovered -= 1
            
            new_len = len(current_blocks) + 1
            new_cost = new_len + 100 * new_uncovered
            
            if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
                for idx_t in b_triples:
                    current_coverage[idx_t] += 1
                current_blocks.append(block)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                    best_coverage = list(current_coverage)
                    best_uncovered = current_uncovered
                    
        else:
            # Try to replace a block
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[idx]
            old_triples = get_block_triples(old_block)
            
            # Generate new block
            # Perturb old block: swap one point
            if k > 1:
                pos = random.randint(0, k - 1)
                old_point = old_block[pos]
                candidates = [p for p in range(v) if p not in old_block]
                new_point = random.choice(candidates)
                new_block = old_block.copy()
                new_block[pos] = new_point
                new_block.sort()
                new_triples = get_block_triples(new_block)
            else:
                continue
            
            # Calculate delta
            # Triples lost: in old, not in new
            # Triples gained: in new, not in old
            
            # Efficient way:
            # 1. Decrement coverage for old_triples
            # 2. Increment coverage for new_triples
            # 3. Count changes in uncovered status
            
            # To avoid modifying global state and rolling back, we calculate delta
            # But since we need to check if coverage hits 0 or goes from 0 to 1,
            # it's safer to simulate on a copy or carefully track.
            
            # Given v=24, k=7, triples per block = 35.
            # We can just simulate the update and rollback if rejected.
            
            # Apply
            for idx_t in old_triples:
                current_coverage[idx_t] -= 1
            for idx_t in new_triples:
                current_coverage[idx_t] += 1
                
            # Recalculate uncovered (expensive but O(num_tsubs) = 2024)
            # Or track delta
            # Let's just recount for simplicity if num_tsubs is small (2024 is small)
            new_uncovered = sum(1 for c in current_coverage if c == 0)
            new_len = len(current_blocks)
            new_cost = new_len + 100 * new_uncovered
            
            if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
                current_blocks[idx] = new_block
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_uncovered == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                    best_coverage = list(current_coverage)
                    best_uncovered = current_uncovered
            else:
                # Rollback
                for idx_t in old_triples:
                    current_coverage[idx_t] += 1
                for idx_t in new_triples:
                    current_coverage[idx_t] -= 1
                # current_uncovered and current_cost remain unchanged in state, 
                # but we calculated them locally. No state change needed.
                
        T *= cooling_rate

    # Return best valid solution found
    # If best_uncovered > 0, we need to repair it
    if best_uncovered > 0:
        # Use greedy to fix
        current_blocks = list(best_blocks)
        current_coverage = list(best_coverage)
        uncovered_set = set(i for i, c in enumerate(current_coverage) if c == 0)
        
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            target_ts = all_tsubs[target_idx]
            remaining_points = [p for p in range(v) if p not in target_ts]
            other_points = random.sample(remaining_points, k - t)
            block = list(target_ts) + other_points
            block.sort()
            current_blocks.append(block)
            b_triples = get_block_triples(block)
            for idx in b_triples:
                current_coverage[idx] += 1
                if current_coverage[idx] == 1:
                    uncovered_set.remove(idx)
                    
        best_blocks = current_blocks

    return best_blocks
