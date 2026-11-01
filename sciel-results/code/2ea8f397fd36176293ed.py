import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all blocks and their triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    num_blocks = len(all_blocks)
    num_triples = len(all_triples)
    
    # Precompute for each block, the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Greedy initialization to get a good starting point
    # We want a valid cover with few blocks
    def greedy_init():
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        blocks_used = []
        in_solution = [False] * num_blocks
        
        # Simple greedy: always pick max gain, random tie break
        # To speed up, we don't update all gains perfectly, just re-scan?
        # No, the previous greedy was okay. Let's use a slightly faster version.
        # We'll just use the standard greedy logic from the seed but optimized.
        
        # Actually, for initialization, we just need *a* valid cover.
        # A simple greedy is sufficient.
        
        # To make it fast, we can just pick random blocks until covered? No, that's too big.
        # Let's use the greedy from the seed but stop early if time is tight.
        
        # Re-implementing a fast greedy:
        # uncovered: set of triple indices
        # For each step, find block with max uncovered triples.
        
        # Optimization: Instead of scanning all 346k blocks every time (which is 76 * 346k ~ 26M ops, ~1s),
        # we can just use a simple heuristic: Pick random blocks, but prioritize those covering many uncovered?
        
        # Let's stick to the standard greedy but ensure it's fast enough for initialization.
        # The seed program did 20 restarts. We only need 1 good init.
        
        # Faster Init:
        # 1. Start with empty set.
        # 2. While uncovered:
        #    Pick a random triple from uncovered.
        #    Pick a random block covering that triple.
        #    Add it.
        #    Update uncovered.
        
        # This "Greedy by Random Triple" is often competitive and much faster to implement/execute.
        
        # Precompute triple_to_blocks for this
        triple_to_blocks = [[] for _ in range(num_triples)]
        for i, ts in enumerate(block_triples):
            for t_idx in ts:
                triple_to_blocks[t_idx].append(i)
                
        uncovered = set(range(num_triples))
        blocks_used = []
        in_sol = [False] * num_blocks
        
        while uncovered:
            # Pick random uncovered triple
            t_idx = random.choice(list(uncovered))
            # Pick random block covering it
            b_idx = random.choice(triple_to_blocks[t_idx])
            
            if not in_sol[b_idx]:
                in_sol[b_idx] = True
                blocks_used.append(b_idx)
                # Remove covered triples
                for tt in block_triples[b_idx]:
                    if tt in uncovered:
                        uncovered.remove(tt)
                        
        return blocks_used

    # Initialize
    random.seed(42) # Fixed seed for reproducibility of init? Or random?
    # Let's try a few greedy inits and pick the best
    best_init = None
    best_init_size = float('inf')
    
    start_time = time.time()
    max_time = 115
    
    # Run a few quick greedy inits
    for _ in range(5):
        sol = greedy_init()
        if len(sol) < best_init_size:
            best_init_size = len(sol)
            best_init = sol
        if time.time() - start_time > 10:
            break
            
    if best_init is None:
        best_init = greedy_init()
        
    # Setup state for Local Search
    current_blocks = list(best_init)
    in_sol = [False] * num_blocks
    for b in current_blocks:
        in_sol[b] = True
        
    triple_counts = [0] * num_triples
    for b in current_blocks:
        for t_idx in block_triples[b]:
            triple_counts[t_idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    num_blocks_curr = len(current_blocks)
    
    # Penalty weight for uncovered triples
    # We want to minimize num_blocks + P * num_uncovered
    # P should be large enough to prioritize validity, but allow trade-offs.
    # If P is too large, we stay in valid region. If too small, we might lose validity.
    # Let's use P = 10.
    P = 10.0
    
    def get_cost(num_blocks, num_uncovered):
        return num_blocks + P * num_uncovered
        
    current_cost = get_cost(num_blocks_curr, num_uncovered)
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    # Move probabilities
    # We want to explore removals heavily if valid, but also swaps/adds to escape local minima.
    
    while time.time() - start_time < max_time:
        # Choose move
        r = random.random()
        
        new_blocks = current_blocks
        new_in_sol = in_sol
        new_triple_counts = triple_counts
        new_num_uncovered = num_uncovered
        new_num_blocks = num_blocks_curr
        
        # We will simulate the move, calculate delta, then apply if accepted.
        # To avoid copying large arrays, we calculate delta directly.
        
        delta_cost = 0.0
        
        if r < 0.5: # Remove
            if num_blocks_curr > 0:
                b_idx = random.choice(current_blocks)
                # Simulate removal
                delta_blocks = -1
                delta_uncovered = 0
                for t_idx in block_triples[b_idx]:
                    if triple_counts[t_idx] == 1:
                        delta_uncovered += 1
                delta_cost = (delta_blocks + P * delta_uncovered)
                
                # Acceptance
                if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                    # Apply
                    current_blocks.remove(b_idx)
                    in_sol[b_idx] = False
                    num_blocks_curr += delta_blocks
                    for t_idx in block_triples[b_idx]:
                        triple_counts[t_idx] -= 1
                        if triple_counts[t_idx] == 0:
                            num_uncovered += 1
                    num_uncovered += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        
        elif r < 0.8: # Add
            # Pick random block not in solution
            # To avoid infinite loop, just pick random and retry few times
            b_idx = random.randint(0, num_blocks - 1)
            if not in_sol[b_idx]:
                delta_blocks = 1
                delta_uncovered = 0
                for t_idx in block_triples[b_idx]:
                    if triple_counts[t_idx] == 0:
                        delta_uncovered -= 1
                delta_cost = (delta_blocks + P * delta_uncovered)
                
                if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                    # Apply
                    current_blocks.append(b_idx)
                    in_sol[b_idx] = True
                    num_blocks_curr += delta_blocks
                    for t_idx in block_triples[b_idx]:
                        triple_counts[t_idx] += 1
                        if triple_counts[t_idx] == 1:
                            num_uncovered -= 1
                    num_uncovered += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        
        else: # Swap
            if num_blocks_curr > 0:
                b_out = random.choice(current_blocks)
                b_in = random.randint(0, num_blocks - 1)
                if b_in != b_out and not in_sol[b_in]:
                    # Simulate: Remove b_out, Add b_in
                    delta_blocks = 0 # -1 + 1
                    delta_uncovered = 0
                    
                    # Effect of removing b_out
                    for t_idx in block_triples[b_out]:
                        if triple_counts[t_idx] == 1:
                            delta_uncovered += 1
                    # Effect of adding b_in
                    for t_idx in block_triples[b_in]:
                        # Note: triple_counts still reflects state BEFORE swap
                        # But b_out and b_in might share triples.
                        # We need to be careful.
                        # Let's just calculate new counts for affected triples?
                        # Or simpler:
                        # 1. Temporarily decrement b_out
                        # 2. Temporarily increment b_in
                        # This is complex to do in-place for delta calculation without modifying state.
                        
                        # Alternative: Calculate new cost directly? No, too slow.
                        
                        # Let's stick to Add/Remove for simplicity and robustness.
                        # Swap is often handled by doing Remove then Add in two steps in SA.
                        # But here we want atomic moves.
                        
                        # Let's implement swap correctly:
                        # Affected triples are union of triples in b_out and b_in.
                        pass # We'll skip complex swap for now, Add/Remove is sufficient for this size.
                        
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Return best valid solution
    # Ensure best_blocks is valid
    # If best_blocks is invalid (num_uncovered > 0 in best state), we might need to repair?
    # But the penalty P should drive it to 0.
    # If it's not 0, we should return the best *valid* one found.
    # We track best_cost which includes penalty.
    # It's possible the global best cost is for an invalid set if P is too small.
    # Let's verify validity of best_blocks.
    
    # Re-verify best_blocks
    # To be safe, if best_blocks has uncovered triples, we should try to repair or return a known valid one.
    # But with P=10, it's very likely valid.
    
    # Convert indices to blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    
    # Final check: if result is empty or invalid, fallback to greedy_init
    # Check validity of result
    covered = [False] * num_triples
    for b in best_blocks:
        for t_idx in block_triples[b]:
            covered[t_idx] = True
    if not all(covered):
        # Fallback to the initial greedy which was valid
        result = [list(all_blocks[i]) for i in best_init]
        
    return result
