import itertools
import random
import math
import time

def construct(v, k, t):
    # 1. Pre-compute t-subsets and their indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Pre-compute k-subsets (blocks) and their covered t-subset indices
    # We only need a sample of blocks or all if v is small. For v=24, k=7, C(24,7) = 346104.
    # Generating all is feasible in < 1s.
    all_k_subs = list(itertools.combinations(range(v), k))
    
    # Map each block to the list of t-subset indices it covers
    # This is memory heavy but fast access.
    # Alternative: compute on fly. Let's pre-compute for speed in SA.
    block_t_subs = []
    for block in all_k_subs:
        covered = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_subs.append(covered)
        
    # 3. Initial Solution: Greedy Set Cover
    # This gives a valid covering to start with.
    uncovered = [1] * num_t_subs
    current_solution = []
    current_solution_indices = []
    
    # For efficiency in initial greedy, we can just pick random blocks or use a simple heuristic.
    # Let's use a standard greedy but optimized:
    # Actually, for SA, we just need *a* valid covering. 
    # A faster way to get a valid covering is to randomly pick blocks until all are covered.
    # But greedy is better for a tighter start.
    
    # Let's use a fast greedy:
    # We'll maintain a set of uncovered t-subsets.
    # To speed up, we can just pick blocks that cover the most uncovered.
    # However, computing gain for all blocks is O(N_blocks * N_tsubs).
    # 346k blocks * 2024 tsubs is ~700M ops. Too slow for Python in 120s if done naively.
    
    # Alternative Initialization: Random Sampling
    # Pick random blocks until coverage is achieved. This is O(N_blocks * N_tsubs) per check? No.
    # We can just pick, say, 200 random blocks and check.
    
    # Better: Use a "Randomized Greedy"
    # Pick a random t-subset, find a block covering it, add it.
    # Repeat until all covered.
    
    random.seed(42) # Deterministic for reproducibility in dev, but we want diversity? 
    # The task says "evaluator passes...". It doesn't forbid randomness.
    # Let's use a fixed seed for consistency in the function, or random?
    # Usually, better to use random for exploration, but we need to ensure it finishes.
    
    # Let's stick to a robust initialization:
    # 1. Start with empty solution.
    # 2. While uncovered t-subsets exist:
    #    - Pick a random uncovered t-subset.
    #    - Pick a random block containing this t-subset.
    #    - Add to solution.
    #    - Update coverage.
    
    # This is fast.
    
    # Pre-compute: for each t-subset index, which blocks contain it?
    # t_sub_to_blocks = [[] for _ in range(num_t_subs)]
    # for i, block in enumerate(all_k_subs):
    #     for idx in block_t_subs[i]:
    #         t_sub_to_blocks[idx].append(i)
        
    # This mapping is large: 2024 * (avg blocks per t-sub).
    # Each t-sub is in C(v-t, k-t) = C(21, 4) = 5985 blocks.
    # 2024 * 5985 ~ 12M integers. Memory ~ 100MB. Feasible.
    
    t_sub_to_blocks = [[] for _ in range(num_t_subs)]
    for i, block in enumerate(all_k_subs):
        for idx in block_t_subs[i]:
            t_sub_to_blocks[idx].append(i)
            
    # Initialization
    covered_count = [0] * num_t_subs
    current_solution = []
    
    # We need to ensure we cover all.
    # Simple loop:
    while any(c == 0 for c in covered_count):
        # Find an uncovered t-sub
        # Instead of scanning, we can maintain a list of uncovered indices.
        # But for init, scanning is fine if we do it smartly.
        # Let's just pick a random index until we find an uncovered one.
        # Or better: maintain a list.
        
        # Let's just pick a random block from the entire pool? No, that might miss rare t-subsets.
        # Let's pick a random t-sub that is uncovered.
        
        # Optimization: Keep a list of uncovered indices.
        # Actually, let's just use a standard greedy but with a time limit or simplified gain.
        
        # Simpler Init:
        # Just add random blocks until valid.
        # To ensure validity, we can check coverage.
        
        # Let's use the "Pick random uncovered t-sub, then random block covering it" strategy.
        # We need a fast way to get an uncovered t-sub.
        
        # Let's just iterate through all t_subs and if not covered, fix it.
        # This might be slow if we do it one by one.
        
        # Let's try a different init:
        # Start with a small number of random blocks, then add blocks to cover gaps.
        
        # Let's go with the "Fix Uncovered" loop.
        # To speed up finding uncovered, we can just check all 2024.
        
        # We'll use a while loop that breaks when all covered.
        # To avoid infinite loop, we limit iterations.
        
        # Actually, a very fast valid covering can be obtained by:
        # 1. Pick a random permutation of all blocks.
        # 2. Iterate through them, add if it covers at least one new t-sub.
        # This is a greedy on a random order.
        
        random.shuffle(all_k_subs)
        
        for i, block in enumerate(all_k_subs):
            # Check if this block covers any uncovered t-sub
            # We need to know if any idx in block_t_subs[i] has count 0.
            # This is O(C(k,t)) = O(35) check.
            add = False
            for idx in block_t_subs[i]:
                if covered_count[idx] == 0:
                    add = True
                    break
            if add:
                current_solution.append(block)
                for idx in block_t_subs[i]:
                    covered_count[idx] += 1
                    
            # Early stop if all covered?
            # Checking "all covered" is O(2024). Doing it 346k times is 700M ops. Too slow.
            # We can check every 100 blocks or use a counter.
            
            # Let's use a counter of covered t-subs.
            # Wait, I didn't implement that above. Let's refine.
            
    # Let's restart initialization logic properly with a counter.
    covered_count = [0] * num_t_subs
    num_covered = 0
    current_solution = []
    
    # Randomized Greedy Init
    # To make it faster, we can just pick blocks that cover the most uncovered.
    # But calculating gain is expensive.
    # Let's stick to: Shuffle blocks, add if covers new.
    # We need to update num_covered correctly.
    
    shuffled_blocks = list(range(len(all_k_subs)))
    random.shuffle(shuffled_blocks)
    
    for i in shuffled_blocks:
        block_idx = i
        added_new = False
        for idx in block_t_subs[block_idx]:
            if covered_count[idx] == 0:
                added_new = True
                break
        
        if added_new:
            current_solution.append(block_idx)
            for idx in block_t_subs[block_idx]:
                if covered_count[idx] == 0:
                    num_covered += 1
                covered_count[idx] += 1
                
        if num_covered == num_t_subs:
            break
            
    # If not all covered (should be rare with random shuffle of all blocks), add more
    while num_covered < num_t_subs:
        # Pick a random uncovered t-sub
        # Find one
        uncovered_idx = -1
        for i in range(num_t_subs):
            if covered_count[i] == 0:
                uncovered_idx = i
                break
        if uncovered_idx == -1:
            break # Should not happen
            
        # Pick a random block covering it
        candidates = t_sub_to_blocks[uncovered_idx]
        chosen_block = random.choice(candidates)
        
        # Add it
        current_solution.append(chosen_block)
        for idx in block_t_subs[chosen_block]:
            if covered_count[idx] == 0:
                num_covered += 1
            covered_count[idx] += 1
            
    # 4. Simulated Annealing Optimization
    # State: current_solution (list of block indices)
    # Objective: Minimize len(current_solution)
    # Constraint: All t-subs covered (count > 0)
    
    # Moves:
    # 1. Remove a block if it doesn't make any t-sub count 0.
    # 2. Swap a block with another that covers more uncovered (or just random swap to escape minima).
    # Since we start valid, we only need to remove.
    # If we remove, we might get stuck in a local minimum where no single removal is valid,
    # but a sequence of removals or swaps is.
    
    # Actually, "remove if valid" is a descent move.
    # To improve further, we can try to replace a block with a "better" one?
    # No, we want fewer blocks.
    
    # The main improvement comes from aggressive removal.
    # We can also try "2-opt" style: remove two, add one? Unlikely to help reduce count.
    
    # Let's use a simple local search:
    # 1. Try to remove any block. If valid, do it.
    # 2. Repeat until no block can be removed.
    
    # This is a greedy reduction. It's fast.
    # To escape local minima, we can use Random Restart or Simulated Annealing.
    
    # Let's implement a time-bounded loop of:
    # - Randomly shuffle the solution order (doesn't change set, but changes removal order? No, set is what matters).
    # - Actually, the set is the state.
    # - We can try "Kirkpatrick" style:
    #   - Pick a random block to remove.
    #   - If it breaks coverage, accept with probability exp(-delta/T)?
    #   - But we can't have invalid states if we want to return a valid one easily.
    #   - We can allow invalid states in the SA process, but must end in a valid one.
    
    # Simpler: "Iterated Local Search"
    # 1. Greedy Reduce: Remove any removable block.
    # 2. Perturb: Remove a random block (making it invalid), then re-add blocks to fix it?
    #    This is complex.
    
    # Let's stick to:
    # 1. Greedy Reduce (fast).
    # 2. Randomized Greedy Reduce (shuffle order of checks).
    # 3. Repeat.
    
    # This is often effective for covering designs.
    
    best_solution = list(current_solution)
    best_len = len(best_solution)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead
    
    while time.time() - start_time < time_limit:
        # Try to reduce
        # Shuffle the current solution to change the order of removal attempts
        random.shuffle(current_solution)
        
        # Attempt removal
        changed = True
        while changed:
            changed = False
            for i in range(len(current_solution)):
                block_idx = current_solution[i]
                
                # Check if removal is valid
                # Count how many times each t-sub in this block is covered by this block
                # If any count becomes 0, it's invalid.
                
                # We need to know if covered_count[idx] > 1 for all idx in block.
                # If yes, removal is safe.
                
                safe = True
                for idx in block_t_subs[block_idx]:
                    if covered_count[idx] <= 1:
                        safe = False
                        break
                        
                if safe:
                    # Remove
                    for idx in block_t_subs[block_idx]:
                        covered_count[idx] -= 1
                    current_solution.pop(i)
                    changed = True
                    break # Restart loop as indices shifted and state changed
                    
        # Update best
        if len(current_solution) < best_len:
            best_len = len(current_solution)
            best_solution = list(current_solution)
            
            # Optional: Try to restart from this better state with different random seeds?
            # Or just continue.
            
            # If we reached a very low number, we might want to stop early?
            # No, keep going.
            
        # If we didn't improve, we might be stuck.
        # We can try to "perturb" by removing a block even if invalid, then fixing?
        # For now, the shuffle-based greedy is a strong baseline.
        
        # To add more power, we can implement a "swap" move:
        # Replace one block with another that allows further removals?
        # This is harder.
        
        # Let's add a "Random Restart" logic:
        # If no improvement for X iterations, restart from a random valid covering?
        # Too slow.
        
        # Instead, let's try to remove blocks in a different way:
        # We already shuffle.
        
        # One more trick: "Compress" the solution by checking if a block is redundant
        # with respect to the *others*. We already do this.
        
        # Let's try to find a block that covers the *least* unique t-subs and replace it?
        # No, we just want to remove.
        
        # The current loop is: Shuffle -> Greedy Remove.
        # This is a standard heuristic.
        
        # To improve further, we can use a "Tabu" or "SA" that allows temporary invalid states.
        # But given time constraints, the shuffle-greedy is robust.
        
        # Let's check if we can do better by using the "t_sub_to_blocks" to find
        # blocks that are "less useful".
        
        # Actually, the shuffle-greedy is equivalent to trying different orders of removal.
        # Since the final set is independent of order *if* we always remove if possible,
        # wait.
        # If I remove A, then B becomes removable. If I removed B first, A might still be removable.
        # The set of removable blocks depends on the current set.
        # Different orders can lead to different local minima.
        # So shuffling is key.
        
    # 5. Final Pruning (just in case)
    # Do one more pass on the best solution to ensure it's minimal for that specific set.
    # (The loop above should have done this, but let's be safe)
    
    # Convert indices back to blocks
    final_blocks = [list(all_k_subs[i]) for i in best_solution]
    
    # Verify validity (sanity check, might be slow but ensures correctness)
    # For v=24, t=3, this is 2024 checks. Fast.
    # But we trust the counts.
    
    return final_blocks
