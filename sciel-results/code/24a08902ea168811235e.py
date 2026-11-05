import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map triples to indices for fast lookup
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # For each block, compute the bitset of triples it covers
    # Using Python integers as bitsets is efficient for num_triples ~ 2024
    block_bitsets = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        bitset = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
        block_bitsets[i] = bitset
        
    # Initial solution: Greedy construction
    uncovered = (1 << num_triples) - 1
    selected_indices = []
    selected_set = set()
    
    # Greedy: Pick block with max new coverage
    while uncovered and time.time() < deadline - 30.0:
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected_set:
                continue
            gain = (block_bitsets[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_bitsets[best_idx]
        
    # If we ran out of time during greedy, return what we have
    if time.time() > deadline - 5.0:
        return [all_blocks[i] for i in selected_indices]
    
    # 2. Simulated Annealing
    # State: selected_indices (list of block indices)
    # Objective: Minimize number of blocks subject to full coverage? 
    # Actually, standard SA for covering usually maximizes coverage for fixed size, 
    # or minimizes size. 
    # Let's try minimizing size directly? No, hard to evaluate "size" change in SA steps easily 
    # unless we allow adding/removing.
    
    # Alternative SA strategy: 
    # Fix the number of blocks to be slightly larger than current greedy result, 
    # then try to swap blocks to maximize coverage. Once full coverage is reached, 
    # try to remove blocks (or swap for redundancy) to reduce size.
    
    # Simpler SA: 
    # State: A set of N blocks. 
    # Move: Replace one block with another.
    # Cost: - (number of covered triples) + lambda * (number of blocks)
    # This is tricky because N changes.
    
    # Let's use the "Fixed N" approach which is robust.
    # Start with N = len(selected_indices).
    # Try to find a solution with N blocks that covers all triples.
    # If found, try N-1, N-2, etc.
    
    # Actually, a common effective heuristic for C(v,k,t) is:
    # 1. Greedy to get a good starting set S.
    # 2. Local search: 
    #    a. Try to remove any redundant block.
    #    b. Try to swap a block with another to increase "slack" or coverage.
    
    # The current program does this. The instruction asks for a DIFFERENT strategy.
    # Let's implement a proper Simulated Annealing on the set of blocks.
    
    # We will maintain a list of block indices `current_solution`.
    # We allow moves that:
    # 1. Replace a block in the solution with a new block.
    # 2. (Optional) Add a block (if we want to grow, but we want to shrink).
    # 3. (Optional) Remove a block (if it's redundant).
    
    # To make SA work well, we define an Energy function.
    # E = - (num_covered_triples) + penalty * (num_blocks)
    # We want to minimize E.
    
    # However, managing variable length in SA is messy.
    # Let's stick to a fixed length N = len(selected_indices).
    # We run SA to maximize coverage for this N.
    # If we reach full coverage, we decrement N and restart SA.
    
    current_N = len(selected_indices)
    best_solution = list(selected_indices)
    best_coverage = (1 << num_triples) - 1 if not uncovered else ((~uncovered) & ((1 << num_triples) - 1)).bit_count()
    
    # If greedy already covered everything, we can try to reduce N immediately.
    # Let's structure the main loop as:
    # While time remains:
    #   Try to improve solution with current N
    #   If full coverage achieved, try to reduce N by 1 and see if we can recover coverage
    
    # For SA, we need to evaluate coverage quickly.
    # We can maintain a count array `counts` of length num_triples.
    
    def compute_counts(indices):
        counts = [0] * num_triples
        for idx in indices:
            b = all_blocks[idx]
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts
        
    def get_coverage(counts):
        # Number of triples with count > 0
        cov = 0
        for c in counts:
            if c > 0:
                cov += 1
        return cov

    # Initialize counts for current solution
    counts = compute_counts(selected_indices)
    current_coverage = get_coverage(counts)
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.995
    max_iterations = 500000
    
    # We will try to optimize the current set of size N.
    # If we find a solution with full coverage, we will try to reduce N.
    
    # To allow reducing N, we can have a move that "removes" a block if it doesn't drop coverage.
    # But standard SA keeps size fixed. 
    # Let's use a variable-length SA or a "Swap + Delete" local search with randomization.
    
    # Strategy: 
    # 1. Local Search Loop (Randomized)
    #    - Pick a random block in solution.
    #    - Try to replace it with a random block from all_blocks.
    #    - Accept if coverage increases, or with prob exp(-delta/T) if decreases.
    #    - Also, occasionally try to remove a block if coverage doesn't drop.
    #    - Occasionally try to add a block if it increases coverage (to escape traps).
    
    # This is essentially a Variable Neighborhood Search (VNS) or SA.
    
    # Let's implement a robust SA that allows size changes.
    # State: list of indices.
    # Moves:
    # 1. Swap: Replace index i with random j.
    # 2. Remove: Remove index i (if N > min_size).
    # 3. Add: Add random j (if N < max_size).
    
    # Energy: -coverage + 0.1 * N (penalize size slightly to encourage minimality, 
    # but primarily we want coverage).
    # Actually, if we just maximize coverage, we might end up with too many blocks.
    # We want the fewest blocks.
    
    # Let's define Energy = -coverage. 
    # And we run separate phases for each N?
    # Phase 1: Maximize coverage for N = current_N.
    # Phase 2: If full coverage, set N = N-1, and try to maximize coverage for new N.
    #          If we can reach full coverage with N-1, keep it. Else revert to N.
    
    # This "Decreasing N" strategy is powerful.
    
    # Let's implement the SA for a fixed N.
    
    def run_sa_for_n(n, initial_indices, max_time=5.0):
        """
        Run SA to maximize coverage using exactly n blocks.
        Returns best_indices, best_coverage.
        """
        if n == 0:
            return [], 0
            
        # Start with a random subset of size n from initial_indices or random?
        # Better to start from a good greedy solution of size n if possible.
        # For simplicity, start from initial_indices (truncated or padded if needed? No, must be size n).
        # If initial_indices has different size, we need to adjust.
        # Let's assume we pass in a valid list of size n.
        
        current_indices = list(initial_indices)
        # Ensure size is n
        if len(current_indices) > n:
            # Randomly drop extras
            random.shuffle(current_indices)
            current_indices = current_indices[:n]
        elif len(current_indices) < n:
            # Randomly add from all_blocks
            while len(current_indices) < n:
                idx = random.randint(0, num_blocks - 1)
                if idx not in current_indices:
                    current_indices.append(idx)
        
        # Avoid duplicates in initial state? 
        # The problem requires distinct blocks.
        # Let's enforce distinctness in the state.
        # If duplicates exist, replace one.
        seen = set()
        unique_indices = []
        for idx in current_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        while len(unique_indices) < n:
            idx = random.randint(0, num_blocks - 1)
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        current_indices = unique_indices
        
        counts = compute_counts(current_indices)
        current_coverage = get_coverage(counts)
        
        best_indices = list(current_indices)
        best_coverage = current_coverage
        
        T = 1.0
        cooling = 0.999
        iterations = 0
        max_iters = 200000 # Limit iterations per N phase
        
        while iterations < max_iters and time.time() < max_time:
            iterations += 1
            
            # Pick a random block to swap
            pos = random.randint(0, n - 1)
            old_idx = current_indices[pos]
            
            # Pick a new random block
            new_idx = random.randint(0, num_blocks - 1)
            
            if new_idx == old_idx:
                continue
                
            # Check if new_idx is already in current_indices
            # If so, we'd have a duplicate. 
            # For simplicity in this SA step, let's only allow swaps that don't create duplicates.
            # We can check if new_idx is in current_indices.
            # O(N) check. N is small (~50-100).
            if new_idx in current_indices:
                continue
                
            # Simulate swap
            # Remove old_idx, add new_idx
            # Update counts
            
            # To make this fast, we pre-computed block_bitsets.
            # But updating counts array is O(k*t) = 35 ops.
            # We can do it in place.
            
            # Decrement counts for old_idx
            old_block = all_blocks[old_idx]
            for tri in itertools.combinations(old_block, t):
                tri_idx = triple_to_idx[tri]
                counts[tri_idx] -= 1
                if counts[tri_idx] == 0:
                    current_coverage -= 1
                    
            # Increment counts for new_idx
            new_block = all_blocks[new_idx]
            for tri in itertools.combinations(new_block, t):
                tri_idx = triple_to_idx[tri]
                if counts[tri_idx] == 0:
                    current_coverage += 1
                counts[tri_idx] += 1
            
            # Acceptance
            delta = current_coverage - best_coverage # Not quite, we need delta from previous state
            # Actually, we should compare to the state BEFORE the move.
            # Let's store prev_coverage.
            
            # Refinement: We need prev_coverage.
            # Let's restructure the loop to store prev.
            
            # For now, let's use a simpler acceptance: 
            # If current_coverage > best_coverage, accept and update best.
            # Else, accept with probability exp((current_coverage - prev_coverage) / T).
            
            # I didn't store prev_coverage. Let's restart the loop logic properly.
            pass
        
        # The above loop was a sketch. Let's write the real one.
        
        # Reset
        current_indices = list(best_indices) # Start from best known
        counts = compute_counts(current_indices)
        current_coverage = get_coverage(counts)
        
        T = 5.0
        cooling = 0.999
        max_iters = 50000
        iterations = 0
        
        while iterations < max_iters and time.time() < max_time:
            iterations += 1
            if T < 0.01:
                break
                
            pos = random.randint(0, n - 1)
            old_idx = current_indices[pos]
            new_idx = random.randint(0, num_blocks - 1)
            
            if new_idx == old_idx or new_idx in current_indices:
                continue
                
            # Calculate delta
            # Remove old
            prev_coverage = current_coverage
            old_block = all_blocks[old_idx]
            for tri in itertools.combinations(old_block, t):
                tri_idx = triple_to_idx[tri]
                counts[tri_idx] -= 1
                if counts[tri_idx] == 0:
                    current_coverage -= 1
                    
            # Add new
            new_block = all_blocks[new_idx]
            for tri in itertools.combinations(new_block, t):
                tri_idx = triple_to_idx[tri]
                if counts[tri_idx] == 0:
                    current_coverage += 1
                counts[tri_idx] += 1
            
            delta = current_coverage - prev_coverage
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                # Accept
                current_indices[pos] = new_idx
                if current_coverage > best_coverage:
                    best_coverage = current_coverage
                    best_indices = list(current_indices)
            else:
                # Reject, revert counts
                # Add back old
                for tri in itertools.combinations(old_block, t):
                    tri_idx = triple_to_idx[tri]
                    if counts[tri_idx] == 0:
                        current_coverage += 1
                    counts[tri_idx] += 1
                    
                # Remove new
                for tri in itertools.combinations(new_block, t):
                    tri_idx = triple_to_idx[tri]
                    counts[tri_idx] -= 1
                    if counts[tri_idx] == 0:
                        current_coverage -= 1
                        
            T *= cooling
            
        return best_indices, best_coverage

    # Main Logic:
    # 1. Get greedy solution.
    # 2. Try to improve it with SA for the same size.
    # 3. If full coverage, try smaller sizes.
    
    current_solution = list(selected_indices)
    current_n = len(current_solution)
    
    # Run SA for current_n to see if we can improve coverage (maybe greedy wasn't perfect)
    # Actually, greedy usually gets full coverage quickly.
    # The main benefit of SA is to find a *different* set of size N that might allow removal later?
    # No, removal depends on redundancy.
    
    # Let's focus on the "Decreasing N" strategy.
    # We know we have a solution of size current_n.
    # Let's try to find a solution of size current_n - 1.
    # If we find one, we accept it and try current_n - 2, etc.
    
    # To find a solution of size N-1:
    # We can use SA starting from a subset of the current solution?
    # Or random initialization?
    # Random initialization is better to explore.
    
    # We have time budget.
    # Let's allocate time to try smaller N.
    
    # Time management:
    # We have ~115s.
    # Greedy took some time.
    # Let's reserve 10s for final cleanup/verification.
    # Remaining time for SA.
    
    # We will try N from current_n down to 1.
    # For each N, we run SA for a fixed time slice.
    # If we find full coverage, we update current_solution and continue to N-1.
    
    # How to split time?
    # Give each N attempt a fraction of remaining time.
    # Or give it a fixed number of iterations.
    
    # Let's give each N attempt 2 seconds.
    # If we have 100s, we can try ~50 sizes.
    # Current N is likely ~76.
    # We can try 75, 74, ... down to 69.
    
    # If we find a solution for N, we use it as the "initial" for N-1?
    # No, N-1 is a different problem.
    # We can use a random subset of the N-solution as a start for N-1?
    # Or just random.
    
    # Let's try:
    # for target_n in range(current_n - 1, 0, -1):
    #    run SA for target_n with time limit.
    #    if full coverage:
    #       update best_solution
    #       break? No, try to go lower.
    
    # But we need to be careful: if we fail to find coverage for N-1, we should keep the N-solution.
    
    # Let's implement this loop.
    
    best_known_solution = list(current_solution)
    best_known_n = current_n
    full_covered = (current_coverage == num_triples)
    
    if not full_covered:
        # Greedy didn't cover all? Unlikely for 76 blocks.
        # But if it did, we need to fix it first.
        # Run SA for current_n to maximize coverage.
        idxs, cov = run_sa_for_n(current_n, current_solution, max_time=10.0)
        if cov == num_triples:
            best_known_solution = idxs
            best_known_n = current_n
            full_covered = True
    
    if full_covered:
        # Try to reduce size
        # Time remaining
        remaining_time = deadline - time.time() - 5.0 # 5s buffer
        
        # Estimate time per attempt
        # Let's try 10 attempts of 1.5s each?
        # Or just loop until time runs out, trying decreasing N.
        
        # To be efficient, we can try N-1, then N-2, etc.
        # If we find a solution for N-1, we set current_n = N-1 and continue.
        
        # If we don't find a solution for N-1, do we try N-2?
        # It's possible that a random solution of size N-2 exists even if we failed to find one for N-1.
        # But it's less likely.
        # Let's stick to sequential: if we fail N-1, we stop?
        # Or we can try N-1 multiple times with different random seeds.
        
        # Let's do:
        # For target_n in range(best_known_n - 1, 0, -1):
        #    Allocate time slice.
        #    Run SA.
        #    If success: update best, break (or continue to next lower).
        #    If fail: continue to next lower? 
        #    Actually, if we can't cover with N-1, we can't cover with N-2 (since N-2 < N-1).
        #    Wait, is that true? 
        #    If there is no covering of size N-1, there is no covering of size N-2.
        #    So if we are confident we failed N-1, we can stop.
        #    But we might have failed due to time limit, not non-existence.
        #    So we should keep trying N-1 with more time?
        #    Or try N-2? No, N-2 is harder.
        
        # Strategy: 
        # Focus all remaining time on finding a solution of size best_known_n - 1.
        # If we find it, update and try best_known_n - 2.
        # If we run out of time, return best_known_solution.
        
        target_n = best_known_n - 1
        while target_n > 0 and time.time() < deadline - 5.0:
            # Run SA for target_n
            # Use random initialization
            random_indices = random.sample(range(num_blocks), target_n)
            
            # Run SA
            idxs, cov = run_sa_for_n(target_n, random_indices, max_time=min(5.0, deadline - time.time() - 5.0))
            
            if cov == num_triples:
                best_known_solution = idxs
                best_known_n = target_n
                target_n -= 1
                # If we found one, we can try to go lower
            else:
                # Failed to find covering for target_n
                # Should we try again with more time?
                # Or move to target_n - 1? (No, impossible if target_n is impossible)
                # So we should probably just break or keep trying target_n.
                # Let's keep trying target_n for a bit longer, then break.
                # Actually, if we used 5s and failed, maybe it's just hard.
                # Let's try one more time with a different seed?
                # To keep it simple, if we fail, we break.
                # This assumes that if we can't find it quickly, it's not there.
                # This is a heuristic risk.
                break
                
    return [all_blocks[i] for i in best_known_solution]
