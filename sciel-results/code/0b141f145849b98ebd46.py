import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (evaluator kills at 90s, but 60s is the task limit stated)
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1
    
    # Helper to calculate cost
    # Priority: 1. No Gaps, 2. Min Peak, 3. Min Sum of Squares (flatness)
    def get_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            # Heavy penalty for gaps
            return 1000000 + gaps * 10000 + np.max(target_sums)
        
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        # We want to minimize peak first. 
        # If peak is 1, sq is 129 (all 1s).
        # If peak is 2, sq is at least 129 + (number of 2s).
        return peak * 10000 + sq

    # Helper to update sums for adding/removing x
    # A is a set of ints
    # sums is a numpy array of size 257
    def apply_add(A, sums, x):
        # Add pairs {a, x} for all a in A, a != x
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        # Add pair {x, x}
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(A, sums, x):
        # Remove pairs {a, x} for all a in A, a != x
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        # Remove pair {x, x}
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    # Greedy Initialization: Build a set that covers the targets
    def greedy_init():
        A = set()
        sums = np.zeros(257, dtype=np.int32)
        
        # We want to cover [128, 256].
        # Start with some small numbers to have flexibility.
        # Let's try to build it by picking elements that cover the most "hard" targets.
        # Targets 128 and 256 are the hardest to cover with low multiplicity?
        # Actually, 256 requires 128+128. So 128 is a good candidate.
        # 128 requires e.g. 0+128, 64+64, etc.
        
        # Strategy: 
        # 1. Add 128 (covers 256 as 128+128, and helps with mid-range if combined with smalls)
        # 2. Add 0 (covers 128 as 0+128)
        # 3. Iteratively add the element that minimizes the cost.
        
        # Let's just use a heuristic start:
        # A set of arithmetic progression often works well.
        # But let's try a greedy approach for better start.
        
        # Candidate pool
        candidates = list(range(N+1))
        
        # Force add 128 and 0 to ensure ends are covered? 
        # 256 = 128+128. If 128 not in A, 256 must be 127+129 (129 out of range), 126+130... 
        # Max element is 128. So 256 MUST be 128+128. Thus 128 MUST be in A.
        # 128 can be 0+128, 1+127, ..., 64+64.
        
        apply_add(A, sums, 128)
        candidates.remove(128)
        
        # Now we need to cover 128.
        # If we add 0, 128 is covered by 0+128.
        # Let's add 0.
        apply_add(A, sums, 0)
        candidates.remove(0)
        
        # Now we have {0, 128}. Sums: 0, 128, 256.
        # Targets 129..255 are uncovered.
        
        # Greedily add elements
        for _ in range(15): # Add ~15 more elements
            best_cand = None
            best_cost = float('inf')
            
            # Sample a subset of candidates to speed up? 
            # With 126 candidates, checking all is 126 * O(|A|) ~ 126 * 17 ~ 2000 ops. Fast.
            for x in candidates:
                # Apply add
                apply_add(A, sums, x)
                c = get_cost(sums)
                # Revert
                apply_remove(A, sums, x)
                
                if c < best_cost:
                    best_cost = c
                    best_cand = x
            
            if best_cand is None:
                break
                
            apply_add(A, sums, best_cand)
            candidates.remove(best_cand)
            
            # If we have no gaps and peak is low, we can stop early
            if best_cost < 20000: # Heuristic: peak <= 1 or 2 with no gaps
                break

        return A, sums

    # Polish Phase: Try to improve the solution via local moves
    def polish(A, sums, max_iters=1000, time_limit=5.0):
        start_t = time.time()
        current_cost = get_cost(sums)
        improved = True
        
        while improved and time.time() - start_t < time_limit and max_iters > 0:
            improved = False
            max_iters -= 1
            
            A_list = list(A)
            random.shuffle(A_list)
            
            # Try removing elements
            for x in A_list:
                if len(A) <= 1: continue
                
                # Check if x is essential (e.g., 128 for 256)
                # We just apply and check cost
                apply_remove(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost <= current_cost:
                    # If cost is lower or equal, accept.
                    # If equal, we prefer smaller set? 
                    # Cost function doesn't explicitly penalize set size, but 
                    # smaller set -> fewer pairs -> potentially lower peak/sq.
                    # However, if we remove an element, we might create gaps.
                    # If new_cost == current_cost and no gaps, it's a win (smaller set).
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    apply_add(A, sums, x)
            
            # Try swapping elements
            if time.time() - start_t >= time_limit: break
            
            for x in A_list:
                if len(A) == 0: break
                # Try swapping x with y
                # We don't want to swap 128 out if it breaks 256 coverage unless y also helps?
                # 256 requires 128+128. If we remove 128, 256 is uncovered unless another pair sums to 256.
                # But max element is 128. So only 128+128 works for 256.
                # So 128 is INVARIANT. It must stay in A.
                if x == 128:
                    continue
                    
                candidates = [y for y in range(N+1) if y not in A and y != 128]
                if not candidates: continue
                
                # Pick a random few candidates to try
                random.shuffle(candidates)
                for y in candidates[:10]:
                    # Apply remove x, add y
                    apply_remove(A, sums, x)
                    apply_add(A, sums, y)
                    
                    new_cost = get_cost(sums)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        x = y # Update x for subsequent swaps? No, we broke out of inner loop logic implicitly by structure. 
                        # Actually, we should break or continue carefully.
                        # Let's just break the candidate loop for this x if we improved, to avoid complex state tracking
                        break
                    else:
                        # Revert
                        apply_remove(A, sums, y)
                        apply_add(A, sums, x)
                        
                    if time.time() - start_t >= time_limit: break
                    
        return A, sums, current_cost

    # Initialize best solution
    best_A = None
    best_cost = float('inf')
    best_sums = None
    
    # 1. Greedy Initialization
    A_greedy, sums_greedy = greedy_init()
    cost_greedy = get_cost(sums_greedy)
    
    if cost_greedy < best_cost:
        best_cost = cost_greedy
        best_A = set(A_greedy)
        best_sums = sums_greedy.copy()
        
    # Polish the greedy result
    A_polished, sums_polished, cost_polished = polish(set(A_greedy), sums_greedy.copy())
    if cost_polished < best_cost:
        best_cost = cost_polished
        best_A = set(A_polished)
        best_sums = sums_polished.copy()

    # 2. Simulated Annealing from multiple starts
    # We will run SA for the remaining time budget
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10: break
        
        # Random initialization
        # Size between 15 and 30
        size = random.randint(15, 30)
        A = set(random.sample(range(N+1), size))
        
        # Ensure 128 is in A? 
        # If 128 is not in A, 256 is uncovered. Cost will be huge.
        # Let's force 128 in A for random starts too to save time on invalid states.
        if 128 not in A:
            A.remove(random.choice(list(A)))
            A.add(128)
            
        sums = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        
        current_cost = get_cost(sums)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(A)
            best_sums = sums.copy()
            
        # Simulated Annealing Parameters
        T = 50.0
        T_min = 0.1
        cooling = 0.99
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Choose move
            move_type = random.random()
            
            if move_type < 0.1:
                # Add
                candidates = [x for x in range(N+1) if x not in A]
                if not candidates: continue
                x = random.choice(candidates)
                
                apply_add(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, x)
                    
            elif move_type < 0.2:
                # Remove
                if len(A) < 10: continue
                x = random.choice(list(A))
                if x == 128: continue # Don't remove 128
                
                apply_remove(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_add(A, sums, x)
                    
            else:
                # Swap
                if len(A) == 0: continue
                x = random.choice(list(A))
                if x == 128:
                    # If we must swap, we can't swap 128 out. 
                    # So we pick a different x to swap.
                    others = [a for a in A if a != 128]
                    if not others: continue
                    x = random.choice(others)
                    
                candidates = [y for y in range(N+1) if y not in A and y != 128]
                if not candidates: continue
                y = random.choice(candidates)
                
                apply_remove(A, sums, x)
                apply_add(A, sums, y)
                
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, y)
                    apply_add(A, sums, x)
            
            T *= cooling
            if T < T_min:
                break
        
        # Polish after SA
        if time.time() < deadline - 5:
            A_pol, sums_pol, cost_pol = polish(set(A), sums.copy(), max_iters=200, time_limit=2.0)
            if cost_pol < best_cost:
                best_cost = cost_pol
                best_A = set(A_pol)
                best_sums = sums_pol.copy()

    if best_A is None:
        # Fallback: Should not happen with greedy init
        return [128, 0, 64, 32, 16, 8, 4, 2, 1]
        
    return sorted(best_A)

