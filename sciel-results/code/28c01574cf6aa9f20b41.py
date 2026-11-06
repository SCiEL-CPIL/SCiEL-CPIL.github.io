import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def calc_score(mask):
        """
        Count distinct d > 0 such that there exists a with a, a+d, a+2d in the set.
        Uses bitmask representation where bit i is set if i is in the set.
        d ranges from 1 to 255 (since max value is 512, 2d <= 511 => d <= 255).
        """
        s = 0
        m = mask
        # We can optimize by checking up to 255
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    best_mask = 0
    best_score = -1

    def update_best(mask, score):
        nonlocal best_mask, best_score
        if score > best_score:
            best_score = score
            best_mask = mask

    def greedy_construct(iterations=10):
        """
        Run greedy set cover heuristic multiple times with random tie-breaking
        to generate diverse high-quality starting points.
        """
        best_local_mask = 0
        best_local_score = -1
        
        for _ in range(iterations):
            mask = 0
            # To speed up greedy, we need to track which differences are already covered.
            # However, recalculating score for each candidate is O(255 * 512 * 32) which is ~4M ops per greedy step?
            # No, greedy step: for each of 512 candidates, calc_score(mask | bit) takes 255 ops.
            # 512 * 255 = ~130k ops per element selection.
            # 32 elements => ~4M ops per greedy run.
            # 10 runs => 40M ops. This is very fast in Python if optimized, but calc_score is a loop.
            # Let's optimize the inner loop of greedy by tracking covered differences.
            
            covered_diffs = [False] * 256 # 1..255
            
            for k in range(32):
                best_candidate = -1
                best_new_count = -1
                best_candidate_mask = 0
                
                # To add randomness, we can shuffle candidates or just pick best with random tie break
                # But we need to evaluate all to find the max gain.
                # Optimization: We can iterate candidates.
                
                for cand in range(1, 513):
                    if mask & (1 << cand):
                        continue
                    
                    # Calculate new differences covered by adding cand
                    # A difference d is covered if there exists a such that a, a+d, a+2d in (mask U {cand}).
                    # This is complex to incrementally update without full recalculation or clever bookkeeping.
                    # Given 512 candidates and 255 diffs, full recalculation of score for each candidate
                    # is 512 * 255 = 130k bit ops.
                    # 32 steps * 130k = 4.1M bit ops per greedy run.
                    # Python big integer bit ops are fast, but the loop overhead is significant.
                    # 10 runs * 32 * 512 * 255 ~ 40 million iterations of the inner loop.
                    # This might take 10-20 seconds in Python. Acceptable.
                    
                    # Let's just use the bitmask score for simplicity and reliability, 
                    # but we can prune: if current max possible gain is small, skip? No.
                    
                    new_mask = mask | (1 << cand)
                    # We need to count how many NEW diffs are covered.
                    # Since we don't track which diffs are covered, we just compare total score?
                    # No, greedy set cover maximizes *marginal gain*.
                    # Gain = score(new_mask) - score(mask).
                    # Since score(mask) is constant for this step, we just maximize score(new_mask).
                    
                    # Wait, score(new_mask) is not just "new diffs", it's total diffs.
                    # Since mask is fixed, maximizing score(new_mask) is equivalent to maximizing new diffs.
                    
                    s = calc_score(new_mask)
                    
                    if s > best_new_count or (s == best_new_count and random.random() < 0.5):
                        best_new_count = s
                        best_candidate = cand
                        best_candidate_mask = new_mask

                if best_candidate == -1:
                    # No candidates left? Should not happen until 32 selected.
                    break
                
                mask = best_candidate_mask

            local_score = calc_score(mask)
            if local_score > best_local_score:
                best_local_score = local_score
                best_local_mask = mask

        return best_local_mask, best_local_score

    # Phase 1: Greedy Construction
    # Run a few greedy instances to find a good starting point
    greedy_mask, greedy_score = greedy_construct(iterations=5)
    update_best(greedy_mask, greedy_score)
    
    # Phase 2: Simulated Annealing / Local Search from the best greedy solution
    # We can also try random starts if time permits, but greedy is usually much better.
    
    current_mask = best_mask
    current_score = best_score
    
    T = 1.0
    cool_rate = 0.9995
    min_T = 0.01
    
    # We can also mix in some random restarts if we have time, 
    # but let's stick to one long SA from the best greedy result first.
    
    # To escape local optima, we might want to occasionally "jump" by randomizing a few bits.
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = 0.5 # Reheat? Or just stop? Let's reheat slightly to explore.
            # Actually, with 140s, we can run many steps.
            # Let's just keep cooling. If we get stuck, we can restart SA with a perturbed mask.
            
        # Generate move: Swap a random element with a random non-element
        x = random.randint(1, 512)
        y = random.randint(1, 512)
        
        if x == y:
            continue
            
        bx = 1 << x
        by = 1 << y
        
        # Check if x is in mask and y is not
        if (current_mask & bx) and not (current_mask & by):
            new_mask = current_mask ^ bx ^ by
            new_score = calc_score(new_mask)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                update_best(current_mask, current_score)
        
        T *= cool_rate

    # Convert best_mask to list
    result = []
    for i in range(1, 513):
        if best_mask & (1 << i):
            result.append(i)
    
    # Verify length
    if len(result) != 32:
        # Fallback
        return list(range(1, 33))
        
    return result
