import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    # Map each t-subset to an index for fast lookup
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Estimate minimum number of blocks needed (Lower Bound)
    # LB = ceil( C(v,t) / C(k,t) )
    lb = math.ceil(math.comb(v, t) / math.comb(k, t))
    
    best_blocks = None
    best_size = float('inf')
    
    # We will try to find a covering with N blocks for various N
    # Start from a reasonable estimate, e.g., lb + 10, up to some max
    # The best known is 151. LB is 131.
    # Let's try N from lb to lb + 30
    
    for N in range(lb, lb + 30):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Run Simulated Annealing for this N
        blocks, success = simulated_annealing_cover(v, k, t, N, all_t_subs, t_sub_index, start_time, time_limit)
        
        if success and blocks is not None:
            # Try to reduce the number of blocks
            reduced_blocks = reduce_blocks(blocks, v, k, t, all_t_subs, t_sub_index)
            if reduced_blocks is not None:
                size = len(reduced_blocks)
                if size < best_size:
                    best_size = size
                    best_blocks = reduced_blocks
                # If we found a very good one, we might stop or continue to see if we can do better
                # Given time constraints, if we find <= 151, it's excellent.
                if best_size <= 151:
                    # Check if we can improve further with remaining time
                    if time.time() - start_time > time_limit - 20:
                        break
                    # Continue trying smaller N? No, we are iterating N upwards.
                    # But we started at lb. If we found a solution at N, we should check if a smaller N works?
                    # We are iterating N from lb upwards. So if we find a solution at N, it's the first (smallest) N that worked in this pass?
                    # Not necessarily, SA might fail for small N and succeed for larger.
                    # But if we find a solution at N, we should record it.
                    # We can break if we are confident no smaller N works, but that's hard to know.
                    # However, the loop goes from lb upwards. If we find a solution at N=140, we record it.
                    # We continue to N=141? No, we want the MINIMUM blocks.
                    # So if we find a solution at N, we should try to see if we can find one at N-1?
                    # The loop structure `for N in range(lb, ...)` tries increasing N.
                    # This is counter-productive for minimizing blocks if we just take the first success.
                    # We should iterate N from high to low? Or just run SA for a fixed N and see if it covers.
                    # Better: Iterate N from lb upwards. If SA finds a full cover for N, that's a candidate.
                    # Since we want min blocks, we should stop at the first N where SA finds a full cover?
                    # Yes, if we are confident SA is good enough to find a cover if one exists.
                    # If SA fails for N, it might still be possible. But if it succeeds, N is a valid upper bound.
                    # So we can break on first success?
                    # Risk: SA might fail for N=132 (which is possible) but succeed for N=135.
                    # Then we return 135. But maybe 132 is possible.
                    # To be safe, we can run SA for a few N values and keep the best.
                    # But time is tight.
                    # Let's assume if SA finds a cover for N, it's likely near optimal for that N.
                    # We will break on first success to save time, hoping lb is close.
                    break
            else:
                # Reduction failed, use the original N blocks
                size = N
                if size < best_size:
                    best_size = size
                    best_blocks = blocks
                break # Same logic, first success
    
    if best_blocks is None:
        # Fallback to greedy if SA fails completely
        return greedy_fallback(v, k, t, start_time, time_limit)
        
    return best_blocks

def simulated_annealing_cover(v, k, t, N, all_t_subs, t_sub_index, start_time, time_limit):
    """
    Attempts to find N blocks that cover all t-subsets using Simulated Annealing.
    Returns (blocks, success).
    """
    rng = random.Random(42)
    
    # Initialize random blocks
    blocks = []
    for _ in range(N):
        block = set(rng.sample(range(v), k))
        blocks.append(block)
    
    # Calculate initial coverage
    covered = [0] * len(all_t_subs)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            covered[idx] += 1
            
    num_covered = sum(1 for c in covered if c > 0)
    target_covered = len(all_t_subs)
    
    if num_covered == target_covered:
        return [sorted(b) for b in blocks], True
        
    # SA Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # Precompute all possible points
    points = list(range(v))
    
    while T > min_T:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Random Move
        # Option 1: Swap a point in a random block
        # Option 2: Replace a random block with a new random block
        
        # Swap is usually more effective for local search
        block_idx = rng.randint(0, N - 1)
        block = blocks[block_idx]
        
        # Pick a point to remove
        remove_point = rng.choice(list(block))
        # Pick a point to add
        add_point = rng.choice([p for p in points if p not in block])
        
        # Calculate change in coverage
        # Remove remove_point from block, add add_point
        old_block = block.copy()
        new_block = block.copy()
        new_block.remove(remove_point)
        new_block.add(add_point)
        
        # Determine which t-subsets are affected
        # They are those containing remove_point and (t-1) from old_block\{remove_point}
        # OR those containing add_point and (t-1) from new_block\{add_point}
        
        # Efficient way:
        # Count change for each affected t-subset
        
        delta_covered = 0
        
        # Subsets involving remove_point that are lost
        rest_old = list(old_block)
        rest_old.remove(remove_point)
        for rest in itertools.combinations(rest_old, t - 1):
            t_sub = tuple(sorted(rest + (remove_point,)))
            idx = t_sub_index[t_sub]
            if covered[idx] == 1:
                delta_covered -= 1 # Becomes uncovered
            
        # Subsets involving add_point that are gained
        rest_new = list(new_block)
        rest_new.remove(add_point)
        for rest in itertools.combinations(rest_new, t - 1):
            t_sub = tuple(sorted(rest + (add_point,)))
            idx = t_sub_index[t_sub]
            if covered[idx] == 0:
                delta_covered += 1 # Becomes covered
            
        # Accept/Reject
        if delta_covered > 0:
            # Improvement
            blocks[block_idx] = new_block
            # Update covered counts
            # Actually, we need to update the `covered` array
            # We lost the ones from remove_point, gained the ones from add_point
            # Let's just recalculate the affected ones properly to keep `covered` accurate
            
            # This is getting complex to update incrementally without errors.
            # Given N is small (~150), and k=7, t=4.
            # Affected subsets per move: C(6,3) + C(6,3) = 20 + 20 = 40 subsets.
            # Updating 40 entries in an array is fast.
            
            # Update `covered` for remove_point subsets (decrement)
            for rest in itertools.combinations(rest_old, t - 1):
                t_sub = tuple(sorted(rest + (remove_point,)))
                idx = t_sub_index[t_sub]
                covered[idx] -= 1
                
            # Update `covered` for add_point subsets (increment)
            for rest in itertools.combinations(rest_new, t - 1):
                t_sub = tuple(sorted(rest + (add_point,)))
                idx = t_sub_index[t_sub]
                covered[idx] += 1
                
            num_covered += delta_covered
            
        else:
            # Worse or same
            # Accept with probability exp(delta / T)
            # delta is negative or 0
            if delta_covered == 0:
                # Neutral, accept to explore
                blocks[block_idx] = new_block
                # Update covered
                for rest in itertools.combinations(rest_old, t - 1):
                    t_sub = tuple(sorted(rest + (remove_point,)))
                    idx = t_sub_index[t_sub]
                    covered[idx] -= 1
                for rest in itertools.combinations(rest_new, t - 1):
                    t_sub = tuple(sorted(rest + (add_point,)))
                    idx = t_sub_index[t_sub]
                    covered[idx] += 1
            else:
                prob = math.exp(delta_covered / T)
                if rng.random() < prob:
                    blocks[block_idx] = new_block
                    # Update covered
                    for rest in itertools.combinations(rest_old, t - 1):
                        t_sub = tuple(sorted(rest + (remove_point,)))
                        idx = t_sub_index[t_sub]
                        covered[idx] -= 1
                    for rest in itertools.combinations(rest_new, t - 1):
                        t_sub = tuple(sorted(rest + (add_point,)))
                        idx = t_sub_index[t_sub]
                        covered[idx] += 1
                    num_covered += delta_covered
                    
        T *= cooling_rate
        
        if num_covered == target_covered:
            return [sorted(b) for b in blocks], True
            
    return [sorted(b) for b in blocks], False

def reduce_blocks(blocks, v, k, t, all_t_subs, t_sub_index):
    """
    Try to remove redundant blocks from a valid covering.
    """
    # Count how many blocks cover each t-subset
    covered_count = [0] * len(all_t_subs)
    
    # Map block index to the t-subsets it covers
    block_t_subs = []
    for i, block in enumerate(blocks):
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            covered_count[idx] += 1
            subs.append(idx)
        block_t_subs.append(subs)
        
    # Check if valid
    if any(c == 0 for c in covered_count):
        return None
        
    # Try to remove blocks
    # Sort blocks by "uniqueness" (number of t-subsets they are the sole cover for)
    # Or just try removing each block and checking validity.
    
    # Simple heuristic: Try removing each block one by one.
    # If removal leaves all t-subsets covered, keep it removed.
    # Repeat until no more blocks can be removed.
    
    current_blocks = blocks[:]
    current_t_subs = block_t_subs[:]
    current_counts = covered_count[:]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks)):
            # Check if block i is redundant
            redundant = True
            for idx in current_t_subs[i]:
                if current_counts[idx] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove block i
                for idx in current_t_subs[i]:
                    current_counts[idx] -= 1
                    
                current_blocks.pop(i)
                current_t_subs.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return current_blocks

def greedy_fallback(v, k, t, start_time, time_limit):
    # Simple greedy as fallback
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() - start_time < time_limit - 2:
        # Pick random uncovered
        start_subset = random.choice(list(uncovered))
        block = list(start_subset)
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            best_x = max(candidates, key=lambda x: sum(1 for rest in itertools.combinations(block, t-1) if tuple(sorted(rest + (x,))) in uncovered) if len(block) >= t-1 else 0)
            block.append(best_x)
        block.sort()
        blocks.append(block)
        for t_sub in itertools.combinations(block, t):
            uncovered.discard(t_sub)
    return blocks
