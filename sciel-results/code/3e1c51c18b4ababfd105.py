import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Coverage count for each t-subset
    # We need to know which ones are uncovered.
    # Using a set of uncovered indices is efficient.
    uncovered = set(range(n_t_subs))
    
    best_blocks = None
    best_count = float('inf')
    
    # Random seed for reproducibility within the run, but we want variety across restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Reset state
        current_uncovered = set(uncovered)
        blocks = []
        
        # Greedy construction
        while current_uncovered:
            # Pick a random uncovered t-subset to seed the new block
            # Picking the one with the lowest "degree" (covered by fewest existing blocks) 
            # might be better, but since we start fresh, all are 0.
            # Random selection helps avoid symmetry.
            seed_idx = rng.choice(list(current_uncovered))
            seed_sub = all_t_subs[seed_idx]
            
            block = list(seed_sub)
            
            # Add points to complete the block
            while len(block) < k:
                # Find the point x not in block that covers the most uncovered t-subsets
                best_x = -1
                best_gain = -1
                # We can optimize this by pre-computing gains, but k is small (8)
                # and v is small (19).
                # Number of candidates: v - len(block) <= 19
                # For each candidate x, we check how many new t-subsets it covers.
                # A new t-subset is formed by x + (t-1 elements from block).
                
                # To speed up, we can just iterate candidates.
                candidates = [x for x in range(v) if x not in block]
                
                # Optimization: We can pre-compute which t-subsets contain a given (t-1) subset.
                # But let's stick to direct counting for simplicity first, 
                # and only if it's too slow, we optimize.
                # With v=19, k=8, t=5, the number of iterations per block is small.
                
                for x in candidates:
                    gain = 0
                    # Check all (t-1) subsets of the current block
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(new_sub)
                        if idx is not None and idx in current_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # Early break if we found a perfect gain?
                        # Max possible gain is C(len(block), t-1)
                        # If gain == C(len(block), t-1), we can't do better.
                        # But we still need to pick x.
                
                if best_x == -1:
                    # Should not happen if v > k
                    break
                block.append(best_x)
                
                # Update uncovered set
                for new_sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx.get(tuple(sorted(new_sub)))
                    if idx is not None:
                        current_uncovered.discard(idx)
            
            blocks.append(tuple(sorted(block)))
            
            # Check deadline inside the loop
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we've hit the lower bound, we can stop early
        # Lower bound for C(19,8,5) is 264.
        if best_count <= 264:
            break
            
        # Optional: Local search / Swapping
        # Given time constraints, a simple restart strategy is often effective.
        # We can vary the random seed or the selection strategy.
        # For now, just restart with a new random seed.
        rng.seed(rng.randint(0, 2**32 - 1))
        
    if best_blocks is None:
        # Fallback: return empty or a trivial set if something went wrong
        # But the loop should always produce something.
        return []
        
    # Convert tuples back to lists as per interface
    return [list(b) for b in best_blocks]
