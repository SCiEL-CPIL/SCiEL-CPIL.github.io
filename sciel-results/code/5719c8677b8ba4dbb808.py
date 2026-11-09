import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # 10s safety margin
    
    points = list(range(v))
    
    # Precompute all blocks and t-subsets
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    
    # Map t-subsets to indices
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Build incidence matrix: M[block_idx, t_idx] = 1 if t-subset is in block
    # This is sparse, but for 75k x 11k it might be manageable with dense if we are careful, 
    # or we can use sparse. However, numpy dense operations on 75kx11k (825M elements) 
    # would be ~6.6GB of memory if float, or 825MB if bool. 
    # Let's try to avoid full dense matrix if possible, or use it if memory allows.
    # 75582 * 11628 * 1 byte = ~878 MB. This is risky but might fit in typical eval environments.
    # Alternatively, we can keep the list of lists but use numpy for the 'gain' calculation.
    
    # Let's use a hybrid approach: 
    # 1. Precompute for each block, the list of t-subset indices it covers.
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(np.array(subs, dtype=np.int32))
    
    # 2. Precompute for each t-subset, the list of blocks that cover it.
    # This is needed for the local search (replacement) and for updating gains efficiently.
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert tsub_blocks to numpy arrays for faster iteration? 
    # No, lists are fine for small counts, but let's check average degree.
    # Each block covers C(8,5)=56 t-subsets.
    # Total incidences = 75582 * 56 = 4,232,592.
    # Avg blocks per t-subset = 4,232,592 / 11,628 = 364.
    # So each t-subset is covered by ~364 blocks.
    
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # Initialize gains
        # gain[i] = number of uncovered t-subsets covered by block i
        # Initially, all are uncovered, so gain[i] = len(block_tsubs[i]) = 56
        gains = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        
        is_covered = np.zeros(N_t, dtype=np.bool_)
        selected_blocks = []
        uncovered_count = N_t
        
        while uncovered_count > 0:
            if time.time() > DEADLINE:
                break
            
            # Find block with max gain
            # If there are ties, we can pick randomly among the max gain blocks to diversify
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find indices with max gain
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1 and random_seed is not None:
                # Random tie-breaking
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected_blocks.append(all_blocks[best_idx])
            
            # Update gains for all blocks that share t-subsets with the selected block
            # For each t-subset in the selected block, if it was uncovered:
            #   - Mark it as covered
            #   - Decrement gain for all blocks that cover this t-subset
            for si in block_tsubs[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    # Update gains for all blocks covering si
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        """
        Local search: Try to replace a set of blocks with a smaller set.
        Specifically, try to find if any block can be removed (if its unique coverage is 0).
        Then try to replace pairs of blocks with a single block that covers their union.
        """
        if not blocks:
            return blocks
            
        n = len(blocks)
        if n == 0: return blocks
        
        # Map block index in 'blocks' to original block id? 
        # It's easier to work with the list of blocks directly and recompute coverage counts.
        
        # 1. Pass 1: Remove redundant blocks (those with no unique t-subsets)
        # Calculate coverage count for each t-subset by the current set of blocks
        cov_count = np.zeros(N_t, dtype=np.int32)
        block_tsubs_list = []
        for b in blocks:
            subs = []
            for s in itertools.combinations(b, t):
                subs.append(t_idx[s])
            block_tsubs_list.append(subs)
            for si in subs:
                cov_count[si] += 1
                
        removed = [False] * n
        changed = True
        while changed:
            if time.time() > DEADLINE:
                break
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                # Check if block i has any unique t-subset
                has_unique = False
                for si in block_tsubs_list[i]:
                    if cov_count[si] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    removed[i] = True
                    changed = True
                    for si in block_tsubs_list[i]:
                        cov_count[si] -= 1
                        
        new_blocks = [b for i, b in enumerate(blocks) if not removed[i]]
        
        # 2. Pass 2: Try to replace pairs of blocks with a single block
        # This is expensive: O(n^2 * search for replacement)
        # We can limit this to a few iterations or random pairs if time is tight.
        
        # Re-calculate coverage for new_blocks if we removed any
        if len(new_blocks) < n:
             blocks = new_blocks
             n = len(blocks)
             # Recalculate cov_count and block_tsubs_list for the next phase?
             # Actually, the logic above already updated cov_count.
             # But block_tsubs_list is still the old one. We need to be careful.
             # Let's simplify: just do the removal pass. The replacement pass is complex.
             # Given the time constraints, the removal pass is the most reliable gain.
             
        return blocks

    best_solution = []
    best_size = float('inf')
    
    # Run multiple greedy restarts with different random seeds for tie-breaking
    # The first run uses no randomization (deterministic)
    for i in range(10):
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            sol = run_greedy(random_seed=None)
        else:
            sol = run_greedy(random_seed=i)
            
        if not sol:
            continue
            
        # Reduce the solution
        sol = try_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            # If we found a very good solution, we might want to spend more time on local search?
            # For now, just continue to see if we can find better.
            
    # If we have time, do a more intensive local search on the best solution?
    # The try_reduce above is a basic local search.
    # A more advanced local search would try to swap blocks.
    # Given the 120s limit and the speed of the numpy-based greedy, we have time.
    
    # Let's implement a simple 1-swap local search:
    # For each block in the solution, try replacing it with a different block that maintains coverage.
    # This is hard to do efficiently without the full incidence matrix.
    # We'll stick with the removal pass which is effective.
    
    return best_solution
