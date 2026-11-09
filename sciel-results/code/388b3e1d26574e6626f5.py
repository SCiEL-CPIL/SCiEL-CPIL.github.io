import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all candidate blocks and their 5-subsets
    points = range(12)
    all_6subsets = list(combinations(points, 6))
    
    # Map each 5-subset to the list of 6-subsets (blocks) that contain it
    # This allows for fast conflict checking and potential backtracking
    subset_to_blocks = {}
    block_subsets = {}  # block -> list of 5-subsets
    
    for b in all_6subsets:
        subsets = list(combinations(b, 5))
        block_subsets[b] = subsets
        for s in subsets:
            if s not in subset_to_blocks:
                subset_to_blocks[s] = []
            subset_to_blocks[s].append(b)
    
    best_blocks = []
    best_size = 0
    
    # Strategy: 
    # 1. Try a few fast randomized greedy restarts.
    # 2. If we haven't found 132, use a smarter heuristic: 
    #    Sort candidates by a "conflict potential" metric or use a 
    #    weighted greedy approach that prioritizes blocks covering 
    #    "scarce" 5-subsets (those contained in fewer total blocks, though 
    #    by symmetry all 5-subsets are contained in C(7,1)=7 blocks? No, 
    #    wait. In the full set of 924 blocks, how many blocks contain a specific 
    #    5-subset? It must contain the 5 points plus one more from the remaining 7. 
    #    So exactly 7 blocks contain any given 5-subset. Symmetry holds.
    #    
    #    However, during the greedy process, the "availability" of subsets changes.
    #    A better heuristic: 
    #    - Use a priority queue or sort by the number of currently available 5-subsets?
    #    - Or, simply try many more restarts with different random seeds, 
    #      but optimize the inner loop.
    #
    #    The current program does simple shuffling. 
    #    Inspiration 1/2 suggests that finding 132 is possible with random greedy 
    #    but might take many tries. 
    #    
    #    Let's try a "conflict-aware" greedy:
    #    At each step, among all valid blocks, pick the one that "locks in" 
    #    the most future conflicts? Or the one that is most constrained?
    #    
    #    Actually, a known technique for Steiner systems is "greedy with backtracking" 
    #    or "simulated annealing". But 120s is short.
    #    
    #    Let's stick to randomized greedy but optimize the check.
    #    The current check is O(6) per block. 
    #    We can maintain a count of how many times each 5-subset is covered.
    #    If count > 0, conflict.
    
    covered_count = {}
    
    # To make it faster, we can pre-sort blocks or use a specific heuristic.
    # Let's try a heuristic: pick the block that minimizes the sum of 
    # "remaining degrees" of its 5-subsets? 
    # Degree of a 5-subset = number of blocks containing it that are still valid.
    # This is expensive to maintain dynamically.
    
    # Alternative: Just do many fast restarts. 
    # The current program does ~10-20 restarts in 110s? 
    # Let's check the speed. 
    # 924 blocks * 6 subsets * set lookup. Very fast. 
    # We can probably do thousands of restarts.
    
    # Let's try a different strategy: 
    # Start with a random block, then greedily add blocks that have the 
    # "least overlap" with already chosen blocks in terms of future potential?
    # No, just random shuffle is often good enough if you do enough trials.
    
    # Let's try to be smarter: 
    # Use a "weighted" random selection where the probability of picking a block 
    # is proportional to something? 
    # Or, simply, try to find a solution by fixing the first few blocks 
    # from a known partial design? 
    # 
    # Actually, the most robust improvement for this specific problem (Witt design) 
    # is to use a known constructive algorithm or a very efficient backtracking 
    # with symmetry breaking. 
    # 
    # But since we can't rely on external libraries for the design, 
    # and the random greedy works (score 1.16 means it often finds near-optimal 
    # or optimal), let's optimize the random greedy to run more iterations.
    
    # Optimization: 
    # Instead of `any(s in covered for s in subsets)`, 
    # we can use a set of covered 5-subsets.
    
    for attempt in range(10000):  # High limit, but time-bounded
        if time.time() > deadline - 1.0:
            break
            
        random.seed(attempt)
        
        covered = set()
        current_blocks = []
        
        shuffled = all_6subsets[:]
        random.shuffle(shuffled)
        
        for b in shuffled:
            subsets = block_subsets[b]
            # Check conflict
            conflict = False
            for s in subsets:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                current_blocks.append(b)
                covered.update(subsets)
        
        if len(current_blocks) > best_size:
            best_size = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
            if best_size == 132:
                return best_blocks
        
        # If we've done a few hundred attempts and haven't found it, 
        # maybe switch to a different heuristic?
        # For now, just keep going.
        
    return best_blocks
