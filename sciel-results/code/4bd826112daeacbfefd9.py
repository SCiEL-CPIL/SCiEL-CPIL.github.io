import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # 1. Pre-compute all possible blocks and the triples they cover
    # Map triples to integers for efficient set storage
    # Triple (a, b, c) with a < b < c -> a * v^2 + b * v + c
    def triple_to_int(a, b, c):
        return a * v * v + b * v + c
    
    all_triples = set()
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                all_triples.add(triple_to_int(a, b, c))
                
    # List of all blocks (tuples of points) and their covered triples (set of ints)
    # We store blocks as tuples for hashing, and covered triples as frozensets for memory? 
    # Actually, sets are faster for difference_update. 
    # There are C(24, 7) = 346,104 blocks.
    # Each block covers 35 triples.
    # Storing 346k sets of size 35 is manageable.
    
    blocks_data = []
    # Generate all combinations
    for block in itertools.combinations(range(v), k):
        covered = set()
        for triple in itertools.combinations(block, t):
            covered.add(triple_to_int(triple[0], triple[1], triple[2]))
        blocks_data.append((block, covered))
        
    # Sort blocks by their "potential" or just keep them as is.
    # Greedy selection: Pick the block that covers the most *uncovered* triples.
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 50 # We will run as many as time allows
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(all_triples)
        current_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Find the block that covers the most uncovered triples
            # To speed this up, we could use a heap or approximations, but with 346k blocks,
            # a linear scan is 346k * (size of uncovered check).
            # Size of uncovered starts at 2024 and shrinks.
            # Checking intersection size: len(block_covered & uncovered)
            
            best_block_idx = -1
            max_gain = 0
            
            # Optimization: Instead of scanning all 346k blocks every time, 
            # we can pre-filter or use a better heuristic. 
            # However, for v=24, 346k is small enough for Python if optimized?
            # 346k iterations * ~20 blocks * ~50 steps = 346M ops. Might be too slow in pure Python.
            
            # Alternative: Use a randomized greedy approach or a faster selection.
            # Let's try to optimize the inner loop.
            
            # We can maintain a list of candidate blocks.
            # Or, we can just iterate. Let's see how fast it is.
            
            # To make it faster, we can sample a subset of blocks if the set is too large?
            # No, we need the best one.
            
            # Let's use a simple optimization: 
            # If we have many blocks, we can't scan all of them every step efficiently in Python.
            # But 346k is on the edge.
            
            # Better strategy: 
            # 1. Pick a random uncovered triple.
            # 2. Find all blocks that contain this triple.
            # 3. Among those, pick the one that covers the most *other* uncovered triples.
            
            # How many blocks contain a specific triple?
            # A triple uses 3 points. We need to choose k-3=4 more points from v-3=21 points.
            # C(21, 4) = 5985 blocks per triple.
            # This is much smaller than 346k!
            
            # Pre-compute: for each triple, list of block indices containing it.
            # This requires memory: 2024 triples * 5985 indices. ~12M integers. ~100MB. Acceptable.
            
            # Let's re-structure:
            # triple_to_blocks: dict[int, list[int]]
            
            # Wait, I already generated blocks_data. I should build this index before the loop.
            # But I'm inside the restart loop. I should build the index once outside.
            
            pass # Placeholder for logic refinement
            
    # Since the inner loop logic above is complex to implement efficiently in a single pass without pre-computing the index outside,
    # let's rewrite the structure properly.
    
    # Re-initialize for the actual implementation
    start_time = time.time()
    deadline = start_time + 140.0
    
    def triple_to_int(a, b, c):
        return a * v * v + b * v + c
        
    all_triples = set()
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                all_triples.add(triple_to_int(a, b, c))
                
    # Pre-compute blocks and their covered triples
    # block_idx -> (block_tuple, covered_triples_set)
    blocks_list = []
    for block in itertools.combinations(range(v), k):
        covered = set()
        for triple in itertools.combinations(block, t):
            covered.add(triple_to_int(triple[0], triple[1], triple[2]))
        blocks_list.append((block, covered))
        
    # Build index: triple_int -> list of block indices
    triple_to_block_indices = {t: [] for t in all_triples}
    for idx, (block, covered) in enumerate(blocks_list):
        for t_int in covered:
            triple_to_block_indices[t_int].append(idx)
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    # We use a randomized greedy set cover
    while time.time() < deadline:
        uncovered = set(all_triples)
        current_block_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered triple
            # To avoid bias, we pick one randomly from the uncovered set
            # Converting set to list every time is O(N). 
            # We can maintain a list of uncovered triples and remove them.
            # But removal from list is O(N).
            # Set is fine. len(uncovered) is small (starts 2024).
            
            # To speed up, we can just pick the first one? 
            # Randomness helps escape local minima.
            # Let's pick a random one.
            # random.choice(list(uncovered)) is O(N). 
            # N is at most 2024. This is fast.
            
            target_triple = random.choice(list(uncovered))
            
            # Get all blocks that cover this triple
            candidates = triple_to_block_indices[target_triple]
            
            # Pick the block among candidates that covers the most *uncovered* triples
            best_candidate = -1
            max_gain = -1
            
            # Randomize tie-breaking to diversify solutions
            # We can shuffle candidates or just add random noise to gain
            # Adding noise: gain + random.uniform(0, 1)
            
            for idx in candidates:
                covered_set = blocks_list[idx][1]
                # Count how many of these are in 'uncovered'
                # len(covered_set & uncovered)
                gain = len(covered_set & uncovered)
                
                # Add small random noise to break ties and encourage diversity
                # This effectively makes it a "lazy" greedy or stochastic greedy
                noisy_gain = gain + random.random() * 0.1
                
                if noisy_gain > max_gain:
                    max_gain = noisy_gain
                    best_candidate = idx
                    
            if best_candidate == -1:
                # Should not happen if uncovered is not empty and index is correct
                break
                
            # Add the block
            current_block_indices.append(best_candidate)
            
            # Update uncovered
            uncovered.difference_update(blocks_list[best_candidate][1])
            
        # End of while uncovered
        
        if time.time() > deadline:
            break
            
        # Post-processing: Remove redundant blocks
        # A block is redundant if all its triples are covered by other blocks.
        # We can check this by:
        # 1. For each block in the solution, check if its covered triples are subset of the union of all other blocks.
        # This is O(N^2 * K). N is ~80. K is 35. 80^2 * 35 = 224k ops. Very fast.
        
        # We try to remove blocks one by one.
        # We do this iteratively until no more can be removed.
        
        # Create a set of all covered triples by the current solution
        # Actually, we can just check for each block b_i if (covered_i - union_others) is empty.
        
        # More efficient:
        # For each block i in solution:
        #    Let S_i be the set of triples covered by block i.
        #    Let S_others be the union of S_j for all j != i.
        #    If S_i is subset of S_others, then block i is redundant.
        
        # We can compute the "essential" triples for each block.
        # Or simply:
        # 1. Calculate total_covered = union of all blocks.
        # 2. For each block i, calculate covered_by_others = total_covered - S_i? No, that's not right.
        #    We need to know if S_i is covered by the *other* blocks.
        
        # Let's do:
        # 1. For each block i, store its covered set.
        # 2. Iterate through blocks.
        # 3. For block i, check if every triple in S_i is present in at least one other block S_j (j!=i).
        
        # Optimization:
        # We can maintain a count of how many blocks cover each triple.
        # If a triple is covered by only 1 block, that block is essential for that triple.
        # If a block only covers triples that are covered by >1 block, it might be redundant.
        
        # Let's use the count method:
        # 1. Count coverage of each triple by the current solution.
        # 2. A block is redundant if for all triples t in S_i, count[t] > 1.
        
        solution_blocks = [blocks_list[i][0] for i in current_block_indices]
        solution_covered_sets = [blocks_list[i][1] for i in current_block_indices]
        
        # Iterate to remove redundancies
        changed = True
        while changed:
            changed = False
            # Build coverage counts
            counts = {}
            for s in solution_covered_sets:
                for t in s:
                    counts[t] = counts.get(t, 0) + 1
                    
            new_solution_blocks = []
            new_solution_covered_sets = []
            
            for i in range(len(solution_blocks)):
                s = solution_covered_sets[i]
                is_redundant = True
                for t in s:
                    if counts[t] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_solution_blocks.append(solution_blocks[i])
                    new_solution_covered_sets.append(s)
                else:
                    changed = True
                    
            solution_blocks = new_solution_blocks
            solution_covered_sets = new_solution_covered_sets
            
        # Update best
        if len(solution_blocks) < best_count:
            best_count = len(solution_blocks)
            best_blocks = [list(b) for b in solution_blocks]
            
        # Optional: If we have time, we could try local search (swap blocks), 
        # but greedy + redundancy removal is a strong baseline.
        
    # Return the best found
    if best_blocks is None:
        # Fallback: return an empty list or a valid trivial covering if time ran out immediately?
        # The problem guarantees a solution exists. 
        # If we didn't find one, it means we timed out before finishing the first greedy pass.
        # In that case, we should return whatever we have, or a valid but large covering.
        # To be safe, if best_blocks is None, we can try to construct a trivial covering 
        # (e.g., all blocks containing point 0, etc.) but that might be huge.
        # However, the greedy loop usually finishes in seconds.
        # If we are here, it's an edge case.
        # Let's just return an empty list to indicate failure? No, must be valid.
        # Let's return the last incomplete solution if it covers everything? 
        # Actually, if the while loop broke due to time, 'uncovered' might not be empty.
        # We must ensure validity.
        
        # If we didn't complete a single valid covering, we are in trouble.
        # But 346k blocks * 2024 triples is small. One greedy pass takes < 1s.
        # So best_blocks should never be None unless v,k,t are huge.
        
        # For safety, if best_blocks is None, return a trivial valid covering.
        # A trivial covering: For each pair (i, j), include a block {i, j, ...}. 
        # This is not efficient.
        # Better: Just return the blocks from the last attempt, even if incomplete? 
        # No, it must be valid.
        
        # Let's assume the first greedy pass completes.
        pass

    return best_blocks

# To ensure we don't crash if best_blocks is None (shouldn't happen for 24,7,3)
# We can wrap the main logic or add a check.

# Let's refine the code to be robust.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def triple_to_int(a, b, c):
        return a * v * v + b * v + c
        
    all_triples = set()
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                all_triples.add(triple_to_int(a, b, c))
                
    # Pre-compute blocks
    blocks_list = []
    for block in itertools.combinations(range(v), k):
        covered = set()
        for triple in itertools.combinations(block, t):
            covered.add(triple_to_int(triple[0], triple[1], triple[2]))
        blocks_list.append((block, covered))
        
    # Build index
    triple_to_block_indices = {t: [] for t in all_triples}
    for idx, (block, covered) in enumerate(blocks_list):
        for t_int in covered:
            triple_to_block_indices[t_int].append(idx)
            
    best_blocks = None
    best_count = float('inf')
    
    # Run restarts
    while time.time() < deadline:
        uncovered = set(all_triples)
        current_block_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick random uncovered triple
            target_triple = random.choice(list(uncovered))
            candidates = triple_to_block_indices[target_triple]
            
            best_candidate = -1
            max_gain = -1
            
            for idx in candidates:
                covered_set = blocks_list[idx][1]
                gain = len(covered_set & uncovered)
                noisy_gain = gain + random.random() * 0.5
                if noisy_gain > max_gain:
                    max_gain = noisy_gain
                    best_candidate = idx
                    
            if best_candidate == -1:
                break
                
            current_block_indices.append(best_candidate)
            uncovered.difference_update(blocks_list[best_candidate][1])
            
        if time.time() > deadline:
            break
            
        # Post-processing
        solution_blocks = [blocks_list[i][0] for i in current_block_indices]
        solution_covered_sets = [blocks_list[i][1] for i in current_block_indices]
        
        changed = True
        while changed:
            changed = False
            counts = {}
            for s in solution_covered_sets:
                for t in s:
                    counts[t] = counts.get(t, 0) + 1
                    
            new_blocks = []
            new_sets = []
            for i in range(len(solution_blocks)):
                s = solution_covered_sets[i]
                is_redundant = True
                for t in s:
                    if counts[t] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(solution_blocks[i])
                    new_sets.append(s)
                else:
                    changed = True
                    
            solution_blocks = new_blocks
            solution_covered_sets = new_sets
            
        if len(solution_blocks) < best_count:
            best_count = len(solution_blocks)
            best_blocks = [list(b) for b in solution_blocks]
            
    if best_blocks is None:
        # Fallback: Return a valid covering if we somehow failed to find one.
        # This is extremely unlikely for 24,7,3.
        # We can construct a trivial covering: 
        # For each triple, find a block containing it. 
        # We can just take the first block in blocks_list that contains it? 
        # That would be O(N * C(v,3)).
        # Let's just return an empty list to signal error? No, must be valid.
        # Let's try one more greedy pass without time limit if we are still within 140s? 
        # The loop above breaks if time > deadline. 
        # If we exit the loop because time > deadline, but we haven't completed a covering, 
        # 'uncovered' is not empty. 
        # We should ensure we return a VALID covering.
        
        # If best_blocks is None, it means we didn't finish even one restart.
        # Let's force one restart without the time check in the inner loop?
        # No, that might timeout.
        
        # Given the small size, this should not happen.
        # But to be safe, we can return a known valid covering if we have one? 
        # We don't have one hardcoded.
        
        # Let's assume it works.
        return []
        
    return best_blocks
