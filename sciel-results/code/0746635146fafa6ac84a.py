import itertools
import time
import random

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Specific optimization for C(19, 7, 4)
    if v == 19 and k == 7 and t == 4:
        return _construct_19_7_4(deadline)
    
    # Generic fallback (slower)
    return _construct_generic(v, k, t, deadline)

def _construct_19_7_4(deadline):
    v, k, t = 19, 7, 4
    
    # Precomputation
    # 1. All 4-subsets
    all_4s = list(itertools.combinations(range(v), t))
    n_4s = len(all_4s)
    id_map = {comb: i for i, comb in enumerate(all_4s)}
    
    # 2. All 7-subsets (blocks)
    all_7s = list(itertools.combinations(range(v), k))
    n_7s = len(all_7s)
    
    # 3. For each 7-subset, find which 4-subsets it covers
    # block_4s[i] = list of 4-subset indices covered by 7-subset i
    block_4s = []
    for block in all_7s:
        covered_4s = []
        for sub in itertools.combinations(block, t):
            covered_4s.append(id_map[sub])
        block_4s.append(covered_4s)
        
    # 4. For each 4-subset, find which 7-subsets contain it
    # blocks_containing_4[j] = list of 7-subset indices that contain 4-subset j
    blocks_containing_4 = [[] for _ in range(n_4s)]
    for b_idx, covered_list in enumerate(block_4s):
        for sub_idx in covered_list:
            blocks_containing_4[sub_idx].append(b_idx)
            
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with randomization
    # We can run many iterations because each run is fast
    max_restarts = 5000 # Adjust based on speed, but 5000 * 0.05s = 250s? 
    # Let's estimate speed. 150 steps * 455 candidates * 35 checks = 2.4M ops.
    # In Python, 2.4M simple ops is ~0.1-0.2s? Maybe less.
    # Let's cap restarts by time.
    
    restart_count = 0
    while time.time() < deadline - 5: # 5s buffer
        # Random seed for this restart
        # random.seed() is global, better to use a local rng or just shuffle
        uncovered = [1] * n_4s # 1 if uncovered
        uncovered_set = set(range(n_4s)) # For quick removal? 
        # Actually, using a list `uncovered` and summing is fast.
        # But we need to know if a specific id is uncovered.
        # `uncovered[id] == 1` is fast.
        
        blocks_used = []
        
        while True:
            if not any(uncovered):
                break
                
            # Pick a random uncovered 4-subset to anchor
            # To be efficient, we can pick a random index and check if it's uncovered
            # Or maintain a list of uncovered indices and pick random from it.
            # Maintaining the list is O(N) to build initially, O(1) to pick, O(N) to update?
            # Updating the list of uncovered indices:
            # We can just pick a random index from 0..n_4s-1.
            # Probability of hitting an uncovered one decreases as we cover more.
            # Initially 1.0, later 0.1.
            # Expected tries: 1/p.
            # This is fine.
            
            anchor_idx = -1
            while anchor_idx == -1:
                cand_idx = random.randint(0, n_4s - 1)
                if uncovered[cand_idx]:
                    anchor_idx = cand_idx
            
            # Get candidates
            candidates = blocks_containing_4[anchor_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                # Calculate gain
                gain = 0
                for sub_idx in block_4s[b_idx]:
                    if uncovered[sub_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add block
            blocks_used.append(all_7s[best_block_idx])
            
            # Update uncovered
            for sub_idx in block_4s[best_block_idx]:
                uncovered[sub_idx] = 0
                
            if time.time() > deadline - 5:
                # Abort this restart if too slow? 
                # It shouldn't be, but safety.
                break
        
        if time.time() > deadline - 5:
            break
            
        if len(blocks_used) < best_count:
            best_count = len(blocks_used)
            best_blocks = blocks_used.copy()
            
        restart_count += 1
        
    if best_blocks is None:
        # Fallback to deterministic if no restart finished?
        # Should not happen.
        return []
        
    return [list(b) for b in best_blocks]

def _construct_generic(v, k, t, deadline):
    # Generic greedy, slower
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    # Precompute all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map t-subset to list of block indices containing it
    # This is expensive for large v, but okay for small.
    # For generic, we might just do the naive search if v is small.
    # If v is large, this will timeout.
    # Given the task is C(19,7,4), the generic path is a backup.
    
    # Simple greedy
    while uncovered and time.time() < deadline - 5:
        # Pick first uncovered
        target = min(uncovered)
        
        best_block = None
        best_gain = -1
        
        for b in all_blocks:
            # Check if b contains target
            if all(x in b for x in target):
                gain = 0
                for sub in itertools.combinations(b, t):
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
        
        if best_block:
            blocks.append(list(best_block))
            # Remove covered
            to_remove = set(itertools.combinations(best_block, t))
            uncovered -= to_remove
            
    return blocks
