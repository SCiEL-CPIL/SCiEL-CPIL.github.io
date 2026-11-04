import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    if v != 19 or k != 8 or t != 5:
        # Fallback for other parameters if ever called (though task is fixed)
        return greedy_fallback(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 145.0

    def check_time(budget=10.0):
        return time.time() > deadline - budget

    # --- Strategy: Combinatorial Group Testing / Disjunct Matrix Construction ---
    # We model the problem as finding a set of blocks (columns in a matrix) such that 
    # for any set of t "defective" items (t-subset), there is at least one block 
    # that contains ALL of them.
    # This is equivalent to a t-cover-free family.
    
    # A simple deterministic construction uses "separating systems" or similar.
    # However, for C(19,8,5), we can use a probabilistic method or a structured code.
    # Let's try a construction based on random blocks but guided by a "disjunct" property check,
    # or better, use a known upper bound construction.
    
    # The current best is 335. 
    # A simple random greedy usually yields ~400+.
    # Let's use a "Greedy with Conflict Resolution" or "Local Search from Random Start".
    
    # Actually, the prompt asks for a MATERIALLY DIFFERENT strategy.
    # The current one is "Random Restart Greedy + Reduction".
    # Let's try "Simulated Annealing on the set of blocks" starting from a dense random set.
    # Or, better: "Exact Cover via Integer Programming relaxation" is too slow.
    
    # Let's try a **Genetic Algorithm** approach on the set of blocks.
    # Representation: A population of sets of blocks.
    # Fitness: -len(blocks) + penalty * (uncovered_tsubs)
    # Crossover: Union of two sets, then prune.
    # Mutation: Add a random block, or remove a random block, or swap.
    
    # To make it efficient, we represent the set of blocks as a boolean vector of length C(19,8)=75582.
    # But that's large. Instead, we keep a list of block indices.
    
    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage masks? 
    # Each block covers C(8,5)=56 t-subsets.
    # We can represent the set of covered t-subsets for a set of blocks as a bitset or boolean array.
    # Using numpy uint8 array of size num_tsubs (11628) is ~11KB per solution.
    # That's manageable.
    
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
    
    # Convert to numpy arrays for fast union operations?
    # block_tsubs_np: (num_blocks, 56)
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    
    # For fast "is this block redundant?" check, we need cover counts.
    
    def evaluate_solution(block_indices):
        """Returns (num_blocks, num_uncovered)"""
        if not block_indices:
            return 0, num_tsubs
        covered = np.zeros(num_tsubs, dtype=np.uint8)
        for b_idx in block_indices:
            covered[block_tsubs_np[b_idx]] = 1
        uncovered = num_tsubs - np.sum(covered)
        return len(block_indices), uncovered

    def prune_redundant(block_indices):
        """Remove blocks that do not uniquely cover any t-subset."""
        block_indices = list(block_indices)
        if not block_indices:
            return []
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1
            
        new_indices = []
        for b_idx in block_indices:
            b_tsubs = block_tsubs_np[b_idx]
            # If this block is the only one covering any t-subset, it's essential
            if np.any(cover_count[b_tsubs] == 1):
                new_indices.append(b_idx)
            else:
                # It's redundant, remove it
                cover_count[b_tsubs] -= 1
                
        return new_indices

    # --- Simulated Annealing ---
    # Start with a greedy solution to get a good initial state
    def greedy_initial():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected = []
        
        # Precompute for each t-subset, the list of blocks covering it
        tsub_blocks = [[] for _ in range(num_tsubs)]
        for i, b_tsubs in enumerate(block_tsubs):
            for j in b_tsubs:
                tsub_blocks[j].append(i)
                
        # Convert to numpy for fast access? 
        # tsub_blocks is a list of lists. 
        # To speed up, we can just use the list.
        
        while uncovered_count > 0:
            if check_time(5.0):
                break
            # Pick a random uncovered t-subset
            # Optimization: maintain a list of uncovered indices? 
            # Or just sample.
            # Since we start dense, sampling is fine.
            while True:
                s_idx = random.randint(0, num_tsubs - 1)
                if uncovered[s_idx]:
                    break
                    
            candidates = tsub_blocks[s_idx]
            if not candidates:
                break
                
            # Choose block with max gain
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                # Gain is number of uncovered t-subsets in this block
                # We can compute this by checking uncovered[block_tsubs_np[b_idx]]
                # But this is slow if done for all candidates every time.
                # Heuristic: just pick a random one from candidates? 
                # Or precompute gains? Gains change as uncovered changes.
                
                # Let's stick to a simple greedy: pick random candidate. 
                # This is fast but might not be optimal. 
                # Better: pick the one that covers the most "rare" uncovered sets?
                # For now, random selection from candidates is a valid "greedy" variant (randomized greedy).
                
                b_idx = random.choice(candidates)
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        uncovered[j] = 0
                        uncovered_count -= 1
                selected.append(b_idx)
                break # Just pick one? No, standard greedy picks best.
                # Let's do best-gain for the first few steps, then random?
                # Actually, let's do a proper best-gain but limit candidates?
                # No, let's just use the random choice for speed in initialization, 
                # then SA will fix it.
                
        return selected

    # Better initialization: Deterministic Greedy with NumPy
    def greedy_init_numpy():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected = []
        
        # tsub_blocks_np: list of arrays
        tsub_blocks_np = [np.array(tb, dtype=np.int32) for tb in tsub_blocks]
        
        while uncovered_count > 0:
            if check_time(5.0):
                break
            while True:
                s_idx = random.randint(0, num_tsubs - 1)
                if uncovered[s_idx]:
                    break
                    
            candidates = tsub_blocks_np[s_idx]
            if len(candidates) == 0:
                break
                
            # Calculate gains for all candidates
            # This might be slow if len(candidates) is large.
            # Max candidates for a t-subset: C(19-5, 8-5) = C(14,3) = 364.
            # 364 * 56 ops = ~20k ops. Acceptable.
            
            cand_tsubs = block_tsubs_np[candidates]
            gains = np.sum(uncovered[cand_tsubs], axis=1)
            best_local = int(np.argmax(gains))
            best_b = int(candidates[best_local])
            
            selected.append(best_b)
            
            # Update uncovered
            for j in block_tsubs[best_b]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected

    # Run initial greedy
    current_blocks = greedy_init_numpy()
    current_blocks = prune_redundant(current_blocks)
    current_len, current_unc = evaluate_solution(current_blocks)
    
    best_blocks = current_blocks
    best_len = current_len
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps = 5000
    
    rng = random.Random(42)
    
    for step in range(steps):
        if check_time(20.0):
            break
            
        # Generate neighbor
        # Move types:
        # 1. Remove a random block (if it doesn't break coverage? Or allow it and penalize)
        # 2. Add a random block
        # 3. Swap a block with another
        
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        new_blocks = list(current_blocks)
        
        if move_type == 'remove':
            if not new_blocks:
                continue
            idx_to_remove = rng.randint(0, len(new_blocks) - 1)
            new_blocks.pop(idx_to_remove)
            
        elif move_type == 'add':
            # Add a random block that is not already present
            # To make it effective, add a block that covers some uncovered t-subsets?
            # Or just random?
            # Random add is slow to converge. 
            # Let's add a block that covers a random uncovered t-subset.
            # First, find uncovered t-subsets.
            covered = np.zeros(num_tsubs, dtype=np.uint8)
            for b_idx in new_blocks:
                covered[block_tsubs_np[b_idx]] = 1
            uncov_indices = np.where(covered == 0)[0]
            if len(uncov_indices) > 0:
                s_idx = int(rng.choice(uncov_indices))
                candidates = tsub_blocks[s_idx]
                if candidates:
                    new_b = rng.choice(candidates)
                    if new_b not in new_blocks:
                        new_blocks.append(new_b)
            else:
                # All covered, just add a random block to try to enable removals later?
                # Or skip?
                pass
                
        elif move_type == 'swap':
            if not new_blocks:
                continue
            idx_to_swap = rng.randint(0, len(new_blocks) - 1)
            old_b = new_blocks[idx_to_swap]
            
            # Find a new block that covers something old_b covered, or better
            # Let's pick a random block that shares at least one t-subset with old_b?
            # Or just random block?
            # To be efficient, pick a random block from the universe.
            new_b = rng.randint(0, num_blocks - 1)
            if new_b == old_b:
                continue
            if new_b in new_blocks:
                continue
            new_blocks[idx_to_swap] = new_b
            
        # Evaluate new solution
        new_len, new_unc = evaluate_solution(new_blocks)
        
        # Fitness: We want to minimize len, but must have unc=0.
        # If unc > 0, it's invalid. We should penalize heavily.
        # Score = len + 100 * unc
        old_score = current_len + 100 * current_unc
        new_score = new_len + 100 * new_unc
        
        # Acceptance
        if new_score < old_score:
            current_blocks = new_blocks
            current_len, current_unc = new_len, new_unc
            
            # Try to prune
            pruned = prune_redundant(current_blocks)
            if len(pruned) < current_len:
                current_blocks = pruned
                current_len, current_unc = evaluate_solution(current_blocks)
                
            if current_unc == 0 and current_len < best_len:
                best_blocks = current_blocks
                best_len = current_len
                
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - old_score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_blocks = new_blocks
                current_len, current_unc = new_len, new_unc
                
        T *= alpha
        if T < T_min:
            break
            
    # Final Pruning
    if best_blocks:
        best_blocks = prune_redundant(best_blocks)
        
    # Convert to list of lists of ints
    return [list(all_blocks[i]) for i in best_blocks]

def greedy_fallback(v, k, t):
    """Simple fallback for non-target parameters."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    all_blocks = list(itertools.combinations(range(v), k))
    
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    uncovered = np.ones(len(all_tsubs), dtype=np.uint8)
    uncovered_count = len(all_tsubs)
    selected = []
    
    while uncovered_count > 0:
        s_idx = random.randint(0, len(all_tsubs) - 1)
        if not uncovered[s_idx]:
            continue
            
        best_b = -1
        best_gain = -1
        for i, b_tsubs in enumerate(block_tsubs):
            if not b_tsubs: continue
            gain = 0
            for j in b_tsubs:
                if uncovered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_b = i
                
        if best_b == -1:
            break
            
        selected.append(best_b)
        for j in block_tsubs[best_b]:
            if uncovered[j]:
                uncovered[j] = 0
                uncovered_count -= 1
                
    return [list(all_blocks[i]) for i in selected]

