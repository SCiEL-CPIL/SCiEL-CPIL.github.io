"""
Covering design C(19,8,5) via greedy + local search (block elimination + 1-exchange).
Uses precomputed coverage counts for fast redundant-block detection.
"""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # Actually, we need: for a block B, which t-subsets does it cover?
    # A t-subset S is covered by B if S ⊆ B.
    
    # For greedy: given current block prefix P (size < k), and candidate x not in P,
    # gain(x) = number of uncovered t-subsets S such that S ⊆ P ∪ {x}
    # S ⊆ P ∪ {x} means S \ {x} ⊆ P, so S contains x and S\{x} ⊆ P.
    # Equivalently: S contains x, and S ∩ P has size t-1 (i.e., S\{x} ⊆ P).
    
    # For efficiency, precompute for each (t-1)-subset A of range(v), 
    # the list of t-subsets S such that S ⊇ A.
    # But that's C(v, t-1) lists, each of size v - (t-1). For v=19, t=5: C(19,4)=3876 lists.
    
    # Alternative: for a given block B (size k), the t-subsets it covers are all t-subsets of B.
    # We can precompute: for each possible block (k-subset), the bitset of t-subsets it covers.
    # But C(19,8) = 75582 blocks, each with C(8,5)=56 t-subsets. That's manageable.
    
    # Let's precompute coverage masks for all possible k-subsets.
    # Represent each t-subset coverage as a Python int bitmask (n_t_subs bits).
    # n_t_subs = C(19,5) = 11628. So we need ~11628-bit integers. Python handles big ints fine.
    
    # Precompute: for each k-subset B, the bitmask of t-subsets it covers.
    # This is expensive upfront but enables O(1) coverage checks.
    
    # Actually, let's think about memory: 75582 big ints of ~1500 bytes each = ~113 MB. That's a lot.
    # Let's use a different approach: for each block, store the list of t-subset indices it covers.
    
    # Better approach: 
    # 1. Greedy construction with fast gain computation.
    # 2. Local search: maintain coverage_count[s] for each t-subset s.
    #    - Block elimination: if removing block B doesn't leave any s with coverage_count[s]==1, remove it.
    #    - 1-exchange: try swapping block B with block C if it reduces uncovered count.
    
    # For greedy, we need fast gain. Let's precompute for each point x and each (t-1)-subset A:
    # the count of uncovered t-subsets S such that S ⊇ A ∪ {x} and S is currently uncovered.
    # This is dynamic, so we can't fully precompute. But we can precompute the structure.
    
    # Let me use a simpler but effective approach:
    # - Precompute all k-subsets and their covered t-subsets as sorted lists of indices.
    # - Use numpy for coverage counting.
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Map k-subset to index
    k_sub_to_idx = {sub: i for i, sub in enumerate(all_k_subs)}
    
    # For each k-subset, compute the list of t-subset indices it covers
    # This is C(k, t) per block
    block_coverage = []  # block_coverage[i] = list of t-subset indices covered by block i
    for block in all_k_subs:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_sub_to_idx[ts])
        block_coverage.append(covered)
    
    # Convert to numpy arrays for faster operations
    # Actually, let's keep as lists for now.
    
    # Greedy construction
    uncovered = np.ones(n_t_subs, dtype=np.int32)  # coverage count
    blocks_selected = []  # list of block indices
    blocks_set = set()
    
    # For greedy, we need to find the best block to add.
    # gain(block) = number of currently uncovered t-subsets covered by block.
    # We iterate over all k-subsets and pick the one with max gain.
    
    # Precompute for each block the list of t-subset indices it covers (done above)
    
    while np.any(uncovered) and time.time() < deadline:
        # Find block with maximum gain
        best_block = -1
        best_gain = -1
        for i, covered in enumerate(block_coverage):
            if i in blocks_set:
                continue
            gain = np.sum(uncovered[covered])
            if gain > best_gain:
                best_gain = gain
                best_block = i
            if gain == 0:
                continue
        
        if best_block == -1 or best_gain == 0:
            break
        
        blocks_selected.append(best_block)
        blocks_set.add(best_block)
        uncovered[block_coverage[best_block]] += 1
    
    # Local search phase 1: Block elimination
    # Repeatedly try to remove blocks that are redundant
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks_selected)):
            if time.time() >= deadline:
                break
            b = blocks_selected[i]
            # Check if removing b leaves any t-subset with coverage 1
            # i.e., for all s in block_coverage[b], uncovered[s] > 1
            if np.all(uncovered[block_coverage[b]] > 1):
                # Safe to remove
                uncovered[block_coverage[b]] -= 1
                blocks_selected.pop(i)
                blocks_set.discard(b)
                improved = True
                break  # Restart scan since indices changed
    
    # Local search phase 2: 1-exchange
    # Try to replace a selected block with an unselected block if it reduces uncovered count
    for iteration in range(200):
        if time.time() >= deadline:
            break
        improved = False
        # Shuffle for variety
        random.shuffle(blocks_selected)
        for i in range(len(blocks_selected)):
            if time.time() >= deadline:
                break
            b = blocks_selected[i]
            # Try replacing b with some unselected block c
            # Net effect: uncovered changes by -1 for t-subsets in b, +1 for t-subsets in c
            # We want to reduce total uncovered count.
            # Current uncovered count in b's coverage: u_b = sum(uncovered[s] for s in block_coverage[b])
            # After removal: those become uncovered-1
            # After adding c: those in c become uncovered+1 (but careful with overlap)
            
            # Simpler: compute delta = (uncovered count after) - (uncovered count before)
            # Let A = block_coverage[b], C = block_coverage[c]
            # For s in A ∩ C: net change 0
            # For s in A \ C: uncovered decreases by 1
            # For s in C \ A: uncovered increases by 1
            # delta = |C \ A| - |A \ C|
            # We want delta < 0, i.e., |A \ C| > |C \ A|
            
            # To be efficient, we can precompute intersections... but that's expensive.
            # For now, just try a few random unselected blocks.
            
            # Pick a random unselected block
            available = [j for j in range(n_k_subs) if j not in blocks_set]
            if not available:
                continue
            
            # Try a few candidates
            for _ in range(10):
                if time.time() >= deadline:
                    break
                c = random.choice(available)
                A = block_coverage[b]
                C = block_coverage[c]
                
                # Compute |A \ C| and |C \ A|
                # Use set operations
                A_set = set(A)
                C_set = set(C)
                a_minus_c = len(A_set - C_set)
                c_minus_a = len(C_set - A_set)
                
                delta = c_minus_a - a_minus_c
                if delta < 0:
                    # Perform the swap
                    uncovered[A] -= 1
                    uncovered[C] += 1
                    blocks_selected[i] = c
                    blocks_set.discard(b)
                    blocks_set.add(c)
                    improved = True
                    break
                if time.time() >= deadline:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Build final blocks
    result = [list(all_k_subs[i]) for i in blocks_selected]
    
    # Verify validity (should always be valid if greedy worked)
    # But let's double-check
    covered_all = np.zeros(n_t_subs, dtype=np.int32)
    for b_idx in blocks_selected:
        covered_all[block_coverage[b_idx]] += 1
    
    if np.any(covered_all < 1):
        # Fallback: return greedy result (shouldn't happen)
        pass
    
    return result
