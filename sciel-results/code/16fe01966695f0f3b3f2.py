import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using:
    1. Beam Search for initial construction
    2. Local search with removal and swap moves
    """
    deadline = time.time() + 140.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subsets it covers (indices)
    block_t_indices = []
    for block in all_blocks:
        indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Reverse index: for each t-subset, which blocks cover it
    t_subset_to_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)

    # Helper: check coverage of a list of block indices
    def check_coverage(block_indices):
        covered = set()
        for idx in block_indices:
            covered.update(block_t_indices[idx])
        return len(covered) == n_t

    # Beam Search
    def beam_search(beam_width=100, max_depth=80):
        # State: (blocks_list, covered_set)
        # Start with empty state
        initial_state = ([], set())
        beam = [initial_state]
        
        for depth in range(max_depth):
            if time.time() > deadline - 25.0:
                break
                
            new_beam = []
            for blocks, covered in beam:
                # If all covered, we can stop expanding this branch (or keep it)
                if len(covered) == n_t:
                    new_beam.append((blocks, covered))
                    continue
                    
                # Generate candidates: blocks that cover the most uncovered triples
                uncovered = [i for i in range(n_t) if i not in covered]
                if not uncovered:
                    new_beam.append((blocks, covered))
                    continue
                
                # Score each potential block by how many new triples it covers
                # To speed up, we can limit candidates to blocks covering at least one uncovered triple
                # and sort by gain.
                
                # Heuristic: Pick a random uncovered triple to focus on, or compute gain for all blocks?
                # Computing gain for all n_blocks is expensive.
                # Instead, pick a few random uncovered triples, find blocks covering them, and evaluate gain.
                
                sample_size = min(5, len(uncovered))
                sampled_uncovered = random.sample(uncovered, sample_size)
                
                candidates = set()
                for ts_idx in sampled_uncovered:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                if not candidates:
                    continue
                
                # Evaluate gain for candidates
                scored_candidates = []
                for b_idx in candidates:
                    # Count how many of this block's triples are NOT in covered
                    gain = 0
                    for idx in block_t_indices[b_idx]:
                        if idx not in covered:
                            gain += 1
                    if gain > 0:
                        scored_candidates.append((gain, b_idx))
                
                if not scored_candidates:
                    continue
                    
                # Sort by gain (descending) and take top few
                scored_candidates.sort(reverse=True)
                
                # Add top 5 candidates to new beam
                for gain, b_idx in scored_candidates[:5]:
                    new_blocks = blocks + [b_idx]
                    new_covered = covered.union(block_t_indices[b_idx])
                    new_beam.append((new_blocks, new_covered))
            
            if not new_beam:
                break
                
            # Prune beam: keep top beam_width states
            # Score: (len(covered), -len(blocks))
            new_beam.sort(key=lambda x: (len(x[1]), -len(x[0])), reverse=True)
            beam = new_beam[:beam_width]
            
            # If any state covers all, we can return it early if we want, 
            # but beam search might find a smaller one later.
            # For now, continue to max_depth or time limit.
            
        # Return the best state from the final beam
        if not beam:
            return []
            
        best_state = max(beam, key=lambda x: (len(x[1]), -len(x[0])))
        return best_state[0]

    # Local search with removal and swap moves
    def local_search(block_indices, seed=0, max_steps=5000):
        rng = random.Random(seed)
        current_indices = list(block_indices)
        best_indices = list(current_indices)
        best_len = len(current_indices)
        
        # Coverage counts for quick update
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in current_indices:
            for j in block_t_indices[idx]:
                counts[j] += 1
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        step = 0
        
        while time.time() < deadline - 15.0 and step < max_steps:
            step += 1
            move_type = rng.random()
            
            if move_type < 0.4 and len(current_indices) > 1:
                # Try to remove a block
                removable_candidates = []
                for i, idx in enumerate(current_indices):
                    can_remove = True
                    for j in block_t_indices[idx]:
                        if counts[j] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable_candidates.append(i)
                
                if not removable_candidates:
                    i = rng.randint(0, len(current_indices) - 1)
                    removable_candidates = [i]
                
                i = rng.choice(removable_candidates)
                idx_to_remove = current_indices[i]
                
                # Check if removal is valid
                new_counts = counts.copy()
                for j in block_t_indices[idx_to_remove]:
                    new_counts[j] -= 1
                
                if np.min(new_counts) >= 1:
                    new_indices = [x for j, x in enumerate(current_indices) if j != i]
                    current_indices = new_indices
                    counts = new_counts
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_indices = list(current_indices)
                
            elif move_type < 0.7:
                # Try a swap: remove one, add another
                if len(current_indices) == 0:
                    continue
                    
                i = rng.randint(0, len(current_indices) - 1)
                idx_to_remove = current_indices[i]
                
                low_count_triples = [j for j in range(n_t) if counts[j] <= 2]
                if not low_count_triples:
                    low_count_triples = list(range(n_t))
                
                sample_size = min(5, len(low_count_triples))
                sampled = rng.sample(low_count_triples, sample_size)
                
                candidates = set()
                for ts_idx in sampled:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                candidates.discard(idx_to_remove)
                
                if not candidates:
                    continue
                
                idx_to_add = rng.choice(list(candidates))
                
                if idx_to_add in current_indices:
                    continue
                
                new_counts = counts.copy()
                for j in block_t_indices[idx_to_remove]:
                    new_counts[j] -= 1
                for j in block_t_indices[idx_to_add]:
                    new_counts[j] += 1
                
                if np.min(new_counts) >= 1:
                    old_score = np.sum(np.minimum(counts, 3))
                    new_score = np.sum(np.minimum(new_counts, 3))
                    delta_score = new_score - old_score
                    
                    if delta_score >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta_score / T)
                    
                    if accept:
                        current_indices[i] = idx_to_add
                        counts = new_counts
                        if len(current_indices) < best_len:
                            best_len = len(current_indices)
                            best_indices = list(current_indices)
            
            else:
                # Pair swap: remove two, add two
                if len(current_indices) < 2:
                    continue
                    
                i, j = rng.sample(range(len(current_indices)), 2)
                idx1_to_remove = current_indices[i]
                idx2_to_remove = current_indices[j]
                
                temp_counts = counts.copy()
                for idx in [idx1_to_remove, idx2_to_remove]:
                    for j_idx in block_t_indices[idx]:
                        temp_counts[j_idx] -= 1
                
                critical_triples = [k for k in range(n_t) if temp_counts[k] <= 0]
                if not critical_triples:
                    low_count = [k for k in range(n_t) if temp_counts[k] <= 2]
                    critical_triples = low_count if low_count else list(range(n_t))
                
                sample_size = min(3, len(critical_triples))
                sampled = rng.sample(critical_triples, sample_size)
                
                candidates = set()
                for ts_idx in sampled:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                candidates.discard(idx1_to_remove)
                candidates.discard(idx2_to_remove)
                
                if len(candidates) < 2:
                    continue
                
                candidates_list = list(candidates)
                idx1_to_add, idx2_to_add = rng.sample(candidates_list, 2)
                
                new_counts = temp_counts.copy()
                for idx in [idx1_to_add, idx2_to_add]:
                    for j_idx in block_t_indices[idx]:
                        new_counts[j_idx] += 1
                
                if np.min(new_counts) >= 1:
                    new_indices = list(current_indices)
                    new_indices[i] = idx1_to_add
                    new_indices[j] = idx2_to_add
                    current_indices = new_indices
                    counts = new_counts
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_indices = list(current_indices)
            
            T = max(T_min, T * cooling_rate)
            if step % 500 == 0 and T == T_min:
                T = 0.3
        
        return best_indices, best_len

    # Run Beam Search to get initial solutions
    # We run it multiple times with different random seeds for candidate selection
    initial_solutions = []
    for seed in range(5):
        random.seed(seed)
        if time.time() > deadline - 20.0:
            break
        sol = beam_search(beam_width=50)
        if check_coverage(sol):
            initial_solutions.append(sol)
    
    if not initial_solutions:
        # Fallback to simple greedy if beam search fails
        def greedy_cover():
            uncovered = set(range(n_t))
            blocks = []
            while uncovered:
                if time.time() > deadline - 10.0:
                    break
                # Pick random uncovered
                ts_idx = random.choice(list(uncovered))
                candidates = t_subset_to_blocks[ts_idx]
                best_block = None
                best_gain = -1
                for b_idx in candidates:
                    gain = 0
                    for idx in block_t_indices[b_idx]:
                        if idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                if best_block is None:
                    break
                blocks.append(best_block)
                uncovered.difference_update(block_t_indices[best_block])
            return blocks
        
        initial_solutions = [greedy_cover()]
    
    # Local search on best initial solution
    best_indices = min(initial_solutions, key=len)
    best_len = len(best_indices)
    
    for sol in initial_solutions:
        if time.time() > deadline - 10.0:
            break
        ls_indices, ls_len = local_search(sol, seed=random.randint(0, 10000))
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Additional local search passes
    for seed in range(3):
        if time.time() > deadline - 10.0:
            break
        ls_indices, ls_len = local_search(best_indices, seed=seed + 1000)
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Final validation and conversion
    if not check_coverage(best_indices):
        for sol in initial_solutions:
            if check_coverage(sol):
                best_indices = sol
                break
        else:
            best_indices = greedy_cover()
    
    result = [all_blocks[i] for i in best_indices]
    result = list(set(result))
    
    # Ensure validity
    final_covered = set()
    for b in result:
        final_covered.update(block_t_indices[all_blocks.index(b)])
        
    if len(final_covered) != n_t:
        # Fix by adding missing blocks
        uncovered = [i for i in range(n_t) if i not in final_covered]
        for ts_idx in uncovered:
            candidates = t_subset_to_blocks[ts_idx]
            for b_idx in candidates:
                b = all_blocks[b_idx]
                if b not in result:
                    result.append(b)
                    final_covered.update(block_t_indices[b_idx])
                    break
    
    return [list(b) for b in result]
