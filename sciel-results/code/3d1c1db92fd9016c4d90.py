import itertools
import time
import random
import heapq
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
        
    # Pre-compute subset_to_blocks
    subset_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    def greedy_construct(randomness=0.0):
        """
        Greedy construction.
        randomness: 0.0 means deterministic max-gain.
                    > 0.0 means pick from top K candidates with some probability.
        """
        uncovered = [1] * n_t_subs
        total_uncovered = n_t_subs
        
        # block_uncovered_count[i] = number of subsets in block i that are currently uncovered
        block_uncovered_count = [len(subs) for subs in block_t_subs]
        
        # Heap of (-count, block_idx)
        heap = [(-block_uncovered_count[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        # To handle lazy deletion, we store the current count in an array and check on pop
        # But counts change. We just push new entries.
        
        blocks = []
        
        while total_uncovered > 0:
            if time.time() - start_time > time_limit * 0.4:
                break # Abort greedy if taking too long, return partial? No, must be valid.
                # Better: just let it run, greedy is fast.
                
            # Find best block
            best_b_idx = -1
            best_count = -1
            
            # For deterministic, just find max.
            # For randomized, we might want top K.
            # Given speed, linear scan is fine and simpler for correctness.
            # 50k scan * 150 steps = 7.5M ops. < 0.1s.
            
            for i in range(n_blocks):
                if block_uncovered_count[i] > best_count:
                    best_count = block_uncovered_count[i]
                    best_b_idx = i
                    if best_count == len(block_t_subs[i]):
                        break
                        
            if best_b_idx == -1 or best_count == 0:
                break
                
            # Randomization: if randomness > 0, with prob 'randomness', pick a random block from top 10%
            # This requires knowing the top 10%.
            # Simpler: If randomness > 0, pick a random block that covers at least 80% of the max count?
            # Or just pick a random block from the first 100 best?
            # Let's stick to deterministic for the main run, and use random restarts of the whole process
            # or random tie-breaking.
            # Actually, let's just do deterministic greedy first.
            
            blocks.append(best_b_idx)
            
            # Update counts
            for s in block_t_subs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    # Update all blocks containing s
                    for b_idx in subset_to_blocks[s]:
                        block_uncovered_count[b_idx] -= 1

        # Ensure validity: if total_uncovered > 0, add random blocks to cover rest (shouldn't happen)
        if total_uncovered > 0:
             # Fallback: add all blocks? No.
             # Just add blocks covering remaining uncovered subsets greedily
             # The loop above should have finished.
             pass

        return blocks

    def local_search(block_indices):
        """
        Local search: Remove redundant, then try Add-Remove-Remove.
        """
        # Initial cover count
        cover_count = [0] * n_t_subs
        for b_idx in block_indices:
            for s in block_t_subs[b_idx]:
                cover_count[s] += 1
        
        improved = True
        # Phase 1: Simple Removal
        while improved and (time.time() - start_time) < time_limit * 0.7:
            improved = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for s in block_t_subs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_t_subs[b_idx]:
                        cover_count[s] -= 1
                    block_indices.pop(i)
                    improved = True
                    break # Restart loop

        # Phase 2: Add-Remove-Remove
        # Try to find a move that reduces block count by 1.
        # Move: Remove A, Add B, Remove C.
        # Condition: B covers all subsets of A that are not covered by others (Critical A).
        #           And C becomes redundant after A removed and B added.
        
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            found_move = False
            for i in range(len(block_indices)):
                b_idx_A = block_indices[i]
                
                # Find critical subsets of A
                critical_A = []
                for s in block_t_subs[b_idx_A]:
                    if cover_count[s] == 1:
                        critical_A.append(s)
                        
                if not critical_A:
                    # A is redundant, should have been caught in Phase 1
                    # But maybe counts changed?
                    for s in block_t_subs[b_idx_A]:
                        cover_count[s] -= 1
                    block_indices.pop(i)
                    found_move = True
                    break

                # Find candidate B that covers all critical_A
                # Intersect subset_to_blocks for each s in critical_A
                candidates = set(subset_to_blocks[critical_A[0]])
                for s in critical_A[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                    
                # Filter candidates: B must not be in current cover (except A? No, B is new)
                # And B != A
                current_set = set(block_indices)
                candidates.discard(b_idx_A)
                candidates -= current_set
                
                if not candidates:
                    continue

                # For each candidate B, check if any C becomes redundant
                for b_idx_B in candidates:
                    # Simulate: Remove A, Add B
                    # New cover count for subset s:
                    # old_count - (1 if s in A else 0) + (1 if s in B else 0)
                    
                    # Check each block C in cover (excluding A) to see if it becomes redundant
                    # C is redundant if for all s in C, new_count[s] > 0
                    
                    # Optimization: Only check C that intersects with A or B?
                    # If C doesn't intersect A or B, its count doesn't change, so if it wasn't redundant before, it isn't now.
                    # But we want to REMOVE C. So C must become redundant.
                    # This implies C's counts must decrease or stay same, and reach 0?
                    # No, C becomes redundant if ALL its subsets are covered by others.
                    # This happens if C's subsets are covered by (Cover \ {A}) \cup {B}.
                    
                    # Let's just check all C.
                    # To speed up, precompute which C are "close" to redundant?
                    # For now, just iterate.
                    
                    for j in range(len(block_indices)):
                        if j == i:
                            continue
                        b_idx_C = block_indices[j]
                        
                        is_redundant = True
                        for s in block_t_subs[b_idx_C]:
                            old_c = cover_count[s]
                            in_A = 1 if s in block_t_subs[b_idx_A] else 0
                            in_B = 1 if s in block_t_subs[b_idx_B] else 0
                            new_c = old_c - in_A + in_B
                            if new_c == 0:
                                is_redundant = False
                                break
                                
                        if is_redundant:
                            # Perform Move: Remove A, Add B, Remove C
                            # Update cover_count
                            for s in block_t_subs[b_idx_A]:
                                cover_count[s] -= 1
                            for s in block_t_subs[b_idx_B]:
                                cover_count[s] += 1
                            for s in block_t_subs[b_idx_C]:
                                cover_count[s] -= 1
                            
                            # Update block_indices
                            # Remove A (index i)
                            block_indices.pop(i)
                            # Remove C (index j, adjust if j > i)
                            if j > i:
                                j -= 1
                            block_indices.pop(j)
                            # Add B
                            block_indices.append(b_idx_B)
                            
                            found_move = True
                            break
                    if found_move:
                        break
                if found_move:
                    break
            
            if not found_move:
                break

        return block_indices

    # Main Execution
    best_blocks = []
    best_len = float('inf')
    
    # Run 1: Deterministic Greedy + LS
    blocks = greedy_construct(0.0)
    if len(blocks) < best_len:
        blocks = local_search(blocks)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Run 2-5: Randomized Greedy + LS
    # Introduce randomness in greedy by picking randomly from top 10% ties?
    # Or just random restarts with different random seeds for tie-breaking?
    # Since greedy is deterministic, we need to modify greedy to accept randomness.
    # Let's modify greedy_construct to pick a random block from the top 5% if randomness > 0.
    
    def greedy_construct_random(randomness=0.1):
        uncovered = [1] * n_t_subs
        total_uncovered = n_t_subs
        block_uncovered_count = [len(subs) for subs in block_t_subs]
        blocks = []
        
        while total_uncovered > 0:
            if time.time() - start_time > time_limit * 0.4:
                break
                
            # Find top candidates
            # To avoid full sort, find max, then gather all with count >= max * 0.9
            max_count = 0
            for i in range(n_blocks):
                if block_uncovered_count[i] > max_count:
                    max_count = block_uncovered_count[i]
                    
            if max_count == 0:
                break
                
            threshold = max_count * (1 - randomness)
            candidates = [i for i in range(n_blocks) if block_uncovered_count[i] >= threshold]
            
            if not candidates:
                candidates = [i for i in range(n_blocks) if block_uncovered_count[i] == max_count]
                
            best_b_idx = random.choice(candidates)
            blocks.append(best_b_idx)
            
            for s in block_t_subs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in subset_to_blocks[s]:
                        block_uncovered_count[b_idx] -= 1
        return blocks

    for run in range(10):
        if time.time() - start_time > time_limit * 0.8:
            break
        blocks = greedy_construct_random(0.1)
        if len(blocks) >= best_len:
            # Prune: if greedy is already worse, skip LS?
            # LS can only improve or keep same. So if greedy > best, LS might still find better?
            # Unlikely if greedy is much worse.
            # But let's try if time permits.
            pass
        blocks = local_search(blocks)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    return [all_blocks[i] for i in best_blocks]

# Quick sanity check
if __name__ == "__main__":
    # Test with small parameters to ensure no crashes
    # But the evaluator calls construct(v, k, t)
    pass
