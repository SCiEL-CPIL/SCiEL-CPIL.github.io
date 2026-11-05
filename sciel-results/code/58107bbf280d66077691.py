import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the set of triplet IDs it covers
    block_triplets = []
    block_triplets_set = []
    for block in all_blocks:
        tids = [triplet_to_id[triplet] for triplet in itertools.combinations(block, t)]
        block_triplets.append(tids)
        block_triplets_set.append(set(tids))
        
    # Map each triplet to the list of block IDs that cover it
    blocks_covering = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            blocks_covering[tid].append(b_idx)

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        score = np.array([len(tids) for tids in block_triplets], dtype=np.int32)
        used_blocks = np.zeros(n_blocks, dtype=bool)
        solution = []
        
        while True:
            if time.time() > deadline - 60:
                break
                
            masked_score = score.copy()
            masked_score[used_blocks] = -1
            max_score = np.max(masked_score)
            
            if max_score <= 0:
                break
                
            candidates = np.where(masked_score == max_score)[0]
            chosen_block = rng.choice(candidates)
            
            solution.append(all_blocks[chosen_block])
            used_blocks[chosen_block] = True
            
            newly_covered = []
            for tid in block_triplets[chosen_block]:
                if covered_count[tid] == 0:
                    newly_covered.append(tid)
                covered_count[tid] += 1
                
            for tid in newly_covered:
                for b_idx in blocks_covering[tid]:
                    score[b_idx] -= 1
                    
            if np.all(covered_count > 0):
                break
                
        return solution

    def local_search_remove(blocks):
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] += 1
                
        new_blocks = []
        changed = True
        while changed:
            changed = False
            next_blocks = []
            for block in blocks:
                redundant = True
                for triplet in itertools.combinations(block, t):
                    tid = triplet_to_id[triplet]
                    if covered_count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    for triplet in itertools.combinations(block, t):
                        tid = triplet_to_id[triplet]
                        covered_count[tid] -= 1
                    changed = True
                else:
                    next_blocks.append(block)
            blocks = next_blocks
        return blocks

    def simulated_annealing(initial_blocks, seed=0):
        rng = random.Random(seed)
        current_blocks = list(initial_blocks)
        current_size = len(current_blocks)
        
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        for block in current_blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] += 1
        
        best_blocks = list(current_blocks)
        best_size = current_size
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < deadline - 5:
            if T < min_T:
                break
                
            # Move: Remove-Add-Remove
            idx_out = rng.randint(0, current_size - 1)
            block_out = current_blocks[idx_out]
            
            del current_blocks[idx_out]
            
            newly_uncovered = []
            for triplet in itertools.combinations(block_out, t):
                tid = triplet_to_id[triplet]
                covered_count[tid] -= 1
                if covered_count[tid] == 0:
                    newly_uncovered.append(tid)
            
            if len(newly_uncovered) == 0:
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = list(current_blocks)
                T *= cooling
                continue
            
            # Find best block to add
            newly_uncovered_set = set(newly_uncovered)
            candidate_blocks = set()
            for tid in newly_uncovered:
                candidate_blocks.update(blocks_covering[tid])
            
            best_in_score = -1
            best_in_candidates = []
            
            for b_idx in candidate_blocks:
                s = len(block_triplets_set[b_idx].intersection(newly_uncovered_set))
                if s > best_in_score:
                    best_in_score = s
                    best_in_candidates = [b_idx]
                elif s == best_in_score:
                    best_in_candidates.append(b_idx)
            
            if best_in_candidates:
                best_in_idx = rng.choice(best_in_candidates)
                
                # Add it
                current_blocks.append(all_blocks[best_in_idx])
                current_size += 1
                
                for triplet in itertools.combinations(all_blocks[best_in_idx], t):
                    tid = triplet_to_id[triplet]
                    covered_count[tid] += 1
                
                # Try to remove another block
                if current_size > 1:
                    idx_out2 = rng.randint(0, current_size - 1)
                    # Avoid removing the one we just added? Or allow it?
                    # If we remove the one we just added, we revert.
                    # Let's pick a different one if possible.
                    if current_size > 2:
                        while idx_out2 == len(current_blocks) - 1:
                            idx_out2 = rng.randint(0, current_size - 1)
                    else:
                        idx_out2 = 0
                    
                    block_out2 = current_blocks[idx_out2]
                    del current_blocks[idx_out2]
                    
                    newly_uncovered2 = []
                    for triplet in itertools.combinations(block_out2, t):
                        tid = triplet_to_id[triplet]
                        covered_count[tid] -= 1
                        if covered_count[tid] == 0:
                            newly_uncovered2.append(tid)
                    
                    if len(newly_uncovered2) == 0:
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_blocks = list(current_blocks)
                        T *= cooling
                        continue
                    else:
                        # Revert second removal
                        current_blocks.insert(idx_out2, block_out2)
                        current_size += 1
                        for triplet in itertools.combinations(block_out2, t):
                            tid = triplet_to_id[triplet]
                            covered_count[tid] += 1
                
                # If we are here, we have a valid cover of same size.
                # Accept with probability 1 (neutral move)
                T *= cooling
            else:
                # Should not happen if cover was valid before
                current_blocks.insert(idx_out, block_out)
                for triplet in itertools.combinations(block_out, t):
                    tid = triplet_to_id[triplet]
                    covered_count[tid] += 1
                T *= cooling
                
        return best_blocks

    best_blocks = None
    best_count = float('inf')
    
    num_seeds = 10
    for seed in range(num_seeds):
        if time.time() > deadline - 70:
            break
        blocks = greedy_set_cover(seed)
        blocks = local_search_remove(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = local_search_remove(greedy_set_cover(0))
        
    if time.time() < deadline - 10:
        sa_blocks = simulated_annealing(best_blocks, seed=42)
        if len(sa_blocks) < len(best_blocks):
            best_blocks = sa_blocks
            
    return best_blocks

if __name__ == "__main__":
    pass
