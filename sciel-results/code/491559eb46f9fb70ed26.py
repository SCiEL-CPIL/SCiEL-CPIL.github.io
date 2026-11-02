import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    # Precomputation
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks_all = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks_all)
    
    # block_to_triples: list of lists of triple indices
    block_to_triples = [[] for _ in range(num_blocks)]
    # triple_to_blocks: list of lists of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(blocks_all):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_to_triples[b_idx].append(t_idx)
            triple_to_blocks[t_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access? 
    # block_to_triples is a list of lists.
    # triple_to_blocks is a list of lists.
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            candidates = triple_to_blocks[start_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                trs = block_to_triples[b_idx]
                gain = np.sum(uncovered[trs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and random.random() < 0.5:
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            blocks.append(best_block_idx)
            
            for t_idx in block_to_triples[best_block_idx]:
                uncovered[t_idx] = False
                
        return blocks

    def local_search_sa(blocks_idx, seed):
        random.seed(seed)
        blocks_idx = list(blocks_idx)
        
        # Initialize counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in block_to_triples[b_idx]:
                counts[t_idx] += 1
        
        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        max_iterations = 50000
        iter_count = 0
        
        current_size = len(blocks_idx)
        best_local_size = current_size
        best_local_blocks = list(blocks_idx)
        
        while T > min_T and iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random block to modify
            if not blocks_idx:
                break
            i = random.randint(0, len(blocks_idx) - 1)
            b_old = blocks_idx[i]
            
            # Try to remove
            if random.random() < 0.3:
                # Check if removable
                # Essential triples are those with count 1
                trs = block_to_triples[b_old]
                # Vectorized check: if any count[trs] == 1
                # But we need to know which ones to update if we swap
                essential_mask = (counts[trs] == 1)
                if not np.any(essential_mask):
                    # Can remove
                    for t_idx in trs:
                        counts[t_idx] -= 1
                    blocks_idx.pop(i)
                    current_size -= 1
                    if current_size < best_local_size:
                        best_local_size = current_size
                        best_local_blocks = list(blocks_idx)
                else:
                    # Try to swap
                    # Find candidates covering all essential triples
                    # essential_trs = trs[essential_mask]
                    essential_trs = trs[essential_mask]
                    if len(essential_trs) == 0:
                        continue
                        
                    candidates = set(triple_to_blocks[essential_trs[0]])
                    for t_idx in essential_trs[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Remove old block from candidates if it's there?
                        # It is, but we are replacing it.
                        # We want a NEW block.
                        # If the only candidate is b_old, we can't swap to itself.
                        candidates.discard(b_old)
                        if candidates:
                            b_new = random.choice(list(candidates))
                            # Apply swap
                            for t_idx in trs:
                                counts[t_idx] -= 1
                            for t_idx in block_to_triples[b_new]:
                                counts[t_idx] += 1
                            
                            blocks_idx[i] = b_new
                            # Size unchanged
                            # Check for improvement? No, size is same.
                            # But maybe it allows future removals.
                            # In SA, we accept worse states. Here size is same.
                            # We should accept if it doesn't increase size.
                            # Actually, we can accept if it leads to a state where removal is possible later.
                            # For now, accept swap.
            else:
                # Try to swap
                trs = block_to_triples[b_old]
                essential_mask = (counts[trs] == 1)
                essential_trs = trs[essential_mask]
                
                if len(essential_trs) == 0:
                    # Can remove, but we chose swap branch.
                    # Maybe just remove?
                    # Let's stick to the 30% remove, 70% swap logic.
                    # If essential is empty, removal is strictly better (size -1).
                    for t_idx in trs:
                        counts[t_idx] -= 1
                    blocks_idx.pop(i)
                    current_size -= 1
                    if current_size < best_local_size:
                        best_local_size = current_size
                        best_local_blocks = list(blocks_idx)
                else:
                    candidates = set(triple_to_blocks[essential_trs[0]])
                    for t_idx in essential_trs[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        candidates.discard(b_old)
                        if candidates:
                            b_new = random.choice(list(candidates))
                            for t_idx in trs:
                                counts[t_idx] -= 1
                            for t_idx in block_to_triples[b_new]:
                                counts[t_idx] += 1
                            blocks_idx[i] = b_new
                            
            T *= cooling_rate
            
        return best_local_blocks

    # Run multiple attempts
    attempts = 0
    while time.time() < deadline:
        seed = attempts
        blocks_idx = greedy_cover(seed)
        blocks_idx = local_search_sa(blocks_idx, seed)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx
            
        attempts += 1

    if best_blocks is None:
        return []
        
    return [list(blocks_all[b_idx]) for b_idx in best_blocks]
