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
            # Pick a random uncovered triple
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Get candidate blocks containing this triple
            candidates = triple_to_blocks[start_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Evaluate gain for each candidate
            for b_idx in candidates:
                # Gain is number of uncovered triples in this block
                # block_to_triples[b_idx] is a list of triple indices
                trs = block_to_triples[b_idx]
                # Sum of uncovered[trs]
                # Vectorized: np.sum(uncovered[trs])
                gain = np.sum(uncovered[trs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and random.random() < 0.5:
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen
                break
                
            blocks.append(best_block_idx)
            
            # Update uncovered
            for t_idx in block_to_triples[best_block_idx]:
                uncovered[t_idx] = False
                
        return blocks

    def local_search(blocks_idx, seed):
        random.seed(seed)
        blocks_idx = list(blocks_idx)
        
        # Initialize counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in block_to_triples[b_idx]:
                counts[t_idx] += 1
        
        improved = True
        max_iterations = 1000 # Safety
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try to remove or swap blocks
            # Iterate in random order
            random.shuffle(blocks_idx)
            
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                
                # Find essential triples for this block
                # Essential if counts[t] == 1
                trs = block_to_triples[b_idx]
                essential_trs = [t_idx for t_idx in trs if counts[t_idx] == 1]
                
                if not essential_trs:
                    # Can remove
                    for t_idx in trs:
                        counts[t_idx] -= 1
                    blocks_idx.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Try to find a replacement block that covers all essential_trs
                    # Intersection of triple_to_blocks for each essential triple
                    # Start with blocks covering the first essential triple
                    candidates = set(triple_to_blocks[essential_trs[0]])
                    
                    for t_idx in essential_trs[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Pick a random candidate
                        # Prefer one that is not already in blocks_idx?
                        # If it is in blocks_idx, we can just remove b_idx?
                        # No, if b_new is in blocks_idx, then essential_trs would have count >= 2.
                        # So b_new must be new.
                        b_new = random.choice(list(candidates))
                        
                        # Replace b_idx with b_new
                        # Remove b_idx
                        for t_idx in trs:
                            counts[t_idx] -= 1
                        # Add b_new
                        for t_idx in block_to_triples[b_new]:
                            counts[t_idx] += 1
                        
                        blocks_idx[i] = b_new
                        improved = True
                        break # Restart loop
                        
        return blocks_idx

    # Run multiple attempts
    attempts = 0
    while time.time() < deadline:
        seed = attempts
        blocks_idx = greedy_cover(seed)
        blocks_idx = local_search(blocks_idx, seed)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx
            
        attempts += 1
        
        # If we find 70 or less, we can stop early? 
        # No, keep trying for 69.
        
    if best_blocks is None:
        # Fallback: return empty or a trivial covering?
        # Must return valid covering.
        # Greedy should always produce one.
        return []
        
    return [list(blocks_all[b_idx]) for b_idx in best_blocks]
