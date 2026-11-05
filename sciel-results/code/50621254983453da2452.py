import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitsets and triple indices for each block
    block_bitsets = [0] * num_blocks
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        bitset = 0
        tris_in_block = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_bitsets[i] = bitset
        block_triple_indices[i] = tris_in_block
    
    # Greedy construction with bitset acceleration
    def greedy_construct(alpha=0.0, seed=None):
        rng = random.Random(seed)
        uncovered = (1 << num_triples) - 1
        selected = []
        selected_set = set()
        
        while uncovered and time.time() < deadline - 10:
            best_gain = -1
            best_indices = []
            min_gain = float('inf')
            
            # Scan for best gain
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (block_bitsets[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                    min_gain = gain
                elif gain == best_gain:
                    best_indices.append(i)
                if gain < min_gain:
                    min_gain = gain
            
            if best_gain <= 0:
                break
            
            if alpha == 0.0:
                chosen = best_indices[0] if best_indices else 0
            else:
                threshold = best_gain - alpha * (best_gain - min_gain)
                candidates = [i for i in best_indices]
                if not candidates:
                    # Re-scan for candidates above threshold
                    for i in range(num_blocks):
                        if i in selected_set:
                            continue
                        gain = (block_bitsets[i] & uncovered).bit_count()
                        if gain >= threshold:
                            candidates.append(i)
                if not candidates:
                    candidates = best_indices
                chosen = rng.choice(candidates)
            
            selected.append(chosen)
            selected_set.add(chosen)
            uncovered &= ~block_bitsets[chosen]
        
        return selected
    
    # Local search: removal of redundant blocks and 1-swap
    def local_search(selected_blocks, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5
        
        sol = list(selected_blocks)
        sol_set = set(sol)
        
        # Compute coverage counts
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove redundant blocks
            i = 0
            while i < len(sol):
                idx = sol[i]
                is_redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                else:
                    i += 1
            
            if time.time() > time_limit:
                break
            
            # Pass 2: 1-Swap
            for i, idx in enumerate(sol):
                if time.time() > time_limit:
                    break
                
                # Find unique triples for this block
                unique_tris = []
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Find a block that covers all unique triples
                # Start with the triple having fewest covering blocks
                min_len = float('inf')
                min_tri = -1
                for tri_idx in unique_tris:
                    if len(triple_to_blocks[tri_idx]) < min_len:
                        min_len = len(triple_to_blocks[tri_idx])
                        min_tri = tri_idx
                
                candidates = triple_to_blocks[min_tri]
                valid_candidate = -1
                for cand_idx in candidates:
                    if cand_idx in sol_set:
                        continue
                    # Check if cand covers all unique_tris
                    cand_bitset = block_bitsets[cand_idx]
                    covers_all = True
                    for tri_idx in unique_tris:
                        if not (cand_bitset & (1 << tri_idx)):
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidate = cand_idx
                        break
                
                if valid_candidate != -1:
                    # Perform swap
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    sol_set.discard(idx)
                    
                    sol[i] = valid_candidate
                    sol_set.add(valid_candidate)
                    for tri_idx in block_triple_indices[valid_candidate]:
                        counts[tri_idx] += 1
                    
                    improved = True
                    break
        
        return sol
    
    best_blocks_list = None
    best_size = num_blocks
    
    # Run multiple starts with different alphas
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 8:
                break
            
            seed = int(time.time() * 1000) + alpha * 1000 + attempt
            sol = greedy_construct(alpha, seed)
            if time.time() > deadline - 5:
                # Just check validity of what we have
                if len(sol) < best_size:
                    # Quick validity check
                    covered = 0
                    for idx in sol:
                        covered |= block_bitsets[idx]
                    if covered == (1 << num_triples) - 1:
                        best_size = len(sol)
                        best_blocks_list = [all_blocks[i] for i in sol]
                continue
            
            sol = local_search(sol)
            if len(sol) < best_size:
                # Verify validity
                covered = 0
                for idx in sol:
                    covered |= block_bitsets[idx]
                if covered == (1 << num_triples) - 1:
                    best_size = len(sol)
                    best_blocks_list = [all_blocks[i] for i in sol]
    
    if best_blocks_list is None:
        # Fallback: use all blocks (guaranteed valid)
        best_blocks_list = all_blocks
        best_size = num_blocks
    
    return best_blocks_list
