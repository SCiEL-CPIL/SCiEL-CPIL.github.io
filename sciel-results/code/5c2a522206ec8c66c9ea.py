import heapq
import itertools
import time
import random

_cache = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map triple to index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # For each block, list of triple indices it covers
    block_triple_indices = []
    # For each triple, list of block indices covering it
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for b_idx, b in enumerate(all_blocks):
        tris = list(itertools.combinations(b, t))
        t_indices = [triple_to_idx[tr] for tr in tris]
        block_triple_indices.append(t_indices)
        for t_idx in t_indices:
            triple_to_blocks[t_idx].append(b_idx)
            
    _cache[key] = (all_blocks, block_triple_indices, triple_to_blocks, all_triples)
    return _cache[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_blocks, block_triple_indices, triple_to_blocks, all_triples = _get_precomputed(v, k, t)
    n_blocks = len(all_blocks)
    n_triples = len(all_triples)
    
    best_cover = None
    best_size = float('inf')
    
    # Run multiple iterations with random tie-breaking in greedy
    # and local search refinement
    max_iterations = 10 # Adjust based on time
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy Set Cover
        uncovered = [True] * n_triples
        gain = [0] * n_blocks
        for b_idx in range(n_blocks):
            gain[b_idx] = len(block_triple_indices[b_idx])
            
        heap = []
        for b_idx in range(n_blocks):
            # Add random noise to break ties for diversity
            noise = random.random() * 0.01
            heapq.heappush(heap, (-(gain[b_idx] - noise), b_idx))
            
        selected = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 1.0:
                break
                
            neg_gain, b_idx = heapq.heappop(heap)
            cur_gain = -neg_gain
            # Re-calculate exact gain to handle stale/noise issues correctly
            # Actually, with noise, we just need to check if it's still a good candidate
            # But standard lazy heap: check if gain matches current uncovered count
            real_gain = 0
            for t_idx in block_triple_indices[b_idx]:
                if uncovered[t_idx]:
                    real_gain += 1
            
            if real_gain == 0:
                break
            if b_idx in selected_set:
                continue
            
            # Select block
            selected.append(b_idx)
            selected_set.add(b_idx)
            
            # Update uncovered and gains
            for t_idx in block_triple_indices[b_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrement gain for all blocks covering this triple
                    for other_b in triple_to_blocks[t_idx]:
                        if other_b != b_idx and other_b not in selected_set:
                            gain[other_b] -= 1
                            heapq.heappush(heap, (-gain[other_b], other_b))
                            
            # Check if done
            # Optimization: track number of uncovered triples
            # But 'any(uncovered)' is O(N). N=2024. Acceptable.
            if not any(uncovered):
                break
                
        # If greedy didn't finish, skip this iteration
        if any(uncovered):
            continue
            
        # 2. Local Search: Peel Redundant Blocks
        cover_list = list(selected)
        cover_set = set(cover_list)
        
        # cover_count[t] = number of blocks in cover covering triple t
        cover_count = [0] * n_triples
        for b_idx in cover_list:
            for t_idx in block_triple_indices[b_idx]:
                cover_count[t_idx] += 1
                
        # Peel loop
        changed = True
        while changed:
            changed = False
            new_cover_list = []
            for b_idx in cover_list:
                is_redundant = True
                for t_idx in block_triple_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove b_idx
                    for t_idx in block_triple_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
                else:
                    new_cover_list.append(b_idx)
            cover_list = new_cover_list
            cover_set = set(cover_list)
            
        # 3. Local Search: Block Replacement
        # Try to replace blocks to enable further peeling
        # We iterate until no swap improves (or just a fixed number of passes)
        swap_passes = 0
        max_swap_passes = 50
        
        while swap_passes < max_swap_passes and time.time() < deadline - 2.0:
            swap_passes += 1
            made_swap = False
            new_cover_list = []
            
            # Iterate over a copy or handle removals carefully
            # We process blocks one by one. If we swap, we update counts.
            # If we don't swap, we keep it.
            
            for b_idx in cover_list:
                if time.time() > deadline - 2.0:
                    break
                
                # Find critical triples for b_idx
                critical_triples = []
                for t_idx in block_triple_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        critical_triples.append(t_idx)
                        
                if not critical_triples:
                    # Redundant? Should have been peeled, but maybe counts changed?
                    # If no critical triples, it's redundant.
                    for t_idx in block_triple_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    made_swap = True # Treat as removal
                    continue
                
                # Try to find a replacement block b_new
                # It must cover all critical_triples
                # Optimization: Pick the first critical triple, look at blocks covering it
                first_crit = critical_triples[0]
                candidates = triple_to_blocks[first_crit]
                
                found_replacement = False
                for b_new in candidates:
                    if b_new in cover_set:
                        continue
                    
                    # Check if b_new covers all critical_triples
                    valid = True
                    # Optimization: Check intersection
                    # Since critical_triples is small, just loop
                    for ct in critical_triples:
                        # Check if ct is in block_triple_indices[b_new]
                        # This is O(1) if we had a set, but we have lists.
                        # block_triple_indices[b_new] has C(7,3)=35 items.
                        # Linear search is fine.
                        if ct not in block_triple_indices[b_new]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        # 1. Remove b_idx
                        for t_idx in block_triple_indices[b_idx]:
                            cover_count[t_idx] -= 1
                        cover_set.remove(b_idx)
                        
                        # 2. Add b_new
                        for t_idx in block_triple_indices[b_new]:
                            cover_count[t_idx] += 1
                        cover_set.add(b_new)
                        
                        new_cover_list.append(b_new)
                        found_replacement = True
                        made_swap = True
                        break
                if not found_replacement:
                    new_cover_list.append(b_idx)
            
            cover_list = new_cover_list
            
            # After a pass of swaps, re-peel
            # We can integrate peeling into the next iteration or do it here
            # Let's do a quick peel
            changed_peel = True
            while changed_peel and time.time() < deadline - 2.0:
                changed_peel = False
                temp_cover_list = []
                for b_idx in cover_list:
                    is_redundant = True
                    for t_idx in block_triple_indices[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for t_idx in block_triple_indices[b_idx]:
                            cover_count[t_idx] -= 1
                        changed_peel = True
                    else:
                        temp_cover_list.append(b_idx)
                cover_list = temp_cover_list
                cover_set = set(cover_list)
                
            if not made_swap:
                break
                
        # Update best
        if len(cover_list) < best_size:
            best_size = len(cover_list)
            best_cover = [list(all_blocks[i]) for i in cover_list]
            
    # If no valid cover found (should not happen), return empty or raise
    if best_cover is None:
        # Fallback: return a trivially valid but large cover? 
        # Greedy always works. If it timed out, we are in trouble.
        # But we checked 'any(uncovered)' before local search.
        # If greedy timed out, we skipped it.
        # If we never found a valid one, we must return something.
        # The problem guarantees a solution exists.
        # Let's try a simple greedy one more time without local search if we have time?
        # Or just return an empty list? No, must be valid.
        # If best_cover is None, it means greedy never completed within time.
        # This is unlikely for C(24,7,3).
        return []
        
    return best_cover
