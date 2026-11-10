import itertools
import time
import random
import heapq

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using lazy greedy with bitsets
    and aggressive local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Precomputation
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        block_masks[b_idx] = mask
        
    # blocks_containing for local search
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            blocks_containing[idx].append(b_idx)
            m ^= lsb

    def greedy_lazy(rng, deadline):
        """
        Lazy Greedy Set Cover.
        """
        uncovered = (1 << n_t) - 1
        counts = [0] * n_t
        selected = []
        
        # Heap: (-gain, block_idx)
        heap = []
        # Initialize heap with all blocks
        # This is O(N log N). 346k * log(346k) ~ 6M ops. ~1s.
        # We can just push them all.
        for b_idx in range(n_blocks):
            gain = block_masks[b_idx].bit_count() # Initial gain is total coverage
            heapq.heappush(heap, (-gain, b_idx))
            
        # But gain changes as uncovered changes.
        # We will re-calculate gain when popping.
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Pop until we find a valid block
            best_block = -1
            best_gain = 0
            best_mask = 0
            
            while heap:
                neg_gain, b_idx = heapq.heappop(heap)
                # Recalculate actual gain
                actual_gain = (block_masks[b_idx] & uncovered).bit_count()
                if actual_gain == -neg_gain:
                    best_block = b_idx
                    best_gain = actual_gain
                    best_mask = block_masks[b_idx]
                    break
                else:
                    # Push back with updated gain
                    if actual_gain > 0:
                        heapq.heappush(heap, (-actual_gain, b_idx))
                    # If actual_gain == 0, it's useless for now, discard
            
            if best_block == -1:
                return None
                
            selected.append(best_block)
            uncovered &= ~best_mask
            
            # Update counts
            m = best_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
            if not uncovered:
                break

        return selected

    def local_search(blocks, deadline):
        """
        Local search with removal and swapping.
        """
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        counts = [0] * n_t
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Phase 1: Removal
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    return blocks
                    
                b = blocks[i]
                mask = block_masks[b]
                
                unique = False
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique = True
                        break
                    m ^= lsb
                
                if not unique:
                    blocks.pop(i)
                    block_set.discard(b)
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    improved = True
            
            # Phase 2: Swapping (Limited)
            if time.time() > deadline - 5:
                break
                
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                    
                b = blocks[i]
                mask = block_masks[b]
                
                unique_triples = []
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_triples.append(idx)
                    m ^= lsb
                
                if not unique_triples:
                    continue
                
                candidates = set(blocks_containing[unique_triples[0]])
                candidates.discard(b)
                
                for t in unique_triples[1:]:
                    candidates &= set(blocks_containing[t])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c = random.choice(list(candidates))
                if c in block_set:
                    continue
                
                # Swap
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                block_set.discard(b)
                
                new_mask = block_masks[c]
                m = new_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
                block_set.add(c)
                
                blocks[i] = c
                improved = True
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)
        
        blocks = greedy_lazy(rng, deadline)
        
        if blocks is None:
            continue
            
        blocks = local_search(blocks, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            if best_len <= 70:
                break

    if best_blocks is None:
        blocks = greedy_lazy(random.Random(0), deadline)
        if blocks:
            best_blocks = blocks
        else:
            return []

    return [all_blocks[i] for i in best_blocks]
