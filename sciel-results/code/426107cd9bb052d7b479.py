import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # For each block, compute its mask of covered triples
    block_masks = [0] * num_blocks
    # For each triple, store list of block indices that cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            mask |= 1 << idx
            triple_to_blocks[idx].append(i)
        block_masks[i] = mask
    
    def greedy_cover(seed=None):
        """Fast greedy using rarest-uncovered-triple strategy."""
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_triples) - 1
        selected = []
        
        while uncovered:
            # Find the rarest uncovered triple
            best_triple = -1
            best_count = num_blocks + 1
            # Scan triples to find rarest uncovered one
            # This is O(num_triples) per step, but num_triples=2024 is small
            for ti in range(num_triples):
                if (uncovered >> ti) & 1:
                    cnt = len(triple_to_blocks[ti])
                    if cnt < best_count:
                        best_count = cnt
                        best_triple = ti
                    if cnt == 1:
                        break  # Can't do better than 1
            
            if best_triple == -1:
                break
            
            # Among blocks covering this triple, pick one with max new coverage
            candidates = triple_to_blocks[best_triple]
            best_gain = -1
            best_cands = []
            
            for bi in candidates:
                inter = uncovered & block_masks[bi]
                gain = inter.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [bi]
                elif gain == best_gain:
                    best_cands.append(bi)
            
            chosen = random.choice(best_cands)
            selected.append(chosen)
            uncovered &= ~block_masks[chosen]
        
        return selected if not uncovered else None
    
    def remove_redundant(block_indices):
        """Remove blocks that are redundant (all their triples covered by others)."""
        if not block_indices:
            return block_indices
        
        n = len(block_indices)
        # Build coverage count: for each triple, how many selected blocks cover it
        triple_cover_count = [0] * num_triples
        for bi in block_indices:
            mask = block_masks[bi]
            # Count bits set in mask
            m = mask
            while m:
                lsb = m & (-m)
                ti = lsb.bit_length() - 1
                triple_cover_count[ti] += 1
                m ^= lsb
        
        # Try removing each block
        kept = []
        for bi in block_indices:
            mask = block_masks[bi]
            # Check if every triple in this block is covered by >= 2 blocks
            redundant = True
            m = mask
            while m:
                lsb = m & (-m)
                ti = lsb.bit_length() - 1
                if triple_cover_count[ti] <= 1:
                    redundant = False
                    break
                m ^= lsb
            
            if redundant:
                # Remove this block: decrement counts for its triples
                m = mask
                while m:
                    lsb = m & (-m)
                    ti = lsb.bit_length() - 1
                    triple_cover_count[ti] -= 1
                    m ^= lsb
            else:
                kept.append(bi)
        
        return kept
    
    best_solution = None
    best_len = float('inf')
    
    # Run many fast greedy restarts
    it = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
        
        result = greedy_cover(seed=it * 7919 + 13)
        if result is not None:
            # Try redundancy removal
            pruned = remove_redundant(result)
            if len(pruned) < best_len:
                best_len = len(pruned)
                best_solution = pruned
        
        it += 1
    
    if best_solution is None:
        # Deterministic fallback
        result = greedy_cover(seed=42)
        if result is not None:
            best_solution = remove_redundant(result)
    
    if best_solution is None:
        return []
    
    return [list(blocks[i]) for i in best_solution]
