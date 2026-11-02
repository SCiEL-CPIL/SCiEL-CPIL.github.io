"""Covering design C(19,8,5): Set Covering with Bitset Optimization."""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    
    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_k = list(itertools.combinations(range(v), k))
    N_k = len(all_k)
    
    # Build incidence matrix: block_covers[i] is a bitset of t-subsets covered by block i
    # Use numpy uint64 arrays for bitsets
    N_words = (N_t + 63) // 64
    block_bitsets = np.zeros((N_k, N_words), dtype=np.uint64)
    
    for i, b in enumerate(all_k):
        for s in itertools.combinations(b, t):
            idx = t_idx[s]
            word_idx = idx // 64
            bit_idx = idx % 64
            block_bitsets[i, word_idx] |= (1 << bit_idx)
    
    # Helper: check if a t-subset is covered by any block in a set of blocks
    def is_covered(t_subset, blocks_set):
        idx = t_idx[t_subset]
        word_idx = idx // 64
        bit_idx = idx % 64
        for b in blocks_set:
            if block_bitsets[b, word_idx] & (1 << bit_idx):
                return True
        return False
    
    # Helper: get coverage count for each t-subset given a set of blocks
    def get_coverage(blocks_set):
        counts = np.zeros(N_t, dtype=np.int32)
        for b in blocks_set:
            bs = block_bitsets[b]
            for word_idx in range(N_words):
                word = bs[word_idx]
                while word:
                    bit = word & (-word)
                    bit_idx = int(np.log2(bit)) if bit > 0 else 0
                    word_idx_actual = word_idx
                    actual_idx = word_idx_actual * 64 + bit_idx
                    if actual_idx < N_t:
                        counts[actual_idx] += 1
                    word ^= bit
        return counts
    
    # More efficient: use direct counting
    def get_coverage_v2(blocks_set):
        counts = np.zeros(N_t, dtype=np.int32)
        for b in blocks_set:
            for s in itertools.combinations(all_k[b], t):
                counts[t_idx[s]] += 1
        return counts
    
    # Greedy set covering algorithm
    def greedy_cover(seed, deadline):
        random.seed(seed)
        uncovered = set(all_t)
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Find the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            best_uncovered_list = None
            
            # Sample blocks for speed
            sample_size = min(N_k, 2000)
            indices = random.sample(range(N_k), sample_size)
            
            for i in indices:
                gain = 0
                for s in itertools.combinations(all_k[i], t):
                    if s in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                    best_uncovered_list = [s for s in itertools.combinations(all_k[i], t) if s in uncovered]
            
            if best_block is None or best_gain <= 0:
                break
            
            blocks.append(best_block)
            for s in best_uncovered_list:
                uncovered.discard(s)
        
        return blocks
    
    # Try to remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        changed = True
        while changed:
            changed = False
            counts = get_coverage_v2(blocks)
            new_blocks = []
            for b in blocks:
                redundant = True
                for s in itertools.combinations(all_k[b], t):
                    if counts[t_idx[s]] == 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        return blocks
    
    # Local search: try swapping blocks
    def local_search(blocks, deadline):
        current_blocks = list(blocks)
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            counts = get_coverage_v2(current_blocks)
            improved = False
            
            # First pass: remove redundant blocks
            new_blocks = []
            for b in current_blocks:
                redundant = True
                for s in itertools.combinations(all_k[b], t):
                    if counts[t_idx[s]] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                improved = True
                no_improve = 0
                continue
            
            # Try to swap a block with another
            random.shuffle(current_blocks)
            for i, b in enumerate(current_blocks):
                if time.time() > deadline:
                    break
                
                # Try replacing block b with another block b'
                # that covers the same unique t-subsets and potentially enables more removals
                unique_to_b = [s for s in itertools.combinations(all_k[b], t) if counts[t_idx[s]] == 1]
                
                if not unique_to_b:
                    # Block is redundant
                    current_blocks.pop(i)
                    improved = True
                    no_improve = 0
                    break
                
                # Sample candidate blocks
                sample_size = min(N_k, 500)
                candidates = random.sample(range(N_k), sample_size)
                
                found = False
                for c in candidates:
                    if c in current_blocks:
                        continue
                    
                    # Check if candidate covers all unique t-subsets of b
                    covers_all = True
                    for s in unique_to_b:
                        if not all(x in all_k[c] for x in s):
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Try the swap
                        new_blocks = current_blocks.copy()
                        new_blocks[i] = c
                        new_counts = get_coverage_v2(new_blocks)
                        
                        # Check if this enables more redundancy removal
                        new_new_blocks = []
                        for nb in new_blocks:
                            redundant = True
                            for s in itertools.combinations(all_k[nb], t):
                                if new_counts[t_idx[s]] == 1:
                                    redundant = False
                                    break
                            if not redundant:
                                new_new_blocks.append(nb)
                        
                        if len(new_new_blocks) < len(current_blocks):
                            current_blocks = new_new_blocks
                            improved = True
                            no_improve = 0
                            found = True
                            break
                        elif len(new_new_blocks) == len(current_blocks):
                            # Same size, but might be better for future swaps
                            current_blocks = new_new_blocks
                            no_improve = 0
                            found = True
                            break
                
                if found:
                    break
            
            if not improved:
                no_improve += 1
        
        return current_blocks
    
    # Main algorithm
    best_blocks = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy restarts
    greedy_deadline = t0 + 40.0
    for seed in range(30):
        if time.time() > greedy_deadline:
            break
        try:
            blocks = greedy_cover(seed, greedy_deadline)
            if not blocks:
                continue
            blocks = remove_redundant(blocks)
            if blocks and len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks.copy()
        except Exception:
            pass
    
    if best_blocks is None:
        return []
    
    # Phase 2: Local search
    if time.time() < deadline - 10:
        try:
            ls_blocks = local_search(best_blocks, deadline - 10)
            ls_blocks = remove_redundant(ls_blocks)
            
            # Verify validity
            if ls_blocks:
                counts = get_coverage_v2(ls_blocks)
                if np.all(counts > 0):
                    if len(ls_blocks) < best_size:
                        best_blocks = ls_blocks
                        best_size = len(ls_blocks)
        except Exception:
            pass
    
    # Final cleanup
    best_blocks = remove_redundant(best_blocks)
    
    return [list(all_k[b]) for b in best_blocks]
