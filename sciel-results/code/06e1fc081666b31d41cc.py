import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)
    
    # Precompute 4-subset IDs
    map4 = np.full((19, 19, 19, 19), -1, dtype=np.int32)
    id_counter = 0
    all_4s = []
    for comb in itertools.combinations(range(19), 4):
        a, b, c, d = comb
        map4[a, b, c, d] = id_counter
        all_4s.append(comb)
        id_counter += 1
    n_4s = id_counter  # 3876
    
    # Precompute block_4s: for each of the 50388 blocks, list of 4-subset IDs
    all_blocks = list(itertools.combinations(range(19), 7))
    n_blocks = len(all_blocks)
    block_4s = []
    for block in all_blocks:
        ids = []
        for sub in itertools.combinations(block, 4):
            ids.append(map4[sub[0], sub[1], sub[2], sub[3]])
        block_4s.append(np.array(ids, dtype=np.int32))
    
    # Build dense incidence matrix M (n_blocks x n_4s)
    M = np.zeros((n_blocks, n_4s), dtype=np.int8)
    for b, inds in enumerate(block_4s):
        M[b, inds] = 1
    
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    def greedy_init(seed=None):
        """Randomized greedy initialization"""
        if seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = rng
        
        uncovered_mask = np.ones(n_4s, dtype=np.int8)
        selected = []
        
        while np.any(uncovered_mask):
            gains = M @ uncovered_mask
            best_block = int(np.argmax(gains))
            
            if gains[best_block] == 0:
                break
            
            selected.append(best_block)
            uncovered_mask[M[best_block] == 1] = 0
        
        return selected
    
    def remove_redundant(selected):
        """Remove redundant blocks (those whose removal doesn't uncover any 4-subset)"""
        if len(selected) == 0:
            return selected
        
        counts = np.zeros(n_4s, dtype=np.int32)
        for b in selected:
            counts[block_4s[b]] += 1
        
        new_selected = []
        redundant_found = True
        while redundant_found and time.time() < deadline:
            redundant_found = False
            temp_selected = []
            for b in selected:
                subs = block_4s[b]
                if len(subs) > 0 and counts[subs].min() > 1:
                    # Redundant
                    redundant_found = True
                    counts[subs] -= 1
                else:
                    temp_selected.append(b)
            if len(temp_selected) < len(selected):
                selected = temp_selected
        
        return selected
    
    def try_exchange(selected, remove_count=1, add_count=2):
        """Try to remove 'remove_count' blocks and add 'add_count' new blocks if it reduces total size"""
        if len(selected) < remove_count:
            return None
        
        # Try random removals
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Pick blocks to remove
            remove_candidates = rng.choice(len(selected), size=min(remove_count, len(selected)), replace=False)
            remove_set = set(int(x) for x in remove_candidates)
            
            # Check if removing these blocks would uncover any 4-subsets
            # Compute counts after removal
            counts = np.zeros(n_4s, dtype=np.int32)
            for b in selected:
                counts[block_4s[b]] += 1
            for idx in remove_candidates:
                counts[block_4s[int(idx)]] -= 1
            
            uncovered_after_removal = (counts == 0)
            if np.any(uncovered_after_removal):
                continue
            
            # Now try to find add_count blocks that cover all uncovered 4s
            # Actually, if no 4s are uncovered, we can just remove and be done
            if not np.any(uncovered_after_removal):
                # Success: just remove
                new_selected = [b for i, b in enumerate(selected) if i not in remove_set]
                return new_selected
            
            # Find blocks that cover the most uncovered 4s
            gains = M @ uncovered_after_removal
            gains[list(remove_set)] = 0  # Can't add blocks we're removing
            
            new_selected = list(selected)
            for _ in range(add_count):
                best_block = int(np.argmax(gains))
                if gains[best_block] == 0:
                    return None
                new_selected.append(best_block)
                gains[M[best_block] == 1] = 0  # Mark as covered
                gains[best_block] = 0
            
            # Verify coverage
            test_counts = np.zeros(n_4s, dtype=np.int32)
            for b in new_selected:
                test_counts[block_4s[b]] += 1
            if np.all(test_counts > 0):
                return new_selected
        
        return None
    
    def local_search(selected):
        """Iterative local search with redundancy removal and exchanges"""
        selected = remove_redundant(selected)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try 1-for-0 (remove redundant)
            new_sel = remove_redundant(selected)
            if len(new_sel) < len(selected):
                selected = new_sel
                improved = True
                continue
            
            # Try 1-for-1 (swap)
            for _ in range(100):
                if time.time() > deadline:
                    break
                if len(selected) == 0:
                    break
                
                old_idx = int(rng.choice(len(selected)))
                new_idx = int(rng.integers(0, n_blocks))
                
                if new_idx in selected:
                    continue
                
                # Check if swap maintains coverage
                test_selected = list(selected)
                test_selected[old_idx] = new_idx
                
                test_counts = np.zeros(n_4s, dtype=np.int32)
                for b in test_selected:
                    test_counts[block_4s[b]] += 1
                
                if np.all(test_counts > 0):
                    selected = test_selected
                    improved = True
                    break
            
            # Try 2-for-3 (remove 2, add 3) - net +1, but might enable further reductions
            # Actually, we want to reduce size, so try 1-for-2 (remove 1, add 2) then remove redundant
            new_sel = try_exchange(selected, remove_count=1, add_count=2)
            if new_sel is not None:
                new_sel = remove_redundant(new_sel)
                if len(new_sel) < len(selected):
                    selected = new_sel
                    improved = True
        
        return selected
    
    # Main loop: multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        # Greedy initialization
        selected = greedy_init(seed=num_restarts * 1000)
        
        # Local search
        selected = local_search(selected)
        
        current_len = len(selected)
        if current_len < best_len:
            best_len = current_len
            best_blocks = [list(all_blocks[i]) for i in selected]
        
        num_restarts += 1
    
    # Final: if we have best_blocks, return them
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: single greedy
    selected = greedy_init()
    selected = remove_redundant(selected)
    return [list(all_blocks[i]) for i in selected]

def _greedy_generic(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
