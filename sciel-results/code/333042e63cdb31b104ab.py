import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_b = len(all_blocks)
    
    cover_indices = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        cover_indices.append(subs)
        
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, subs in enumerate(cover_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].append(b_idx)

    # Greedy Init
    blocks = []
    cover_counts = np.zeros(n_t, dtype=np.int32)
    uncovered_set = set(range(n_t))
    
    while uncovered_set and time.time() < deadline:
        # Pick random uncovered
        s_idx = random.choice(list(uncovered_set))
        candidates = t_to_blocks[s_idx]
        
        best_block = -1
        best_gain = -1
        for b_idx in candidates:
            gain = 0
            for s in cover_indices[b_idx]:
                if s in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1: break
        
        blocks.append(best_block)
        for s in cover_indices[best_block]:
            cover_counts[s] += 1
            if cover_counts[s] == 1:
                uncovered_set.discard(s)
                
    if time.time() > deadline:
        return [list(all_blocks[b]) for b in blocks]

    # SA
    current_blocks = blocks
    current_counts = cover_counts.copy()
    
    def can_remove(b_idx):
        for s in cover_indices[b_idx]:
            if current_counts[s] == 1:
                return False
        return True

    def do_remove(b_idx):
        for s in cover_indices[b_idx]:
            current_counts[s] -= 1
        # Remove from list
        # List removal is O(N), but N is small (~150)
        current_blocks.remove(b_idx)

    def do_add(b_idx):
        for s in cover_indices[b_idx]:
            current_counts[s] += 1
        current_blocks.append(b_idx)

    def try_swap(old_idx, new_idx):
        # Check if new_idx is already in current_blocks (duplicate)
        if new_idx in current_blocks:
            return False
            
        old_subs = cover_indices[old_idx]
        new_subs = cover_indices[new_idx]
        
        # Check validity
        for s in old_subs:
            if current_counts[s] == 1 and s not in new_subs:
                return False
        
        # Perform swap
        for s in old_subs:
            current_counts[s] -= 1
        for s in new_subs:
            current_counts[s] += 1
            
        # Update list
        # Find index of old_idx
        # Since we know it's there
        idx = current_blocks.index(old_idx)
        current_blocks[idx] = new_idx
        return True

    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # 1. Try Remove
        if len(current_blocks) > 0:
            b_idx = random.choice(current_blocks)
            if can_remove(b_idx):
                do_remove(b_idx)
                continue
        
        # 2. Try Swap
        if len(current_blocks) > 0:
            old_idx = random.choice(current_blocks)
            # Pick a random new block
            # To make it more effective, maybe pick a block that covers similar things?
            # Random is fine for now.
            new_idx = random.randint(0, n_b - 1)
            if try_swap(old_idx, new_idx):
                continue
                
        # 3. Try Add (SA)
        new_idx = random.randint(0, n_b - 1)
        if new_idx not in current_blocks:
            if random.random() < np.exp(-1.0 / T):
                do_add(new_idx)
                # Try to clean up
                for _ in range(20):
                    if len(current_blocks) == 0: break
                    b_idx = random.choice(current_blocks)
                    if can_remove(b_idx):
                        do_remove(b_idx)
                        break # Found one, restart loop
            T *= cooling
            if T < 0.01:
                T = 0.01

    return [list(all_blocks[b]) for b in current_blocks]
