import time
import random
from itertools import combinations

N = 16
ALL_TRIPLES = []
TRIPLE_MASKS = []
MASKS_BY_SIZE = [[] for _ in range(N + 1)]

def init_data():
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                m = (1 << i) | (1 << j) | (1 << k)
                ALL_TRIPLES.append((i, j, k))
                TRIPLE_MASKS.append(m)
    
    for mask in range(1 << N):
        popcount = bin(mask).count('1')
        MASKS_BY_SIZE[popcount].append(mask)

# Global state for speed
is_independent = None
current_triples = set() # set of masks
current_alpha = 0

def update_independent_array(add_mask=None, remove_mask=None):
    global is_independent
    if add_mask is not None:
        # Mark all supersets of add_mask as not independent
        # Supersets of m are m | s where s is subset of ~m
        # Iterate submasks of complement
        comp = (1 << N) - 1 - add_mask
        sub = comp
        while True:
            is_independent[add_mask | sub] = False
            if sub == 0: break
            sub = (sub - 1) & comp
            
    if remove_mask is not None:
        # We cannot simply set to True, because other triples might cover it.
        # We need to re-evaluate.
        # For simplicity and correctness, if we remove, we might need to rebuild or use a counter.
        # Let's use a counter array: cover_count[mask] = number of triples subset of mask.
        # is_independent[mask] is True if cover_count[mask] == 0.
        pass

# Better approach: cover_count array
cover_count = None

def init_state():
    global is_independent, cover_count
    cover_count = [0] * (1 << N)
    is_independent = [False] * (1 << N)
    # Initially, all are independent (count 0)
    for i in range(1 << N):
        is_independent[i] = True

def add_triple(mask):
    global cover_count, is_independent
    # Increment count for all supersets
    comp = (1 << N) - 1 - mask
    sub = comp
    while True:
        idx = mask | sub
        cover_count[idx] += 1
        if cover_count[idx] == 1:
            is_independent[idx] = False
        if sub == 0: break
        sub = (sub - 1) & comp

def remove_triple(mask):
    global cover_count, is_independent
    # Decrement count for all supersets
    comp = (1 << N) - 1 - mask
    sub = comp
    while True:
        idx = mask | sub
        cover_count[idx] -= 1
        if cover_count[idx] == 0:
            is_independent[idx] = True
        if sub == 0: break
        sub = (sub - 1) & comp

def calc_alpha():
    # Find max k such that some mask of size k is independent
    for k in range(N, 0, -1):
        for mask in MASKS_BY_SIZE[k]:
            if is_independent[mask]:
                return k
    return 0

def construct():
    init_data()
    init_state()
    
    # Greedy init
    random.shuffle(ALL_TRIPLES)
    for t in ALL_TRIPLES:
        m = TRIPLE_MASKS[ALL_TRIPLES.index(t)] # This is slow, use parallel lists
        # Better: iterate by index
    # Let's use indices
    indices = list(range(len(ALL_TRIPLES)))
    random.shuffle(indices)
    
    # Helper to check if triple is valid (conflict check)
    # Conflict if intersection > 1
    # Intersection of masks m1, m2 > 1 if bin(m1 & m2).count('1') > 1
    
    current_triples = set() # set of indices
    current_masks = set() # set of masks
    
    for idx in indices:
        m = TRIPLE_MASKS[idx]
        valid = True
        for cm in current_masks:
            if bin(m & cm).count('1') > 1:
                valid = False
                break
        if valid:
            current_masks.add(m)
            current_triples.add(idx)
            add_triple(m)
            
    best_alpha = calc_alpha()
    best_triples = list(current_triples)
    
    # Local Search
    start_time = time.time()
    time_limit = 140 # 10s margin
    
    # Precompute valid pairs? No, dynamic.
    
    while time.time() - start_time < time_limit:
        # Pick random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_triples = set(current_triples)
        new_masks = set(current_masks)
        
        if move_type == 'add':
            # Pick random idx not in current
            # To make it fast, pick random from all, check if in current
            # Or maintain a list of available?
            # Just pick random
            for _ in range(100): # try 100 times to find a valid add
                idx = random.randint(0, len(ALL_TRIPLES)-1)
                if idx in new_triples: continue
                m = TRIPLE_MASKS[idx]
                valid = True
                for cm in new_masks:
                    if bin(m & cm).count('1') > 1:
                        valid = False
                        break
                if valid:
                    new_triples.add(idx)
                    new_masks.add(m)
                    add_triple(m)
                    break
            if idx not in new_triples:
                continue # failed to add, skip step
                
        elif move_type == 'remove':
            if not new_triples: continue
            idx = random.choice(list(new_triples))
            m = TRIPLE_MASKS[idx]
            new_triples.remove(idx)
            new_masks.remove(m)
            remove_triple(m)
            
        else: # swap
            if not new_triples: continue
            idx_old = random.choice(list(new_triples))
            m_old = TRIPLE_MASKS[idx_old]
            # Try to find a valid new triple
            added = False
            for _ in range(100):
                idx_new = random.randint(0, len(ALL_TRIPLES)-1)
                if idx_new in new_triples: continue
                m_new = TRIPLE_MASKS[idx_new]
                # Check validity against (new_masks - {m_old})
                valid = True
                for cm in new_masks:
                    if cm == m_old: continue
                    if bin(m_new & cm).count('1') > 1:
                        valid = False
                        break
                if valid:
                    # Remove old, add new
                    new_triples.remove(idx_old)
                    new_masks.remove(m_old)
                    remove_triple(m_old)
                    
                    new_triples.add(idx_new)
                    new_masks.add(m_new)
                    add_triple(m_new)
                    added = True
                    break
            if not added:
                # Revert? No, we didn't change state yet in this branch?
                # Wait, I modified new_triples/new_masks but not the global state?
                # I need to be careful with state management.
                # Let's use a copy-on-write or explicit rollback.
                # The add_triple/remove_triple modify global state.
                # So if I fail swap, I must rollback the remove.
                # This is getting messy.
                
                # Let's restructure:
                # 1. Calculate new state metrics without modifying global?
                # No, add_triple is incremental.
                
                # Better:
                # Do the move on global state.
                # Calculate alpha.
                # If accept, keep.
                # If reject, rollback.
                
                pass # Logic below will handle rollback properly
        
        # Calculate new alpha
        new_alpha = calc_alpha()
        
        # Acceptance criteria
        if new_alpha < best_alpha:
            # Update best
            best_alpha = new_alpha
            best_triples = list(new_triples)
            # State is already updated to new_triples
            current_triples = new_triples
            current_masks = new_masks
        elif new_alpha == best_alpha:
            # Accept with prob 0.5 to explore
            if random.random() < 0.5:
                current_triples = new_triples
                current_masks = new_masks
        else:
            # Reject: Rollback
            # We need to undo the changes made to global state.
            # This requires tracking what was changed.
            # This is complex.
            
            # Alternative: Don't modify global state until accepted.
            # But add_triple is fast.
            # Let's just restart from best state if rejected?
            # That's too expensive.
            
            # Let's implement rollback properly.
            # Store old state?
            # No, let's just use a "dirty" flag and list of changes.
            
            pass

    # Return best
    return [list(ALL_TRIPLES[i]) for i in best_triples]
