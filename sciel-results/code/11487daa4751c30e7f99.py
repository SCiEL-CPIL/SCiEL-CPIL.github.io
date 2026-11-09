import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    d = 11
    best_points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    best_k = 2
    
    # Try different sparsities t
    for t in [7, 8, 9]:
        if time.time() > deadline - 10:
            break
            
        # Generate all t-sparse vectors
        vectors = []
        # Indices
        indices = list(range(d))
        
        for combo in itertools.combinations(indices, t):
            for signs in itertools.product([-1, 1], repeat=t):
                v = np.zeros(d, dtype=np.int64)
                for idx, sign in zip(combo, signs):
                    v[idx] = sign
                vectors.append(v)
        
        if not vectors:
            continue
            
        # Greedy matching
        # Slots are (combo of t-1 indices, signs)
        # We need to map each vector to its t slots
        # Slot key: (frozenset of indices, tuple of signs)
        
        used_slots = set()
        selected = []
        
        # Shuffle to break symmetry
        np.random.shuffle(vectors)
        
        for v in vectors:
            if time.time() > deadline - 5:
                break
                
            # Get non-zero indices and signs
            nz_idx = np.nonzero(v)[0]
            nz_signs = v[nz_idx]
            
            slots = []
            valid = True
            for i in range(t):
                # Exclude i-th index
                slot_idx = [j for j in nz_idx if j != nz_idx[i]]
                slot_signs = [s for j, s in zip(nz_idx, nz_signs) if j != nz_idx[i]]
                
                key = (tuple(sorted(slot_idx)), tuple(slot_signs))
                if key in used_slots:
                    valid = False
                    break
                slots.append(key)
            
            if valid:
                selected.append(v)
                for key in slots:
                    used_slots.add(key)
                    
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = np.array(selected, dtype=np.int64)
            
    return best_points
