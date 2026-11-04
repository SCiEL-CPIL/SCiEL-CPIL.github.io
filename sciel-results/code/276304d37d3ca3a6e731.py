import numpy as np
import itertools
import time

D = 11

def construct():
    # Strategy: Use vectors of norm squared 4.
    # 1. Axis vectors: (±2, 0, ..., 0). Count = 22.
    # 2. Four-nonzero vectors: (±1, ±1, ±1, ±1, 0, ...). Count = 5280.
    # Condition for norm squared 4 vectors: dot product <= 2.
    
    # Generate axis vectors
    axis_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            axis_vectors.append(v)
            
    # Generate four-nonzero vectors
    four_nonzero = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            four_nonzero.append(v)
            
    four_nonzero = np.array(four_nonzero)
    N = len(four_nonzero)
    
    # Precompute compatibility for four-nonzero vectors
    # Compatible if dot product <= 2
    # We can compute dot products in batches or use matrix multiplication.
    # N=5280, N^2 = 27M. 27M int64 ops is fast in NumPy.
    
    # To save memory/time, we can process in chunks or just compute the full matrix if memory allows.
    # 5280 * 5280 * 8 bytes = ~220 MB. Acceptable.
    
    # Calculate dot products
    # Note: We only need to know if dot > 2.
    # We can compute D = four_nonzero @ four_nonzero.T
    
    # Optimization: We can filter vectors against axis vectors first?
    # Axis vectors are compatible with all 4-nonzero vectors (max dot 2).
    # So we just need max independent set in 4-nonzero graph.
    
    # Let's use a Bitset-based greedy for speed.
    # Represent compatibility as a list of bitsets.
    # Python integers can be used as bitsets.
    
    # Compute compatibility bitsets
    # For each vector i, compat[i] is a bitmask of all j such that dot(v_i, v_j) <= 2.
    
    # We can compute this in batches to avoid huge memory spikes if needed, 
    # but 220MB is fine.
    
    dots = four_nonzero @ four_nonzero.T
    
    # Create bitsets
    # We want to keep the current program's interface and return type.
    # The current program returns np.array(all_vectors).
    
    # Greedy with Bitsets
    # We try multiple random orders.
    
    best_selected_mask = np.zeros(N, dtype=bool)
    best_count = 0
    
    np.random.seed(42)
    
    for trial in range(10):
        order = np.random.permutation(N)
        selected = np.zeros(N, dtype=bool)
        # To speed up, we maintain a bitset of 'available' candidates?
        # Or just iterate and check against selected.
        
        # A faster greedy:
        # 1. Pick random start.
        # 2. Add all compatible neighbors? No, that's not greedy max independent set.
        
        # Standard Greedy:
        # Iterate through order. If v is compatible with all currently selected, add it.
        # Checking "compatible with all selected" is O(selected_size).
        # Total O(N * K).
        
        # Bitset optimization:
        # Maintain a bitset `selected_bits`.
        # For each candidate v, check if (selected_bits & conflict_bits[v]) == 0.
        # Where conflict_bits[v] is the set of vectors INCOMPATIBLE with v.
        
        # Precompute conflict_bits (incompatible set)
        # incompat[i] = bitmask of j where dot(v_i, v_j) > 2
        
        # Since N=5280, we can use Python ints as bitsets.
        
        # Precompute incompat masks
        # This takes time. 5280 * 5280 checks.
        # We can do this once.
        
        # Let's do it efficiently.
        # We already have `dots`.
        
        # Create a list of int bitmasks for incompatibility.
        # This might be slow in Python loop.
        
        # Alternative: Use NumPy to find indices where dots > 2.
        
        # For the sake of speed and simplicity within the time limit,
        # I will use a vectorized greedy check.
        
        selected_vecs = []
        selected_indices = []
        
        for i in order:
            v = four_nonzero[i]
            if not selected_vecs:
                selected_vecs.append(v)
                selected_indices.append(i)
                continue
                
            # Check against selected
            # Convert selected_vecs to array?
            # This is the bottleneck.
            
            # Let's try to optimize the inner loop.
            # If we have many selected, this is slow.
            
            # Heuristic: Only check against a subset? No, must be all.
            
            # Let's stick to the bitset idea but implement it carefully.
            # We need `conflict_mask[i]` as a Python int.
            
            # Precomputation step (once per construct call, or cached?)
            # Since construct is called once, we can spend up to 120s.
            
            # Let's precompute conflict masks.
            # This is expensive. 5280^2 = 27M comparisons.
            # In Python, this is slow.
            
            # I will use a different strategy:
            # Random restarts with a faster check.
            
            # Check:
            # dots_selected = four_nonzero[selected_indices] @ v
            # if np.all(dots_selected <= 2):
            #     add
            
            # This is vectorized!
            
            if selected_indices:
                sel_arr = four_nonzero[selected_indices]
                dots_sel = sel_arr @ v
                if np.all(dots_sel <= 2):
                    selected_vecs.append(v)
                    selected_indices.append(i)
            else:
                selected_vecs.append(v)
                selected_indices.append(i)
                
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_selected_mask = np.zeros(N, dtype=bool)
            best_selected_mask[selected_indices] = True
            
    # Combine
    all_vectors = axis_vectors + [four_nonzero[i] for i in np.where(best_selected_mask)[0]]
    
    return np.array(all_vectors)
