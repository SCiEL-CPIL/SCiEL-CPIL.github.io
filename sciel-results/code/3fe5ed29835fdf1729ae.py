import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115
    
    # Generate all weight-5 vectors
    # Support: 5-subset of range(11)
    # Signs: 5 bits
    supports = list(itertools.combinations(range(D), 5))
    n_supports = len(supports)
    
    # Precompute sign patterns
    # We want to maximize count.
    # Start with all 32 sign patterns.
    sign_patterns = list(itertools.product([1, -1], repeat=5))
    
    # Create list of all vectors
    # Represent vector as (support_idx, sign_idx)
    all_vectors = []
    for i, s in enumerate(supports):
        for j, sig in enumerate(sign_patterns):
            all_vectors.append((i, j))
            
    n_vectors = len(all_vectors)
    
    # Precompute dot products? No, too big.
    # We will compute on the fly.
    
    # Helper to get vector values
    # v[i] is support, v[j] is sign pattern
    # To compute dot product between (i1, j1) and (i2, j2):
    # s1 = supports[i1], s2 = supports[i2]
    # sig1 = sign_patterns[j1], sig2 = sign_patterns[j2]
    # Intersect s1, s2.
    # Sum sig1[k] * sig2[l] for matching positions.
    
    # To speed up, precompute intersection masks?
    # 462 supports. 462x462 intersections.
    # Store intersection size and mapping?
    
    # Let's use a greedy approach with random restarts.
    
    best_set = set()
    best_size = 0
    
    while time.time() < deadline:
        # Random order
        order = list(range(n_vectors))
        random.shuffle(order)
        
        current_set = set()
        
        for v_idx in order:
            if time.time() > deadline:
                break
                
            i, j = all_vectors[v_idx]
            s = supports[i]
            sig = sign_patterns[j]
            
            # Check conflict with all in current_set
            conflict = False
            for u_idx in current_set:
                k, l = all_vectors[u_idx]
                t = supports[k]
                sig_u = sign_patterns[l]
                
                # Compute dot product
                # Find common elements
                # Since supports are small, just loop
                dot = 0
                # Optimize: if intersection size < 3, no conflict possible for k=5?
                # If |S \cap T| <= 2, max dot is 2. Condition dot <= 2. OK.
                # So only check if intersection >= 3.
                
                # Quick check: are supports same?
                if i == k:
                    # Hamming distance
                    # sig and sig_u
                    # Count differences
                    diffs = 0
                    for a, b in zip(sig, sig_u):
                        if a != b:
                            diffs += 1
                    if diffs < 2:
                        conflict = True
                        break
                else:
                    # Check intersection
                    # Use set intersection
                    inter = set(s).intersection(t)
                    if len(inter) < 3:
                        continue
                    
                    # Compute dot product on intersection
                    # Map element to index in s and t
                    # This is slow.
                    # Precompute element to index for each support?
                    # Or just use dict?
                    
                    # Let's precompute a global element to index map? No, support specific.
                    
                    # For now, compute dot
                    dot = 0
                    for e in inter:
                        # Find index in s
                        idx_s = s.index(e)
                        idx_t = t.index(e)
                        dot += sig[idx_s] * sig_u[idx_t]
                    
                    if dot > 2:
                        conflict = True
                        break
            
            if not conflict:
                current_set.add(v_idx)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we got a very good result, we might stop early or continue
        if best_size > 7000:
             break

    # Build final array
    if not best_set:
        return np.array([], dtype=np.int64)
        
    vectors = []
    for v_idx in best_set:
        i, j = all_vectors[v_idx]
        s = supports[i]
        sig = sign_patterns[j]
        v = [0] * D
        for pos, sign in zip(s, sig):
            v[pos] = sign
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
