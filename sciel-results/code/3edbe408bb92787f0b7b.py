import sys
from math import gcd
import time

def construct():
    MAX_M = 128
    
    nodes = []
    mod_to_indices = {}
    idx = 0
    for m in range(2, MAX_M + 1):
        for r in range(m):
            nodes.append((r, m))
            if m not in mod_to_indices:
                mod_to_indices[m] = []
            mod_to_indices[m].append(idx)
            idx += 1
            
    N = len(nodes)
    
    conflicts = [0] * N
    
    moduli_list = list(range(2, MAX_M + 1))
    num_moduli = len(moduli_list)
    
    for i_mod in range(num_moduli):
        m1 = moduli_list[i_mod]
        for j_mod in range(i_mod + 1, num_moduli):
            m2 = moduli_list[j_mod]
            g = gcd(m1, m2)
            
            inds1 = mod_to_indices[m1]
            inds2 = mod_to_indices[m2]
            
            buckets2 = {}
            for idx2 in inds2:
                r2, _ = nodes[idx2]
                key = r2 % g
                if key not in buckets2:
                    buckets2[key] = []
                buckets2[key].append(idx2)
                
            for idx1 in inds1:
                r1, _ = nodes[idx1]
                key = r1 % g
                if key in buckets2:
                    mask2 = 0
                    for idx2 in buckets2[key]:
                        mask2 |= (1 << idx2)
                    conflicts[idx1] |= mask2
                    for idx2 in buckets2[key]:
                        conflicts[idx2] |= (1 << idx1)
                        
    degrees = [c.bit_count() for c in conflicts]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    best_size = 11
    best_set = 0
    
    # Initialize best_set with the known solution
    p = 11
    known_moduli = [p * k for k in range(1, 12)]
    for i, m in enumerate(known_moduli):
        r = i
        # Find index
        # We can just construct the mask
        # (r, m) is at index ?
        # We need to map (r,m) to index
        # Let's create a map
        pass
        
    # Create map for known solution
    idx_map = {}
    for i, (r, m) in enumerate(nodes):
        idx_map[(r, m)] = i
        
    for i, m in enumerate(known_moduli):
        r = i
        idx = idx_map[(r, m)]
        best_set |= (1 << idx)
        
    start_time = time.time()
    TIME_LIMIT = 55
    
    def bb(candidates, current_size):
        nonlocal best_size, best_set
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                # We don't track the set in this simplified version easily without passing it
                # Let's pass the set mask
            return
            
        if current_size + candidates.bit_count() <= best_size:
            return
            
        # Pick v
        v = -1
        for v_idx in order:
            if (candidates >> v_idx) & 1:
                v = v_idx
                break
                
        if v == -1:
            return
            
        # Include v
        new_cand_include = candidates & ~(1 << v) & ~conflicts[v]
        bb(new_cand_include, current_size + 1)
        
        # Exclude v
        new_cand_exclude = candidates & ~(1 << v)
        bb(new_cand_exclude, current_size)
        
    # We need to track the set.
    # Let's modify bb to take the current set mask.
    
    def bb_set(candidates, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        if current_size + candidates.bit_count() <= best_size:
            return
            
        v = -1
        for v_idx in order:
            if (candidates >> v_idx) & 1:
                v = v_idx
                break
                
        if v == -1:
            return
            
        # Include
        new_cand_include = candidates & ~(1 << v) & ~conflicts[v]
        bb_set(new_cand_include, current_size + 1, current_set | (1 << v))
        
        # Exclude
        new_cand_exclude = candidates & ~(1 << v)
        bb_set(new_cand_exclude, current_size, current_set)
        
    initial_cand = (1 << N) - 1
    bb_set(initial_cand, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            r, m = nodes[i]
            result.append([r, m])
            
    return result

# EVOLVE-BLOCK-END
