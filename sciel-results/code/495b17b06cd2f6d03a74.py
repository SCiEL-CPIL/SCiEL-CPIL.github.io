import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit
    
    # Generate all norm^2 = 2 vectors: two +/-1 entries, rest 0
    # Count: C(11,2) * 4 = 55 * 4 = 220
    norm2_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    norm2_vectors.append(v)
    
    # All 220 norm^2=2 vectors are mutually compatible:
    # For distinct u, v with norm^2=2: u.v <= 1 (max when they share one coordinate with same sign)
    # dist^2 = 2 + 2 - 2*u.v = 4 - 2*u.v >= 4 - 2*1 = 2 = max_norm^2
    # So all 220 form a valid set.
    
    base_set = norm2_vectors[:]
    base_size = len(base_set)
    
    # Now try to add norm^2=1 vectors (22 of them: +/-e_i for i=0..10)
    # A norm^2=1 vector u is compatible with a norm^2=2 vector v iff u.v <= 0
    # (since dist^2 = 1 + 2 - 2*u.v = 3 - 2*u.v >= 2 requires u.v <= 0.5, i.e., u.v <= 0 for integers)
    
    norm1_vectors = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        norm1_vectors.append(v)
        norm1_vectors.append(-v)
    
    # For each norm^2=1 vector, check compatibility with all norm^2=2 vectors
    # u (norm^2=1) is compatible with v (norm^2=2) iff u.v <= 0
    # 
    # e_i is compatible with v iff v[i] <= 0 (since u.v = v[i])
    # -e_i is compatible with v iff v[i] >= 0
    #
    # For a given coordinate i, among the 20 norm^2=2 vectors with a non-zero entry at position i:
    #   10 have +1 at position i, 10 have -1 at position i
    # So e_i conflicts with the 10 that have +1, and -e_i conflicts with the 10 that have -1.
    #
    # We want to select a subset S of norm^2=1 vectors such that for each v in norm2_vectors,
    # at most... wait, no. The constraint is pairwise: each selected norm^2=1 vector must be
    # compatible with EACH norm^2=2 vector. So if we select e_i, we need e_i.v <= 0 for ALL v in norm2_vectors.
    # But e_i.v = v[i], and there exist v with v[i] = 1. So e_i is NOT compatible with all norm^2=2 vectors!
    #
    # This means we CANNOT add any norm^2=1 vector to the full set of 220 norm^2=2 vectors.
    # The max norm^2 is 2, and we need dist^2 >= 2 for ALL pairs.
    # e_i and a norm^2=2 vector with v[i]=1: dist^2 = 1+2-2*1 = 1 < 2. INVALID.
    #
    # So the 220 norm^2=2 vectors are already maximal if we stick to this structure.
    #
    # Can we do better with a different construction?
    # What if we use ONLY norm^2=1 vectors? Max 22, worse.
    # What if we use a mix but drop some norm^2=2 vectors to add norm^2=1 vectors?
    #
    # Alternatively, can we find MORE than 220 vectors with norm^2=2 that are mutually compatible?
    # No - there are exactly 220 such vectors, and all are mutually compatible.
    #
    # What about using vectors with different norms?
    # If we include a vector with norm^2=3, then max_norm^2=3, and we need all pairwise dist^2 >= 3.
    # The 220 norm^2=2 vectors have pairwise dist^2 >= 2, but some have dist^2=2 < 3. INVALID.
    # So we can't mix norm^2=2 and norm^2=3 vectors easily.
    #
    # Let me try a pure norm^2=3 construction.
    # Norm^2=3 vectors: three +/-1 entries. Count: C(11,3)*8 = 165*8 = 1320.
    # For two such vectors u, v: u.v can range from -3 to 3.
    # dist^2 = 3+3-2*u.v = 6-2*u.v. Need >= 3, so u.v <= 1.5, i.e., u.v <= 1.
    # Conflict if u.v >= 2.
    #
    # This is a packing problem on 1320 vectors. Let's try greedy.
    
    best_set = base_set[:]
    best_size = base_size
    
    # Try norm^2=3 vectors with greedy MIS
    if time.time() < deadline - 100:
        norm3_vectors = []
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                v = np.zeros(d, dtype=np.int64)
                                v[i] = s1
                                v[j] = s2
                                v[k] = s3
                                norm3_vectors.append(v)
        
        n3 = len(norm3_vectors)
        # n3 = 1320
        # Compute conflict: u.v >= 2 means conflict
        # dist^2 = 6 - 2*u.v < 3 iff u.v > 1.5 iff u.v >= 2
        
        # Build conflict graph using dot products
        arr3 = np.array(norm3_vectors, dtype=np.int64)
        dots = arr3 @ arr3.T
        
        # Conflict if dots[i,j] >= 2 (for i != j)
        conflict = dots >= 2
        np.fill_diagonal(conflict, False)
        deg = conflict.sum(axis=1)
        
        # Greedy MIS: repeatedly pick min-degree node, remove it and neighbors
        for trial in range(50):
            if time.time() > deadline - 50:
                break
            
            available = np.ones(n3, dtype=bool)
            selected = []
            
            # Use a priority queue approach or just iterative min-degree
            # For speed, use numpy operations
            while available.any():
                avail_idx = np.where(available)[0]
                if len(avail_idx) == 0:
                    break
                
                # Get degrees of available nodes
                avail_deg = deg[avail_idx]
                min_deg = np.min(avail_deg)
                min_nodes = avail_idx[avail_deg == min_deg]
                
                if trial == 0:
                    pick = min_nodes[0]
                else:
                    pick = min_nodes[random.randint(0, len(min_nodes) - 1)]
                
                selected.append(pick)
                
                # Remove pick and its neighbors
                to_remove = conflict[pick] | (np.arange(n3) == pick)
                available[to_remove] = False
                
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [norm3_vectors[i] for i in selected]
    
    # Also try norm^2=4 vectors
    if time.time() < deadline - 100:
        norm4_vectors = []
        # Type 1: (2,0,...) -> 22 vectors
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2
            norm4_vectors.append(v)
            norm4_vectors.append(-v)
        # Type 2: (1,1,1,1,0,...) -> C(11,4)*16 = 330*16 = 5280 vectors
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for l in range(k + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    for s4 in [1, -1]:
                                        v = np.zeros(d, dtype=np.int64)
                                        v[i] = s1
                                        v[j] = s2
                                        v[k] = s3
                                        v[l] = s4
                                        norm4_vectors.append(v)
        
        n4 = len(norm4_vectors)
        # n4 = 22 + 5280 = 5302
        # For norm^2=4 vectors: dist^2 = 4+4-2*u.v = 8-2*u.v. Need >= 4, so u.v <= 2.
        # Conflict if u.v >= 3.
        
        arr4 = np.array(norm4_vectors, dtype=np.int64)
        dots4 = arr4 @ arr4.T
        
        conflict4 = dots4 >= 3
        np.fill_diagonal(conflict4, False)
        deg4 = conflict4.sum(axis=1)
        
        for trial in range(20):
            if time.time() > deadline - 50:
                break
            
            available = np.ones(n4, dtype=bool)
            selected = []
            
            while available.any():
                avail_idx = np.where(available)[0]
                if len(avail_idx) == 0:
                    break
                
                avail_deg = deg4[avail_idx]
                min_deg = np.min(avail_deg)
                min_nodes = avail_idx[avail_deg == min_deg]
                
                if trial == 0:
                    pick = min_nodes[0]
                else:
                    pick = min_nodes[random.randint(0, len(min_nodes) - 1)]
                
                selected.append(pick)
                
                to_remove = conflict4[pick] | (np.arange(n4) == pick)
                available[to_remove] = False
                
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [norm4_vectors[i] for i in selected]
    
    # Verify and return
    if best_size >= 2:
        result = np.array(best_set, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        
        # Verify
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = int(np.dot(result[i] - result[j], result[i] - result[j]))
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    # Fallback: return the 220 norm^2=2 vectors
    result = np.array(base_set, dtype=np.int64)
    return np.around(result).astype(np.int64)
