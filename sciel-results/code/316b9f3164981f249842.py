import numpy as np
import itertools
import time
import random

D = 11
TARGET_NORM_SQ = 4  # Norm 2, vectors with four 1s

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate all candidate vectors with exactly 4 non-zero entries (+/-1)
    # Supports are combinations of 4 indices from 11
    supports = list(itertools.combinations(range(D), 4))
    candidates = []
    support_indices = []
    
    for s in supports:
        # Generate all 2^4 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(s, signs):
                v[idx] = val
            candidates.append(v)
            support_indices.append(s)
            
    n_vectors = len(candidates)
    
    # 2. Build conflict graph
    # Conflict if:
    # - Supports are different
    # - Intersection size is 3
    # - Signs on the intersection are identical
    
    # Precompute support to list of vector indices
    sup_to_vecs = {}
    for i, s in enumerate(support_indices):
        if s not in sup_to_vecs:
            sup_to_vecs[s] = []
        sup_to_vecs[s].append(i)
        
    # Map intersection of 3 indices to supports containing them?
    # Faster: For each pair of supports, check intersection.
    # But 330 supports is small. 330^2 = 108900 pairs.
    # We can precompute conflicts for each vector.
    
    adjacency = [set() for _ in range(n_vectors)]
    
    # Group supports by their 3-element subsets to find overlaps of size 3
    # For each 3-subset T of 11, find all supports containing T.
    # Any two vectors on supports containing T might conflict if signs on T match.
    
    # Actually, simpler:
    # Iterate through all pairs of supports.
    # If they share 3 indices, check vectors.
    
    # Let's optimize by iterating supports
    sup_list = list(sup_to_vecs.keys())
    
    for i in range(len(sup_list)):
        s1 = sup_list[i]
        vecs1 = sup_to_vecs[s1]
        for j in range(i + 1, len(sup_list)):
            s2 = sup_list[j]
            # Check intersection size
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                inter_tuple = tuple(sorted(inter))
                vecs2 = sup_to_vecs[s2]
                
                # For each v1 in vecs1, v2 in vecs2:
                # Check if signs on inter_tuple are same.
                for v1_idx in vecs1:
                    v1 = candidates[v1_idx]
                    for v2_idx in vecs2:
                        v2 = candidates[v2_idx]
                        # Check signs on intersection
                        match = True
                        for idx in inter:
                            if v1[idx] != v2[idx]:
                                match = False
                                break
                        if match:
                            adjacency[v1_idx].add(v2_idx)
                            adjacency[v2_idx].add(v1_idx)
        
    # 3. Greedy Independent Set with Random Restarts
    best_size = 0
    best_set = []
    
    def greedy_random():
        order = list(range(n_vectors))
        random.shuffle(order)
        selected = [False] * n_vectors
        count = 0
        for v_idx in order:
            if selected[v_idx]:
                continue
            # Check if any neighbor is selected
            has_conflict = False
            for n_idx in adjacency[v_idx]:
                if selected[n_idx]:
                    has_conflict = True
                    break
            if not has_conflict:
                selected[v_idx] = True
                count += 1
        return count, selected
        
    def local_search(selected):
        # Try to add nodes
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Find a node not selected
            unselected = [i for i in range(n_vectors) if not selected[i]]
            random.shuffle(unselected)
            for v_idx in unselected:
                # Check conflicts
                conflicting = []
                for n_idx in adjacency[v_idx]:
                    if selected[n_idx]:
                        conflicting.append(n_idx)
                
                if not conflicting:
                    selected[v_idx] = True
                    changed = True
                elif len(conflicting) == 1:
                    # Remove the one conflicting node and add this one?
                    # This maintains size, might help escape local minima?
                    # Actually, we want to maximize size.
                    # If we remove 1 and add 1, size is same.
                    # But it might allow adding more later?
                    # Let's try swapping if it doesn't reduce size.
                    # But simply adding is better.
                    # If we can't add, we are stuck.
                    # Try removing a random conflicting node and adding v_idx?
                    # This is a swap.
                    # If we do this, we might break other things? No, v_idx only conflicts with these.
                    # And the removed node might have allowed others?
                    # Let's just do simple add for now.
                    pass
        return selected

    num_restarts = 0
    while time.time() < deadline - 10:
        size, selected = greedy_random()
        if size > best_size:
            best_size = size
            best_set = [i for i in range(n_vectors) if selected[i]]
            # Try to improve with local search
            # Simple local search: try to add any unselected node
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                unselected = [i for i in range(n_vectors) if i not in set(best_set)]
                random.shuffle(unselected)
                for v_idx in unselected:
                    conflict = False
                    for n_idx in adjacency[v_idx]:
                        if n_idx in set(best_set): # Slow: O(N) check
                            conflict = True
                            break
                    if not conflict:
                        best_set.append(v_idx)
                        best_size += 1
                        improved = True
                        break # Restart check
        num_restarts += 1
        if num_restarts > 500: # Prevent infinite loop if very slow
             break

    # Convert best_set to array
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
