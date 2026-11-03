import itertools
import numpy as np
import random

def construct():
    D = 11
    # Generate all 4-subsets of range(11)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Map support to index
    sup_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Precompute sign patterns for 4 indices
    # 16 patterns: 0..15
    # Represented as tuple of 4 signs
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    
    # Total vectors: n_sup * 16
    n_vec = n_sup * 16
    
    # Build conflict graph adjacency list
    # adj[i] is a set of indices j such that vector i and j conflict
    adj = [set() for _ in range(n_vec)]
    
    # For each pair of supports, check if they intersect in 3
    # If so, add conflicts between specific sign patterns
    
    # To speed up, group supports by their 3-subsets?
    # Or just iterate pairs. n_sup = 330. Pairs ~ 54k.
    # 54k * (check intersection) is fast.
    
    for i in range(n_sup):
        s_i = supports[i]
        set_i = set(s_i)
        for j in range(i + 1, n_sup):
            s_j = supports[j]
            set_j = set(s_j)
            inter = set_i.intersection(set_j)
            if len(inter) == 3:
                # Find the shared indices
                # We need to map the positions in s_i and s_j to the shared indices
                # s_i is a tuple of 4 indices.
                # Let's find the index of the non-shared element in s_i and s_j
                non_shared_i = None
                for k, idx in enumerate(s_i):
                    if idx not in inter:
                        non_shared_i = k
                        break
                non_shared_j = None
                for k, idx in enumerate(s_j):
                    if idx not in inter:
                        non_shared_j = k
                        break
                
                # The shared indices are the other 3.
                # We need to check sign patterns.
                # Two vectors conflict if their signs on the shared indices are identical.
                # Let's create a mapping from "signature on shared indices" to list of vector indices.
                
                # For support i:
                # A vector is defined by (sup_idx, sign_pattern_idx)
                # vec_idx = sup_idx * 16 + sign_pattern_idx
                
                # We can iterate over the 8 possible sign signatures on the 3 shared indices.
                # But it's easier to just iterate over the 16 patterns for i and 16 for j?
                # 16*16 = 256 checks per pair of supports.
                # 5280 pairs of supports (with inter 3) * 256 = 1.3M checks.
                # This is acceptable.
                
                for sp_i in range(16):
                    signs_i = sign_patterns[sp_i]
                    vec_i = i * 16 + sp_i
                    for sp_j in range(16):
                        signs_j = sign_patterns[sp_j]
                        vec_j = j * 16 + sp_j
                        
                        # Check if signs match on shared indices
                        # We need to align the signs to the shared indices.
                        # s_i[k] is the index in R^11.
                        # We need to compare signs_i[k] and signs_j[l] where s_i[k] == s_j[l] and s_i[k] in inter.
                        
                        # Since we know non_shared_i and non_shared_j, the other 3 positions correspond to the shared indices.
                        # Let's construct the list of shared signs for i and j.
                        
                        # Actually, we can just check all 3 shared indices.
                        conflict = True
                        # Iterate over the 3 shared indices
                        # We need to know which position in signs_i corresponds to which shared index.
                        # This is messy.
                        
                        # Alternative: Precompute for each support and each sign pattern, the "signature" on its 3-subsets?
                        # No, the shared subset depends on the pair.
                        
                        # Let's do it explicitly.
                        # Create a dict for s_i: index_in_tuple -> index_in_R11
                        # And for s_j.
                        
                        # Simpler:
                        # For each index in inter:
                        #   Find pos_i in s_i where s_i[pos_i] == idx
                        #   Find pos_j in s_j where s_j[pos_j] == idx
                        #   Check if signs_i[pos_i] == signs_j[pos_j]
                        
                        # To avoid doing this 256 times, we can precompute the mapping for the pair (i,j).
                        pass # Placeholder

    # This nested loop with explicit index finding inside is too slow if not optimized.
    # Let's optimize the conflict detection.
    
    # Precompute for each support i, and for each 3-subset of it, the mapping of positions.
    # Actually, let's just use a helper function and hope it's fast enough, or optimize the inner loop.
    
    # Let's rewrite the conflict building part to be faster.
    # We can precompute the "shared signature" for each vector relative to a specific 3-subset?
    # No, the 3-subset is the intersection.
    
    # Let's try a different approach:
    # For each vector v, it has a support S_v.
    # It conflicts with vector w if:
    # 1. S_v != S_w
    # 2. |S_v \cap S_w| == 3
    # 3. Signs match on S_v \cap S_w.
    
    # We can group vectors by their support.
    # For each pair of supports (S1, S2) with |S1 \cap S2| == 3:
    #   Let I = S1 \cap S2.
    #   For each vector v in S1:
    #     Let sig_v be the tuple of signs of v on indices in I.
    #     For each vector w in S2:
    #       Let sig_w be the tuple of signs of w on indices in I.
    #       If sig_v == sig_w, then v and w conflict.
    
    # To make this fast:
    # 1. Precompute for each support S, a list of its 16 vectors.
    # 2. For each pair of supports (S1, S2) with inter I:
    #    Create a dict for S1: sig -> list of vec_ids.
    #    Create a dict for S2: sig -> list of vec_ids.
    #    For each sig in dict1:
    #       if sig in dict2:
    #          for v in dict1[sig]:
    #             for w in dict2[sig]:
    #                adj[v].add(w)
    #                adj[w].add(v)
    
    # This is much faster.
    
    # Re-initialize adj
    adj = [set() for _ in range(n_vec)]
    
    # Precompute vectors for each support
    # vecs_by_sup[sup_idx] = list of (vec_id, sign_pattern)
    # We also need to compute the signature on a given 3-subset.
    
    # To compute signature on I for a vector (sup_idx, sp):
    #   I is a set of 3 indices.
    #   sup is a tuple of 4 indices.
    #   We need to extract the signs at the positions in sup that correspond to I.
    
    # Let's precompute for each support sup_idx, and for each of its 4 3-subsets, the mapping from sign_pattern to signature.
    # There are 4 such 3-subsets per support.
    # sig_maps[sup_idx][subset_idx] = { sp: signature_tuple }
    
    # Actually, we can just compute it on the fly for the pairs we visit.
    # Number of pairs with inter 3 is ~5280.
    # For each pair, we build two dicts of size 16.
    # 5280 * (16 + 16) ops is small.
    
    # Let's implement this.
    
    # First, identify all pairs of supports with intersection 3.
    conflict_sup_pairs = []
    for i in range(n_sup):
        s_i = supports[i]
        set_i = set(s_i)
        for j in range(i + 1, n_sup):
            s_j = supports[j]
            set_j = set(s_j)
            inter = set_i.intersection(set_j)
            if len(inter) == 3:
                conflict_sup_pairs.append((i, j, inter))
                
    # Now process each pair
    for i, j, inter in conflict_sup_pairs:
        s_i = supports[i]
        s_j = supports[j]
        
        # Build signature dicts
        # sig_dict_i: signature -> list of vec_ids
        sig_dict_i = {}
        for sp_i in range(16):
            signs_i = sign_patterns[sp_i]
            # Compute signature on inter
            # We need to map index in s_i to sign
            # Create a dict: R11_index -> sign
            sign_map_i = {s_i[k]: signs_i[k] for k in range(4)}
            sig_i = tuple(sign_map_i[idx] for idx in sorted(inter))
            vec_i = i * 16 + sp_i
            if sig_i not in sig_dict_i:
                sig_dict_i[sig_i] = []
            sig_dict_i[sig_i].append(vec_i)
            
        sig_dict_j = {}
        for sp_j in range(16):
            signs_j = sign_patterns[sp_j]
            sign_map_j = {s_j[k]: signs_j[k] for k in range(4)}
            sig_j = tuple(sign_map_j[idx] for idx in sorted(inter))
            vec_j = j * 16 + sp_j
            if sig_j not in sig_dict_j:
                sig_dict_j[sig_j] = []
            sig_dict_j[sig_j].append(vec_j)
            
        # Add edges
        for sig, list_i in sig_dict_i.items():
            if sig in sig_dict_j:
                list_j = sig_dict_j[sig]
                for v in list_i:
                    for w in list_j:
                        adj[v].add(w)
                        adj[w].add(v)
                        
    # Now we have the conflict graph.
    # Solve Max Independent Set using Simulated Annealing / SQAA heuristic.
    
    N = n_vec
    # Initialize state: all 1s
    x = np.ones(N, dtype=np.int8)
    
    # Calculate initial energy (number of conflicts)
    # We want to minimize conflicts, and maximize size.
    # Since we want a valid IS, we enforce conflicts == 0.
    # We use a penalty method: E = -sum(x) + L * conflicts.
    # L should be large.
    L = 1000
    
    def calc_conflicts(x):
        # Count number of edges (u,v) where x[u]=1 and x[v]=1
        c = 0
        for u in range(N):
            if x[u] == 1:
                for v in adj[u]:
                    if v > u and x[v] == 1:
                        c += 1
        return c
        
    # This is O(E) per call. Too slow for SA if called every step.
    # We need incremental updates.
    
    # Let's use a greedy + local search approach which is faster to implement correctly.
    # 1. Greedy: Start with empty set. Add vertices in random order if no conflict.
    # 2. Local Search:
    #    - Try to add any vertex not in set that has no neighbors in set.
    #    - If none, try to remove a vertex u from set, and add all its neighbors that are not in set and don't conflict with each other?
    #    - Simpler: 
    #      - Maintain set S.
    #      - Candidate set C = vertices not in S with no neighbors in S.
    #      - If C not empty, add one (or all?) to S.
    #      - If C empty, pick a vertex u in S. Remove u. 
    #        This might unlock some vertices.
    #        To avoid getting stuck, use SA:
    #        - With prob p, remove a random u from S.
    #        - With prob 1-p, add a random v from C (if C not empty).
    
    best_size = 0
    best_x = None
    
    for restart in range(50):
        x = np.zeros(N, dtype=np.int8)
        # Greedy init
        # Shuffle vertices
        indices = list(range(N))
        random.shuffle(indices)
        for v in indices:
            # Check if v conflicts with any in S
            conflict = False
            for u in adj[v]:
                if x[u] == 1:
                    conflict = True
                    break
            if not conflict:
                x[v] = 1
                
        current_size = np.sum(x)
        
        # Local Search
        T = 10.0
        for step in range(5000):
            # Try to add
            # Find candidates
            # To speed up, maintain a list of candidates?
            # For now, just pick a random vertex not in S and check.
            
            if random.random() < 0.5:
                # Try add
                v = random.randint(0, N-1)
                if x[v] == 0:
                    conflict = False
                    for u in adj[v]:
                        if x[u] == 1:
                            conflict = True
                            break
                    if not conflict:
                        x[v] = 1
                        current_size += 1
            else:
                # Try remove
                v = random.randint(0, N-1)
                if x[v] == 1:
                    # Accept with prob exp(-1/T) ? No, removing decreases size.
                    # We remove to escape local optima.
                    # Always accept removal? No, that will shrink the set.
                    # Accept with prob exp(-delta_E / T) where delta_E = 1 (size loss).
                    if random.random() < np.exp(-1.0 / T):
                        x[v] = 0
                        current_size -= 1
                        
            # Anneal
            T *= 0.995
            
        if current_size > best_size:
            best_size = current_size
            best_x = x.copy()
            
    # Return the vectors
    vectors = []
    for i in range(N):
        if best_x[i] == 1:
            sup_idx = i // 16
            sp_idx = i % 16
            sup = supports[sup_idx]
            signs = sign_patterns[sp_idx]
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
