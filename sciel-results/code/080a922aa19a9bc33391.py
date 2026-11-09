"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    D = 11
    # Baseline: Weight 2 vectors
    baseline_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                baseline_vectors.append(v)
    baseline_count = len(baseline_vectors)
    
    # Strategy: Greedy selection of weight-4 vectors.
    # We can select up to 8 vectors per support (Max Independent Set of Q4).
    # Constraint: For any two selected vectors u on S, v on T:
    # if |S \cap T| == 3, they must not have identical signs on the intersection.
    # if |S \cap T| == 4, they must not be identical (handled by MIS within support).
    # if |S \cap T| <= 2, always compatible.
    
    supports = list(itertools.combinations(range(D), 4))
    
    # Precompute neighbors for each support (intersection size 3)
    # support -> list of (neighbor_support, intersection_indices, free_index_in_support, free_index_in_neighbor)
    # Actually, we just need the intersection set.
    support_neighbors = {s: [] for s in supports}
    for i, s1 in enumerate(supports):
        for s2 in supports[i+1:]:
            inter = set(s1).intersection(s2)
            if len(inter) == 3:
                support_neighbors[s1].append(s2)
                support_neighbors[s2].append(s1)
                
    # Precompute sign patterns for a 4-support
    # Map integer 0-15 to tuple of signs
    # Bit 0 -> index 0, Bit 1 -> index 1, etc.
    # 0 -> 1, 1 -> -1
    all_sign_patterns = []
    for mask in range(16):
        signs = tuple(1 if (mask & (1 << k)) else -1 for k in range(4))
        all_sign_patterns.append(signs)
        
    # Precompute Hamming distance / Dot product compatibility for patterns within a support
    # compatible[i][j] is True if pattern i and j are compatible (dot product <= 2)
    compatible = [[False]*16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            # Dot product of signs
            dot = sum(all_sign_patterns[i][k] * all_sign_patterns[j][k] for k in range(4))
            if dot <= 2:
                compatible[i][j] = True
                
    # Greedy selection
    selected_vectors = [] # List of (support_tuple, sign_pattern_index)
    
    # To speed up, we can process supports in a fixed order or shuffled.
    # Shuffling might help avoid bad local minima, but for deterministic output, fixed is fine.
    # Let's try a simple greedy: for each support, find all valid patterns, then find MIS.
    
    for supp in supports:
        supp_set = set(supp)
        # valid_patterns_mask: bitmask of 16 bits, 1 if valid
        valid_mask = (1 << 16) - 1
        
        # Check constraints from previously selected vectors on neighboring supports
        for neighbor in support_neighbors[supp]:
            # Find all selected vectors on this neighbor
            # We need to know which patterns are selected on 'neighbor'
            # Let's maintain a list of selected patterns per support
            pass # We need a data structure for selected patterns per support
            
    # Revised data structure
    # selected_patterns_on_support: dict support -> list of pattern indices
    selected_patterns_on_support = {s: [] for s in supports}
    
    for supp in supports:
        supp_set = set(supp)
        valid_mask = (1 << 16) - 1
        
        for neighbor in support_neighbors[supp]:
            # Get intersection
            inter = supp_set.intersection(set(neighbor))
            # inter has size 3
            # We need to map the 3 intersection indices to positions in supp (0-3) and neighbor (0-3)
            supp_inter_indices = [supp.index(idx) for idx in inter]
            neigh_inter_indices = [neighbor.index(idx) for idx in inter]
            
            # For each selected pattern on neighbor, determine which patterns on supp are forbidden
            for pat_idx in selected_patterns_on_support[neighbor]:
                # Get signs of neighbor pattern
                neigh_signs = all_sign_patterns[pat_idx]
                # Projection on intersection
                # We need to find patterns on supp that match these signs on the intersection positions
                
                # Instead of iterating all 16, we can calculate the forbidden mask directly.
                # A pattern on supp is forbidden if its signs on supp_inter_indices match neigh_signs on neigh_inter_indices.
                
                # Let's just iterate the 16 patterns for supp and check.
                # 16 * (number of selected on neighbor) * (number of neighbors)
                # Max selected on neighbor is 8.
                # Max neighbors is ~ C(11,4) * something.
                # This is fast enough.
                
                for k in range(16):
                    if not (valid_mask & (1 << k)):
                        continue
                    supp_signs = all_sign_patterns[k]
                    match = True
                    for i, idx_in_supp in enumerate(supp_inter_indices):
                        idx_in_neigh = neigh_inter_indices[i]
                        if supp_signs[idx_in_supp] != neigh_signs[idx_in_neigh]:
                            match = False
                            break
                    if match:
                        valid_mask &= ~(1 << k)
                        
        # Now find Max Independent Set in the subgraph of valid_mask
        # We want the largest subset of indices from valid_mask such that all pairs are compatible.
        # Since max size is 8, we can try to find a size 8 set first.
        # If not, size 7, etc.
        
        valid_indices = [i for i in range(16) if (valid_mask & (1 << i))]
        
        best_set = []
        # Try to find a set of size 8
        # We can use a simple backtracking or just check all combinations of 8?
        # C(16,8) is 12870. This is small.
        # But we only check if they are mutually compatible.
        
        # Optimization: The maximum independent sets of Q4 are the two parity codes.
        # Parity 0: masks with even number of 1s (if we map 1->0, -1->1? No, let's stick to signs).
        # Let's define parity based on number of -1s.
        # Even parity: 0, 2, 4 minus ones.
        # Odd parity: 1, 3 minus ones.
        
        parity_0 = [i for i in valid_indices if bin(all_sign_patterns[i].count(-1)).count('1') % 2 == 0]
        parity_1 = [i for i in valid_indices if bin(all_sign_patterns[i].count(-1)).count('1') % 2 == 1]
        
        # Check if parity_0 is fully compatible (it always is, as it's an MIS)
        # But we need to ensure all pairs in parity_0 are compatible. They are.
        # So if len(parity_0) == 8, we can pick it.
        # If len(parity_0) < 8, we might be able to pick a mix?
        # No, any MIS of size 8 in Q4 is a parity code.
        # If we can't pick a full parity code, we pick the largest subset.
        
        # Let's just greedily pick from valid_indices using the compatibility matrix
        # This is safer.
        
        chosen = []
        remaining = list(valid_indices)
        while remaining:
            # Pick the first one? Or the one with most conflicts?
            # Let's just pick the first one to be deterministic.
            cand = remaining[0]
            chosen.append(cand)
            # Remove all incompatible from remaining
            remaining = [r for r in remaining[1:] if compatible[cand][r]]
            
        if len(chosen) > 0:
            selected_patterns_on_support[supp] = chosen
            for pat_idx in chosen:
                # Convert to vector
                v = [0] * D
                signs = all_sign_patterns[pat_idx]
                for idx, sign in zip(supp, signs):
                    v[idx] = sign
                selected_vectors.append(v)
                
    # Compare with baseline
    if len(selected_vectors) > baseline_count:
        return np.array(selected_vectors, dtype=np.int64)
    else:
        return np.array(baseline_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
