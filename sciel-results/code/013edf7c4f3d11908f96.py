"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    D = 11
    # Generate all weight-4 vectors
    # Support: 4 indices
    # Signs: 4 signs (+/-1)
    # Total: C(11,4) * 16 = 330 * 16 = 5280 vectors
    
    vectors = []
    supports = list(itertools.combinations(range(D), 4))
    
    # Precompute sign patterns
    sign_patterns = []
    for mask in range(16):
        signs = tuple(1 if (mask & (1 << k)) else -1 for k in range(4))
        sign_patterns.append(signs)
        
    # Create list of (support, sign_pattern_index, vector_tuple)
    candidates = []
    for supp in supports:
        for pat_idx, signs in enumerate(sign_patterns):
            v = [0] * D
            for idx, sign in zip(supp, signs):
                v[idx] = sign
            candidates.append((supp, pat_idx, tuple(v)))
            
    N = len(candidates)
    
    # Build conflict graph
    # conflict[i] is a set of indices j such that i and j are incompatible
    # Incompatible if:
    # 1. Same support and same sign pattern (identical) - actually we only have one of each, so no self-conflict.
    #    But wait, we generated unique (support, pattern). So no identicals.
    # 2. Same support, different pattern:
    #    Dot product must be <= 2.
    #    If dot > 2, conflict.
    #    Dot = 4 - 2 * HammingDistance.
    #    Dot > 2 => 4 - 2d > 2 => 2 > 2d => d < 1 => d=0.
    #    So only identicals conflict. But we have unique patterns.
    #    So NO conflicts within same support?
    #    Let's re-verify.
    #    u, v on same support.
    #    u . v <= 2.
    #    If u != v, min Hamming dist is 1.
    #    Dot = 4 - 2(1) = 2.
    #    2 <= 2. OK.
    #    So all vectors on the same support are compatible!
    
    # 3. Different supports:
    #    Let S1, S2 be supports.
    #    If |S1 \cap S2| <= 2:
    #       Max dot is 2 (if they share 2 indices with same signs).
    #       2 <= 2. OK.
    #    If |S1 \cap S2| == 3:
    #       Let I be intersection.
    #       Dot = sum_{i in I} u_i v_i + 0 (other indices disjoint).
    #       We need Dot <= 2.
    #       Dot can be 3, 1, -1, -3.
    #       Conflict if Dot == 3.
    #       This happens if u_i == v_i for all i in I.
    #    If |S1 \cap S2| == 4:
    #       Impossible for distinct supports of size 4.
        
    # So conflicts only occur between vectors on supports with intersection size 3,
    # and only if they have identical signs on the intersection.
    
    # To speed up, we can group candidates by support.
    # And precompute for each support, which other supports are neighbors (intersection 3).
    
    support_to_indices = {s: [] for s in supports}
    for i, (supp, pat, v) in enumerate(candidates):
        support_to_indices[supp].append(i)
        
    # Precompute neighbor supports
    support_neighbors = {s: [] for s in supports}
    for i, s1 in enumerate(supports):
        for s2 in supports[i+1:]:
            inter = set(s1).intersection(s2)
            if len(inter) == 3:
                support_neighbors[s1].append(s2)
                support_neighbors[s2].append(s1)
                
    # For each pair of neighbor supports, determine which patterns conflict.
    # Conflict if signs on intersection match.
    # We can precompute a compatibility matrix between patterns of neighbor supports.
    # But there are many pairs.
    # Instead, we can do a greedy selection.
    
    # Greedy Independent Set
    # active: set of indices
    active = set(range(N))
    selected = []
    
    # To make it faster, we can iterate in a smart order.
    # Random shuffle might help.
    # Or, we can just iterate.
    
    # Optimization:
    # If we pick a vector, we remove all vectors that conflict with it.
    # Conflicts are:
    # 1. Vectors on same support? No.
    # 2. Vectors on neighbor supports with matching signs on intersection.
    
    # We can maintain for each support, the set of active pattern indices.
    # And for each pair of neighbor supports, we can check conflicts.
    
    # Simpler:
    # Just iterate through active candidates.
    # For each candidate, check if it conflicts with any ALREADY SELECTED candidate.
    # If not, select it.
    # This is O(N^2) worst case, but N=5280. 27M checks.
    # In Python, this might be too slow (120s).
    # We need to be smarter.
    
    # Let's use the support structure.
    # We process supports one by one.
    # For each support, we decide which patterns to pick.
    # This is what the previous code did, but it was flawed.
    # The flaw was that it didn't consider that picking a pattern might block patterns in FUTURE supports.
    # Greedy by support order is a heuristic.
    
    # Let's try a different heuristic:
    # Randomly shuffle the candidates.
    # Iterate.
    # For candidate c:
    #   Check if c conflicts with any selected candidate.
    #   To check fast, we store selected candidates in a structure.
    #   Selected candidates are grouped by support.
    #   For c on support S:
    #     Check neighbors of S.
    #     For each neighbor T:
    #       For each selected vector v on T:
    #         Check if signs on S \cap T match.
    
    # This is O(Degree(S) * Selected(T)) per candidate.
    # Degree(S) is ~ 15?
    # C(11,4) = 330.
    # Neighbors of S: choose 3 from 4, 1 from 7. 4*7 = 28 neighbors.
    # So ~28 neighbors.
    # Selected(T) is at most 16.
    # So ~28 * 16 = 448 checks per candidate.
    # 5280 candidates * 448 = 2.3M checks.
    # This is fast in Python.
    
    selected_by_support = {s: [] for s in supports}
    
    # Shuffle candidates for better greedy
    # Deterministic shuffle?
    # Let's just use the natural order or a fixed random seed.
    # To improve score, we might want to try multiple random orders and keep the best?
    # But we must finish in 120s.
    # One pass should be enough.
    
    # Let's try to pick vectors that have many conflicts first?
    # Or just random?
    # Let's use a random shuffle with a fixed seed for reproducibility.
    import random
    random.seed(42)
    indices = list(range(N))
    random.shuffle(indices)
    
    for idx in indices:
        supp, pat_idx, v = candidates[idx]
        
        # Check conflicts with already selected
        is_conflict = False
        for neighbor in support_neighbors[supp]:
            inter = set(supp).intersection(neighbor)
            # Map inter indices to positions in supp and neighbor
            # We need to compare signs on inter.
            # v is a tuple of length D.
            # For each selected vector on neighbor, check if signs match on inter.
            
            for sel_idx in selected_by_support[neighbor]:
                sel_v = candidates[sel_idx][2]
                # Check if v and sel_v have same signs on inter
                match = True
                for i in inter:
                    if v[i] != sel_v[i]:
                        match = False
                        break
                if match:
                    is_conflict = True
                    break
            if is_conflict:
                break
        
        if not is_conflict:
            selected.append(idx)
            selected_by_support[supp].append(idx)
            
    # Construct result
    if len(selected) > 110:
        result = np.array([candidates[i][2] for i in selected], dtype=np.int64)
        return result
    else:
        # Fallback to baseline
        baseline_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    baseline_vectors.append(v)
        return np.array(baseline_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
