import time
import numpy as np
import itertools
import sys

D = 11
DEADLINE = 115.0  # Safety margin below 120s
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _check_deadline():
    return _elapsed() > DEADLINE

# EVOLVE-BLOCK-START

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Use the known kissing number for dimension 11, which is 512.
    We will construct a configuration of 512 vectors based on the E11 lattice
    structure or use a systematic approach that guarantees the kissing condition.
    
    For dimension 11, the kissing number is known to be 512.
    One well-known construction uses vectors of the form:
    - Type 1: (±2, 0, 0, ..., 0) - 2*11 = 22 vectors
    - Type 2: (±1, ±1, 0, ..., 0) with exactly two nonzero entries - 2^2 * C(11,2) = 4*55 = 220 vectors
    - Type 3: (±1, ±1, ±1, ±1, 0, ...) with exactly four nonzero entries - need to select carefully
    
    Actually, let me use a more systematic approach. The kissing number in dimension n
    for the E8 lattice and related constructions... 
    
    For n=11, a known construction gives 512 vectors. Let me use the approach of
    combining different types of vectors.
    
    Alternative: Use the fact that the maximum kissing number in dimension 11 is 512.
    I'll construct vectors based on the Leech lattice projection or use a greedy approach
    with known good candidates.
    
    Let me try a different approach: use vectors with coordinates in {-2, -1, 0, 1, 2}
    and systematically select those that satisfy the distance condition.
    
    Actually, the simplest robust approach: 
    The kissing number in dimension 11 is 512. Let me construct this explicitly.
    
    Known construction for kissing number 512 in R^11:
    This can be derived from the E8 lattice or from combinatorial designs.
    
    Let me use a constructive approach:
    - All vectors have norm^2 = 4 (i.e., 4 nonzero entries of ±1, or 1 entry of ±2, etc.)
    - We need min pairwise distance >= max norm = 2
    - So we need ||v - w||^2 >= 4 for all pairs
    - ||v - w||^2 = ||v||^2 + ||w||^2 - 2<v,w> = 8 - 2<v,w> (if both have norm^2=4)
    - Need 8 - 2<v,w> >= 4 => <v,w> <= 2
    
    So we need a set of 512 vectors, each with 4 nonzero coordinates of ±1,
    such that the dot product of any two is at most 2.
    
    Let me generate all possible such vectors and select 512 that work.
    
    Total number of vectors with 4 nonzero ±1 entries in R^11:
    C(11,4) * 2^4 = 330 * 16 = 5280
    
    We need to select 512 from these 5280 such that all pairwise dot products <= 2.
    
    This is equivalent to finding a code of length 11 over {-1, 0, 1} with weight 4,
    minimum Hamming distance... actually let me think in terms of the dot product constraint.
    
    Two vectors u and v with 4 nonzero entries:
    - If they have disjoint supports: <u,v> = 0
    - If they share k coordinates: <u,v> = sum of products on shared coords
      Each product is ±1, so <u,v> can range from -k to k in steps of 2.
    
    We need <u,v> <= 2 for all pairs.
    
    If supports are disjoint (k=0): OK
    If k=1: <u,v> = ±1, OK
    If k=2: <u,v> ∈ {-2, 0, 2}, OK (max is 2)
    If k=3: <u,v> ∈ {-3, -1, 1, 3}, need <= 2, so we can't have <u,v> = 3
    If k=4: <u,v> ∈ {-4, -2, 0, 2, 4}, need <= 2, so we can't have <u,v> = 4 or 2 is OK
    
    So the constraint is:
    - No two vectors can have the same support with all signs matching (dot=4)
    - No two vectors can have intersection of size 3 with all 3 signs matching (dot=3)
    
    Let me use a greedy approach with random restarts to find 512 such vectors.
    """
    
    if _check_deadline():
        # Fallback: return a small valid set
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Generate all candidate vectors: 4 nonzero entries of ±1
    candidates = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(v)
    
    n_candidates = len(candidates)  # 5280
    
    # Precompute: for each vector, its support (tuple of 4 coords) and signs
    supports = []
    sign_tuples = []
    for v in candidates:
        nz = tuple(i for i in range(D) if v[i] != 0)
        signs = tuple(v[i] for i in nz)
        supports.append(nz)
        sign_tuples.append(signs)
    
    # Group vectors by support
    from collections import defaultdict
    support_to_indices = defaultdict(list)
    for i in range(n_candidates):
        support_to_indices[supports[i]].append(i)
    
    # Build incompatibility: two vectors are incompatible if their dot product > 2
    # This happens when:
    # 1. Same support and same signs (dot = 4) - but these are the same vector, so skip self
    # 2. Intersection of size 3, and on those 3 coords, signs match (dot >= 3)
    # 3. Intersection of size 4 (same support), signs match on all 4 (dot = 4) - same as 1
    # 4. Intersection of size 4, signs match on 3 or 4? No, if same support and 3 match, dot could be 2 or -2 or... 
    #    If same support: dot = sum of 4 products, each ±1. Values: -4, -2, 0, 2, 4.
    #    We need dot <= 2, so dot = 4 is bad (all signs match), dot = 2 is OK.
    #    So for same support: only the identical vector is a problem (but we don't compare with self).
    #    Actually, distinct vectors with same support have at least one sign different, so dot <= 2.
    
    # So the only incompatibilities are:
    # - Same vector (skip)
    # - Vectors whose supports intersect in exactly 3 coords, and on those 3 coords, the signs match.
    
    # Let me build a conflict graph efficiently.
    # For each vector i, find all j != i that conflict with i.
    
    # Conflict if:
    # |S_i ∩ S_j| = 3 and for all c in S_i ∩ S_j, v_i[c] == v_j[c]
    
    # Let me precompute for efficiency.
    # For each 3-subset of coordinates, and each sign pattern on those 3 coords,
    # list which vectors have that 3-subset as part of their support with those signs.
    
    # Actually, let me just do a pairwise check with early termination.
    # 5280^2 / 2 = ~14 million pairs. Each check is fast.
    # But we can be smarter.
    
    # For each vector i with support S_i:
    #   For each 3-subset T of S_i:
    #     For each vector j with support S_j such that T ⊂ S_j and S_j ≠ S_i:
    #       Check if signs match on T.
    
    # Let me index vectors by their 3-subsets.
    # three_subset_to_indices: maps (3-tuple of coords, 3-tuple of signs) to list of vector indices
    
    three_subset_to_indices = defaultdict(list)
    for i in range(n_candidates):
        S = supports[i]
        sig = sign_tuples[i]
        # Get all 3-subsets of S and corresponding signs
        for k in range(4):
            # Remove S[k]
            sub_coords = tuple(S[j] for j in range(4) if j != k)
            sub_signs = tuple(sig[j] for j in range(4) if j != k)
            key = (sub_coords, sub_signs)
            three_subset_to_indices[key].append(i)
    
    # Now build adjacency list for conflicts
    conflict = [set() for _ in range(n_candidates)]
    
    for i in range(n_candidates):
        if _check_deadline():
            break
        S = supports[i]
        sig = sign_tuples[i]
        for k in range(4):
            sub_coords = tuple(S[j] for j in range(4) if j != k)
            sub_signs = tuple(sig[j] for j in range(4) if j != k)
            key = (sub_coords, sub_signs)
            for j in three_subset_to_indices[key]:
                if j == i:
                    continue
                # Check if S_j is different from S_i and |S_i ∩ S_j| = 3
                # S_j contains sub_coords and one more coord.
                # S_i contains sub_coords and S[k].
                # If S_j's extra coord != S[k], then |S_i ∩ S_j| = 3.
                # If S_j's extra coord == S[k], then S_j == S_i, and j has same signs on all 4,
                #   meaning j is the same as i (since signs determine the vector given support).
                #   So j == i in that case, which we skip.
                # Actually, if S_j == S_i and signs match on all 4, j == i.
                # If S_j == S_i and signs don't all match, then they don't match on the 4th coord,
                #   but we're looking at a specific 3-subset where they DO match.
                #   So j could be a different vector with same support.
                #   But for same support, distinct vectors are compatible (dot <= 2).
                #   So we only conflict if |S_i ∩ S_j| = 3 and signs match on those 3.
                
                # Determine if S_j != S_i:
                # S_j has 4 coords: 3 in sub_coords + 1 extra.
                # The extra coord is the one not in sub_coords that's in S_j.
                # We don't directly have S_j here, but we know j's support.
                if supports[j] == S:
                    # Same support. For distinct vectors with same support, they're compatible.
                    # So skip.
                    continue
                # |S_i ∩ S_j| = 3 (since S_j contains sub_coords and one other, and S_i contains sub_coords and S[k],
                # and S_j != S_i means the extra coord of S_j is not S[k])
                # Signs match on sub_coords by construction of the key.
                # So they conflict.
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Now we need to find a maximum independent set in the conflict graph.
    # The independent set = set of vectors with no conflicts = valid kissing configuration.
    # We want at least 512.
    
    # Use greedy with random restarts.
    best_size = 0
    best_set = []
    
    # Also try a simple greedy first
    def greedy_random(seed=42):
        rng = np.random.RandomState(seed)
        order = list(range(n_candidates))
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        
        for i in order:
            if i in selected_set:
                continue
            # Check if i conflicts with any selected vector
            ok = True
            for j in conflict[i]:
                if j in selected_set:
                    ok = False
                    break
            if ok:
                selected.append(i)
                selected_set.add(i)
        
        return selected
    
    # Try multiple random orders
    for seed in range(20):
        if _check_deadline():
            break
        result = greedy_random(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # If we have at least 512, great. Otherwise, try local improvement.
    if best_size < 512:
        # Try to add more vectors to the best set
        if best_set:
            current_set = set(best_set)
            # Try adding unselected vectors that don't conflict
            for i in range(n_candidates):
                if i in current_set:
                    continue
                ok = True
                for j in conflict[i]:
                    if j in current_set:
                        ok = False
                        break
                if ok:
                    current_set.add(i)
                    best_set.append(i)
                    best_size = len(best_set)
                    if best_size >= 512:
                        break
                if _check_deadline():
                    break
    
    # Try swapping: remove one, add two?
    # For now, let's also try a different greedy strategy:
    # Greedy by degree (pick lowest degree first)
    
    def greedy_degree():
        # Start with all candidates
        remaining = set(range(n_candidates))
        selected = []
        selected_set = set()
        
        while remaining and not _check_deadline():
            # Pick node with minimum degree in the remaining graph
            min_deg = float('inf')
            best_node = None
            for i in remaining:
                deg = len([j for j in conflict[i] if j in remaining])
                if deg < min_deg:
                    min_deg = deg
                    best_node = i
                if _check_deadline():
                    break
            if best_node is None:
                break
            
            # Select best_node
            selected.append(best_node)
            selected_set.add(best_node)
            remaining.discard(best_node)
            
            # Remove all neighbors of best_node from remaining
            for j in conflict[best_node]:
                remaining.discard(j)
        
        return selected
    
    result_deg = greedy_degree()
    if len(result_deg) > best_size:
        best_size = len(result_deg)
        best_set = result_deg
    
    # Build the result array
    if not best_set:
        # Fallback
        result_vec = np.array(candidates[:1], dtype=np.int64)
        return result_vec
    
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
