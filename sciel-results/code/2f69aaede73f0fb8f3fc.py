import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Time budget
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for evaluation overhead

    # Generate all weight-4 vectors with coordinates in {+1, -1} on their support
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # For each support, there are 16 sign combinations (2^4)
    # Total vectors: 330 * 16 = 5280
    # Each vector is represented by (support_idx, sign_idx)
    
    # Build the conflict graph:
    # Two vectors conflict (distance < max_norm) if their dot product constraint is violated.
    # For unit-norm normalized vectors, min pairwise distance >= max norm means:
    #   ||v_i - v_j|| >= ||v_k|| for all i,j,k
    # Since we normalize to norm 2, this means ||v_i - v_j|| >= 2 for all i != j
    # Which means v_i . v_j <= 0 for all i != j (for unit vectors: 2 - 2*v_i.v_j >= 4 => v_i.v_j <= -1)
    # Wait, let me recalculate.
    # Original vectors have norm sqrt(4) = 2.
    # After normalizing to norm 2, they stay the same.
    # Condition: min_{i!=j} ||v_i - v_j|| >= max_k ||v_k|| = 2
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # Need: 8 - 2*v_i.v_j >= 4 => v_i.v_j <= 2
    # For integer vectors with 4 non-zero coords of +/-1:
    # v_i . v_j = sum over common support positions of product of signs
    # If supports are disjoint: dot = 0, OK
    # If 3 common: dot = +/-1, +/-3, OK (<=2)
    # If 4 common: dot = -4,-2,0,2,4
    #   dot=4: NOT OK (4 > 2)
    #   dot=2: OK
    # So conflicts only when supports are identical AND dot product = 4 (same signs)
    # Actually wait: if same support, dot can be -4,-2,0,2,4
    # dot=4 means same signs -> same vector -> distance 0 -> conflict
    # But we only generate each vector once, so no duplicates.
    
    # Let me re-examine: with support overlap of 3:
    # 3 positions contribute +/-1 each, 1 position is 0 for one vector
    # dot = sum of 3 terms, each +/-1, so dot in {-3,-1,1,3}
    # All <= 2, so no conflict.
    
    # With support overlap of 4 (same support):
    # dot in {-4,-2,0,2,4}
    # dot=4 only if same signs -> same vector, won't happen
    # dot=2: OK
    # dot=0: OK
    # So actually NO conflicts between distinct weight-4 +/-1 vectors!
    
    # Wait, that can't be right. Let me recheck the problem.
    # The task says "minimum pairwise distance is at least the maximum vector norm"
    # For weight-4 vectors with entries +/-1, norm = 2
    # Distance between two such vectors:
    # If same support, different signs: distance^2 = sum of (a-b)^2 where a,b in {-1,1}
    #   If they differ in k positions: distance^2 = 4k for k=1,2,3,4 (since (1-(-1))^2=4)
    #   k=1: dist^2=4, dist=2 >= 2 OK
    #   k=2: dist^2=8, dist=2sqrt(2) OK
    #   k=3: dist^2=12 OK
    #   k=4: dist^2=16 OK
    # If different supports with 3 overlap:
    #   3 positions: a,b in {-1,1}, contribution 0 or 4
    #   1 position: one is 0, other is +/-1, contribution 1
    #   dist^2 = 4*(# differing signs in overlap) + 1
    #   Min: 1 (if same signs in all 3 overlap positions)
    #   1 < 4, so dist = 1 < 2 -> CONFLICT!
    
    # So conflicts DO exist. Let me redo this carefully.
    # v_i and v_j: supports S_i, S_j
    # dist^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # Need dist^2 >= 4, i.e., 8 - 2*v_i.v_j >= 4, i.e., v_i.v_j <= 2
    
    # v_i.v_j = sum_{k in S_i ∩ S_j} v_i[k]*v_j[k]
    
    # If |S_i ∩ S_j| = 0: dot = 0 <= 2, OK
    # If |S_i ∩ S_j| = 1: dot = +/-1 <= 2, OK
    # If |S_i ∩ S_j| = 2: dot in {-2,0,2}, all <= 2, OK
    # If |S_i ∩ S_j| = 3: dot in {-3,-1,1,3}
    #   dot = 3 > 2 -> CONFLICT
    # If |S_i ∩ S_j| = 4: dot in {-4,-2,0,2,4}
    #   dot = 4 > 2 -> CONFLICT (but only if same vector, which we don't have)
    #   dot = 2 <= 2, OK
    
    # So conflicts occur when:
    # 1. Same support (|overlap|=4) and dot=4 (same signs) - won't happen for distinct vectors
    # 2. |overlap|=3 and dot=3 (all 3 overlapping positions have same signs)
    
    # So we need to avoid pairs of vectors whose supports overlap in exactly 3 positions 
    # AND whose signs match on all 3 overlapping positions.
    
    # Let's build the conflict graph properly.
    # Total vectors: 5280
    # For each support, 16 sign patterns.
    # Two vectors conflict iff:
    #   - They have the same support AND same signs (impossible for distinct)
    #   - Their supports share exactly 3 positions AND their signs agree on all 3 shared positions
    
    # Let me index vectors as (sup_idx, sign_idx) where sign_idx in 0..15
    
    n_vectors = n_sup * 16
    
    # Precompute sign patterns for 4 positions
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    
    # Build list of vectors
    # vec[sup_idx][sign_idx] = tuple of (pos, val) for the 4 non-zero positions
    # Actually let's just store the full vector representation
    
    # For efficiency, let's precompute for each pair of supports with |overlap|=3,
    # which sign patterns conflict.
    
    # First, build support overlap map
    # For each pair of supports (i, j) with |overlap|=3, record the 3 shared positions
    # and the position in each support that is unique
    
    conflict_support_pairs = {}  # (i, j) -> (shared_pos_in_i, shared_pos_in_j) or similar
    
    # For each support pair with exactly 3 overlap
    support_overlaps = {}
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            s_i = set(supports[i])
            s_j = set(supports[j])
            inter = s_i & s_j
            if len(inter) == 3:
                # Find the position in i not in j, and in j not in i
                pos_i_only = list(s_i - s_j)[0]
                pos_j_only = list(s_j - s_i)[0]
                # Map shared positions to indices in each support
                shared_sorted = sorted(inter)
                idx_in_i = [supports[i].index(p) for p in shared_sorted]
                idx_in_j = [supports[j].index(p) for p in shared_sorted]
                support_overlaps[(i, j)] = (idx_in_i, idx_in_j)
    
    # Now for each such pair, determine which sign patterns conflict
    # Conflict if signs match on all 3 shared positions
    
    # Build conflict graph as adjacency sets
    # n_vectors = 5280, so adjacency list of sets
    
    # Vector index: v_idx = sup_idx * 16 + sign_idx
    
    conflicts = [set() for _ in range(n_vectors)]
    
    for (i, j), (idx_i, idx_j) in support_overlaps.items():
        # For each sign pattern pair (si, sj), check if they conflict
        # Conflict: for all k in 0..2, sign_patterns[si][idx_i[k]] == sign_patterns[sj][idx_j[k]]
        for si in range(16):
            pat_i = sign_patterns[si]
            for sj in range(16):
                pat_j = sign_patterns[sj]
                # Check if signs match on shared positions
                conflict = True
                for k in range(3):
                    if pat_i[idx_i[k]] != pat_j[idx_j[k]]:
                        conflict = False
                        break
                if conflict:
                    vi = i * 16 + si
                    vj = j * 16 + sj
                    conflicts[vi].add(vj)
                    conflicts[vj].add(vi)
    
    # Now we want to find the maximum independent set in this conflict graph.
    # The graph has 5280 vertices. We want the largest set of non-conflicting vectors.
    
    # Use simulated annealing / greedy approach
    
    # First, try a greedy initialization
    def greedy_init():
        remaining = set(range(n_vectors))
        selected = set()
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            selected.add(best_v)
            remaining.difference_update({best_v} | (conflicts[best_v] & remaining))
        return selected
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Simulated annealing to improve
    S = best_set.copy()
    current_size = len(S)
    
    T = 2.0
    cooling = 0.9995
    n_steps = 200000
    
    elapsed_check = 0
    
    for step in range(n_steps):
        if step % 10000 == 0:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Pick random move
        r = random.random()
        
        if r < 0.7:
            # Swap: remove a vertex, add a non-adjacent vertex
            if not S:
                continue
            v_remove = random.choice(list(S))
            # Find candidates to add: not in S, not adjacent to v_remove... 
            # Actually, we need a vertex that is not adjacent to any vertex in S - {v_remove}
            # This is expensive. Let's try a different move.
            pass
        
        if r < 0.5:
            # Try to add a vertex
            v = random.randint(0, n_vectors - 1)
            if v in S:
                continue
            # Check if v can be added (no conflicts with S)
            # v can be added if conflicts[v] & S is empty
            # But checking this is expensive. Let's just check.
            if not (conflicts[v] & S):
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
        elif r < 0.8:
            # Try to remove a vertex (to enable future additions)
            if S:
                v = random.choice(list(S))
                S.discard(v)
                current_size -= 1
        else:
            # Swap: remove v1, add v2
            if not S:
                continue
            v1 = random.choice(list(S))
            v2 = random.randint(0, n_vectors - 1)
            if v2 in S:
                continue
            # Check: v2 not adjacent to any in S - {v1}, and v1 not adjacent to v2
            new_S = S - {v1}
            if v2 in new_S:
                continue
            if conflicts[v2] & new_S:
                continue
            if v1 in conflicts[v2]:
                continue
            # Also need v1 not in conflicts[v2] - already checked
            # Apply swap
            S.discard(v1)
            S.add(v2)
            # Size unchanged
            # Accept with probability based on temperature (for exploration)
            if random.random() < 0.5:
                pass  # accept
            else:
                S.discard(v2)
                S.add(v1)
        
        # Cooldown
        T *= cooling
    
    # Final greedy pass to see if we can add more
    for v in range(n_vectors):
        if v not in S and not (conflicts[v] & S):
            S.add(v)
            current_size += 1
    if current_size > best_size:
        best_size = current_size
        best_set = S.copy()
    
    # Construct the output vectors
    vectors = []
    for v_idx in best_set:
        sup_idx = v_idx // 16
        sign_idx = v_idx % 16
        sup = supports[sup_idx]
        signs = sign_patterns[sign_idx]
        vec = [0] * D
        for k in range(4):
            vec[sup[k]] = signs[k]
        vectors.append(vec)
    
    return np.array(vectors, dtype=np.int64)
