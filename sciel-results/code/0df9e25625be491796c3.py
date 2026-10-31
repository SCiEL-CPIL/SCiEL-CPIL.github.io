"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # We focus on weight-4 vectors (norm^2 = 4)
    # Condition for valid kissing config: min pairwise dist^2 >= max norm^2
    # For weight-4 vectors, norm^2 = 4.
    # Distance squared between two weight-4 vectors u, v:
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v
    # We need 8 - 2 u.v >= 4  =>  2 u.v <= 4  =>  u.v <= 2.
    #
    # u.v for weight-4 vectors:
    # If they share k coordinates, and signs agree on s of those, disagree on k-s:
    # u.v = s - (k-s) = 2s - k
    # We need 2s - k <= 2.
    #
    # Possible intersections k (max 4 since weight 4):
    # k=0: u.v = 0 <= 2. OK.
    # k=1: u.v = 2s-1. s=0 -> -1 OK. s=1 -> 1 OK.
    # k=2: u.v = 2s-2. s=0 -> -2 OK. s=1 -> 0 OK. s=2 -> 2 OK.
    # k=3: u.v = 2s-3. s=0 -> -3 OK. s=1 -> -1 OK. s=2 -> 1 OK. s=3 -> 3 FAIL.
    # k=4: u.v = 2s-4. s=0 -> -4 OK. s=1 -> -2 OK. s=2 -> 0 OK. s=3 -> 2 OK. s=4 -> 4 FAIL.
    #
    # So conflicts happen when:
    # - k=3 and s=3 (share 3 coords, all same sign)
    # - k=4 and s=4 (identical vectors, which we won't have duplicates of)
    #
    # Wait, if k=4 and s=4, they are the same vector. We use unique vectors so this won't happen.
    # If k=4 and s=3, u.v = 2. OK.
    #
    # So the ONLY conflict for distinct weight-4 vectors is:
    # They share exactly 3 coordinates AND have the same sign on all 3 shared coordinates.
    
    # Total number of weight-4 vectors:
    # Choose 4 coords from 11: C(11,4) = 330
    # 2^4 = 16 sign combinations
    # Total = 330 * 16 = 5280
    
    # Generate all weight-4 vectors
    all_vecs = []
    support_to_vecs = defaultdict(list)
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            all_vecs.append(v)
            support_to_vecs[support].append(v)
    
    # For each support, group vectors by sign pattern on that support
    # Conflict: two vectors with supports S1, S2 where |S1 ∩ S2| = 3 and same signs on intersection.
    
    # Strategy: Greedy with vectorized conflict checking
    # For efficiency, we'll maintain a set of selected vectors.
    # When considering a new vector v with support S_v:
    #   It conflicts with existing vector u if:
    #     - |S_u ∩ S_v| = 3
    #     - signs agree on the intersection
    #
    # Precompute for each support S, the list of vectors with that support.
    # For a new vector v with support S_v and sign pattern p_v on S_v:
    #   For each support S_u that intersects S_v in exactly 3 elements:
    #     The intersection is S_int = S_v ∩ S_u (size 3)
    #     The "other" element in S_u is S_u \ S_int
    #     For each existing vector u with support S_u:
    #       Check if u's signs on S_int match v's signs on S_int.
    
    # Let's build a more efficient structure.
    # For each triple of coordinates T (size 3), and each sign pattern on T (8 patterns),
    # track how many selected vectors have T as a subset of their support with that sign pattern.
    
    # Actually, let's think differently.
    # A vector v with support S_v (size 4) and signs p_v.
    # It has 4 triples: S_v \ {i} for i in S_v.
    # For each triple T = S_v \ {i}, the sign pattern on T is p_v restricted to T.
    # v conflicts with u if u has some triple T_u such that T_u = T (for some T in v's triples)
    # and u's sign pattern on T matches v's sign pattern on T.
    
    # So: maintain a dictionary: triple_sign_count[(triple, sign_pattern)] = count
    # When adding v:
    #   For each of v's 4 triples T with sign pattern p_T:
    #     If triple_sign_count[(T, p_T)] > 0, then v conflicts.
    #   If no conflict, add v and increment counts for its 4 triples.
    
    # This is O(4) per vector check! Very fast.
    
    # Let's implement this.
    
    # Map triple (tuple of 3 sorted coords) and sign pattern (tuple of 3 signs) to count
    triple_sign_count = defaultdict(int)
    
    def get_triples_and_signs(v):
        """Get the 4 triples of v's support and the sign pattern on each."""
        support = tuple(i for i in range(D) if v[i] != 0)
        triples = []
        for i in range(4):
            # Remove i-th element from support
            T = support[:i] + support[i+1:]
            # Sign pattern on T
            signs_on_T = tuple(v[j] for j in T)
            triples.append((T, signs_on_T))
        return triples
    
    # Precompute triples and signs for all vectors
    vec_triples = []
    for v in all_vecs:
        vec_triples.append(get_triples_and_signs(v))
    
    # Greedy: try to add vectors in some order
    # To maximize packing, we should try to add vectors that have the fewest conflicts.
    # But with the triple_sign_count approach, we can just iterate and add if no conflict.
    # The order matters. Let's try multiple random orders and keep the best.
    
    best_set = []
    best_count = 0
    
    n_vecs = len(all_vecs)
    indices = list(range(n_vecs))
    
    # Try several random orders
    np.random.seed(42)
    n_attempts = 200
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        np.random.shuffle(indices)
        
        triple_sign_count.clear()
        current_set = []
        
        for idx in indices:
            v = all_vecs[idx]
            triples = vec_triples[idx]
            
            # Check if any triple has a conflict
            conflict = False
            for T, signs_on_T in triples:
                if triple_sign_count[(T, signs_on_T)] > 0:
                    conflict = True
                    break
            
            if not conflict:
                current_set.append(v)
                for T, signs_on_T in triples:
                    triple_sign_count[(T, signs_on_T)] += 1
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
    
    # Also try a deterministic order as fallback
    if time.time() < deadline:
        triple_sign_count.clear()
        current_set = []
        for idx in range(n_vecs):
            v = all_vecs[idx]
            triples = vec_triples[idx]
            conflict = False
            for T, signs_on_T in triples:
                if triple_sign_count[(T, signs_on_T)] > 0:
                    conflict = True
                    break
            if not conflict:
                current_set.append(v)
                for T, signs_on_T in triples:
                    triple_sign_count[(T, signs_on_T)] += 1
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
    
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
