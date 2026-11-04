import numpy as np
from itertools import product
from collections import defaultdict

D = 11

def construct():
    """
    Greedy + local search for kissing configuration in R^11.
    
    Strategy:
    1. Generate candidate integer vectors with norm^2 = 8 (small, dense).
    2. Greedily add vectors maintaining the condition: for all pairs, ||u-v||^2 >= 8.
    3. Local search: try to add more vectors by checking against existing set.
    4. Try multiple norm^2 values and take the best.
    
    The condition min_dist >= max_norm with all vectors having norm^2 = N
    means: for all pairs u,v in the set, ||u-v||^2 >= N.
    Equivalently: u·v <= N/2 for all pairs (since ||u-v||^2 = 2N - 2u·v >= N iff u·v <= N/2).
    """
    
    def generate_candidates(norm_sq, max_coord=3):
        """Generate all integer vectors with given norm_sq, coords in [-max_coord, max_coord]."""
        candidates = []
        if norm_sq <= 0:
            return candidates
        
        # Use recursive generation
        def gen(coords, remaining, dim):
            if dim == 0:
                if remaining == 0:
                    candidates.append(tuple(coords))
                return
            for c in range(-max_coord, max_coord + 1):
                c2 = c * c
                if c2 <= remaining:
                    coords.append(c)
                    gen(coords, remaining - c2, dim - 1)
                    coords.pop()
        
        gen([], norm_sq, D)
        # Remove zero vector
        candidates = [c for c in candidates if any(x != 0 for x in c)]
        return candidates
    
    def check_condition(vec, existing, norm_sq):
        """Check if vec can be added to existing set."""
        v = np.array(vec, dtype=np.int64)
        for u in existing:
            diff = v - u
            if np.dot(diff, diff) < norm_sq:
                return False
        return True
    
    def greedy_construct(norm_sq, max_coord=3):
        """Greedily construct a valid set."""
        candidates = generate_candidates(norm_sq, max_coord)
        if not candidates:
            return []
        
        # Sort candidates by some heuristic (e.g., first non-zero coordinate)
        # to get a deterministic and potentially good ordering
        candidates.sort()
        
        selected = []
        selected_np = []
        
        for cand in candidates:
            if check_condition(cand, selected_np, norm_sq):
                selected.append(cand)
                selected_np.append(np.array(cand, dtype=np.int64))
        
        return selected_np
    
    def local_search(selected, norm_sq, max_coord=4, iterations=50):
        """
        Local search: repeatedly try to add vectors that weren't previously tried,
        or remove one and add two (swap).
        """
        if not selected:
            return selected
        
        selected = list(selected)
        selected_set = set(tuple(v) for v in selected)
        
        # Generate a broader set of candidates
        all_candidates = generate_candidates(norm_sq, max_coord)
        
        improved = True
        it = 0
        while improved and it < iterations:
            improved = False
            it += 1
            
            # Try adding any candidate not already in set
            for cand in all_candidates:
                if cand in selected_set:
                    continue
                if check_condition(cand, selected, norm_sq):
                    selected.append(np.array(cand, dtype=np.int64))
                    selected_set.add(cand)
                    improved = True
            
            # Try 1-for-2 swaps: remove one, add two others
            if not improved:
                for i in range(len(selected)):
                    orig = selected[i]
                    orig_tuple = tuple(orig)
                    # Temporarily remove
                    selected[i] = None
                    remaining = [s for s in selected if s is not None]
                    
                    added_count = 0
                    new_vecs = []
                    for cand in all_candidates:
                        if cand == orig_tuple:
                            continue
                        if cand in selected_set:
                            continue
                        if check_condition(cand, remaining, norm_sq):
                            # Check against new_vecs too
                            can_add = True
                            for nv in new_vecs:
                                diff = np.array(cand) - nv
                                if np.dot(diff, diff) < norm_sq:
                                    can_add = False
                                    break
                            if can_add:
                                new_vecs.append(np.array(cand, dtype=np.int64))
                                added_count += 1
                    
                    if added_count >= 2:
                        selected[i] = new_vecs[0]
                        for nv in new_vecs[1:]:
                            selected.append(nv)
                        selected_set.discard(orig_tuple)
                        for nv in new_vecs:
                            selected_set.add(tuple(nv))
                        improved = True
                    else:
                        selected[i] = orig
    
        return [s for s in selected if s is not None]
    
    # Try multiple norm_sq values
    best_result = None
    best_count = 0
    
    for norm_sq in [4, 8, 12, 16, 20]:
        max_coord = int(np.sqrt(norm_sq)) + 1
        # Greedy
        result = greedy_construct(norm_sq, max_coord)
        
        # Local search
        result = local_search(result, norm_sq, max_coord=min(max_coord + 1, 5), iterations=20)
        
        count = len(result)
        if count > best_count:
            best_count = count
            best_result = result
    
    if best_result is None or len(best_result) == 0:
        # Fallback: use the E8 x A3 construction
        vectors = []
        # E8 Type 1: (±2, ±2, 0...) in first 8 dims
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * 11
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        vectors.append(v)
        # E8 Type 2: (±1,...,±1) even minus in first 8 dims
        for mask in range(256):
            v = [0] * 11
            mc = 0
            for i in range(8):
                if (mask >> i) & 1:
                    v[i] = -1
                    mc += 1
                else:
                    v[i] = 1
            if mc % 2 == 0:
                vectors.append(v)
        # A3 in dims 8-10: (±2,±2,0) perms
        for i in range(8, 11):
            for j in range(i + 1, 11):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * 11
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        vectors.append(v)
        best_result = np.array(vectors, dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)
