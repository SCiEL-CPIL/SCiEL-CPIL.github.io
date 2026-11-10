# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    random.seed(42)
    
    def gen_vectors(nonzeros):
        """Generate all vectors with exactly `nonzeros` non-zero entries (each ±1)."""
        vecs = []
        for combo in itertools.combinations(range(d), nonzeros):
            for signs in itertools.product([1, -1], repeat=nonzeros):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs
    
    # Generate candidates: 6-norm, 4-norm, 2-norm
    candidates_6 = gen_vectors(6)
    candidates_4 = gen_vectors(4)
    candidates_2 = gen_vectors(2)
    
    # Greedy selection with vectorized checks
    selected = []  # list of selected vectors
    selected_arr = np.zeros((0, d), dtype=np.int64)
    
    # Process in order: 6-norm first (most constrained), then 4-norm, then 2-norm
    for group in [candidates_6, candidates_4, candidates_2]:
        # Shuffle for diversity
        random.shuffle(group)
        for v in group:
            if time.time() > deadline - 200:
                break
            if not selected:
                selected.append(v)
                selected_arr = v.reshape(1, -1).astype(np.int64)
                continue
            # Check dot products: need max dot <= min_norm/2 for pairs
            # For vectors with norms n1, n2: need |v1·v2| <= min(n1,n2)/2
            # Actually the constraint is: min pair dist >= max point norm
            # dist^2 = n1 + n2 - 2*v1·v2 >= max(n1, n2)
            # So: 2*v1·v2 <= n1 + n2 - max(n1,n2) = min(n1,n2)
            # And also: n1 + n2 - 2*v1·v2 >= min(n1,n2) is automatic if above holds
            # Wait, we need BOTH: dist^2 >= max(n1,n2)
            # dist^2 = n1+n2-2dot >= max(n1,n2) => 2dot <= min(n1,n2)
            # Also need points distinct: dist^2 > 0 => dot < (n1+n2)/2
            v_norm2 = int(np.dot(v, v))
            if len(selected) == 0:
                selected.append(v)
                selected_arr = v.reshape(1, -1).astype(np.int64)
                continue
            dots = selected_arr @ v
            # For each selected vector, check: 2*|dot| <= min(norm_sel, v_norm2)
            # Actually dot can be negative, so we need: 2*dot <= min(n_sel, v_norm2) 
            # AND dist^2 = n_sel + v_norm2 - 2*dot >= max(n_sel, v_norm2)
            # => 2*dot <= min(n_sel, v_norm2)
            # The second constraint (distinctness) is: n_sel + v_norm2 - 2*dot > 0
            # which is weaker when dot is negative
            norms_sel = np.sum(selected_arr ** 2, axis=1)
            min_norms = np.minimum(norms_sel, v_norm2)
            max_dots_allowed = min_norms // 2  # 2*dot <= min_norms => dot <= min_norms//2
            # Also need dot >= -(n_sel + v_norm2 - 1) // 2 for distinctness (dist^2 >= 1)
            min_dots_needed = (-(norms_sel + v_norm2) + 1) // 2
            # Actually: dist^2 >= 1 => n_sel + v_norm2 - 2*dot >= 1 => dot <= (n_sel+v_norm2-1)//2
            # This is always satisfied if dot <= min_norms//2 since min <= max
            if np.max(dots) <= np.min(max_dots_allowed):
                selected.append(v)
                selected_arr = np.vstack((selected_arr, v.reshape(1, -1).astype(np.int64)))
    
    # Local search: try to add more vectors via random restarts of greedy
    if time.time() < deadline - 100:
        # Try adding 1-norm vectors (unit vectors)
        for i in range(d):
            for sign in [1, -1]:
                if time.time() > deadline - 50:
                    break
                v = np.zeros(d, dtype=np.int64)
                v[i] = sign
                v_norm2 = 1
                if len(selected) == 0:
                    selected.append(v)
                    selected_arr = v.reshape(1, -1).astype(np.int64)
                    continue
                dots = selected_arr @ v
                norms_sel = np.sum(selected_arr ** 2, axis=1)
                min_norms = np.minimum(norms_sel, v_norm2)
                max_dots_allowed = min_norms // 2
                if np.max(dots) <= np.min(max_dots_allowed):
                    selected.append(v)
                    selected_arr = np.vstack((selected_arr, v.reshape(1, -1).astype(np.int64)))
    
    # Swap-based local search: try replacing one selected vector to allow more additions
    if time.time() < deadline - 30 and len(selected) > 10:
        # For a few iterations, try swapping
        best_count = len(selected)
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            if len(selected) <= best_count:
                break
            # Pick a random vector to remove
            idx = random.randint(0, len(selected) - 1)
            removed = selected[idx]
            selected[idx] = None
            selected_arr = np.vstack((selected_arr[:idx], selected_arr[idx+1:]))
            # Try to add multiple new vectors
            added_count = 0
            for group in [candidates_6, candidates_4, candidates_2]:
                # Try a few random candidates
                for _ in range(100):
                    if time.time() > deadline - 30:
                        break
                    v = group[random.randint(0, len(group)-1)]
                    v_norm2 = int(np.dot(v, v))
                    if len(selected_arr) == 0:
                        selected[idx] = v
                        selected_arr = v.reshape(1, -1).astype(np.int64)
                        added_count += 1
                        continue
                    dots = selected_arr @ v
                    norms_sel = np.sum(selected_arr ** 2, axis=1)
                    min_norms = np.minimum(norms_sel, v_norm2)
                    max_dots_allowed = min_norms // 2
                    if np.max(dots) <= np.min(max_dots_allowed):
                        selected[idx] = v
                        selected_arr = np.vstack((selected_arr, v.reshape(1, -1).astype(np.int64)))
                        added_count += 1
            # If we didn't add enough, restore
            if added_count < 2:
                selected[idx] = removed
                # Rebuild selected_arr
                selected_clean = [s for s in selected if s is not None]
                selected = selected_clean
                selected_arr = np.array(selected, dtype=np.int64)
    
    # Final validation and cleanup
    selected = [s for s in selected if s is not None]
    if not selected:
        # Fallback
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1; points.append(p1)
                p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1; points.append(p2)
                p3 = np.zeros(d, dtype=np.int64); p3[i] = 1; p3[j] = -1; points.append(p3)
                p4 = np.zeros(d, dtype=np.int64); p4[i] = -1; p4[j] = 1; points.append(p4)
        return np.array(points, dtype=np.int64)
    
    result = np.array(selected, dtype=np.int64)
    return np.around(result).astype(np.int64)

# EVOLVE-BLOCK-END
