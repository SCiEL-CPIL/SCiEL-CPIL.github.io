import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_words, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_words, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            uncovered = ~is_covered
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            sel_list = list(selected_set)
            to_remove = []
            for idx in sel_list:
                if np.all(coverage_count[neighbors[idx]] > 1):
                    to_remove.append(idx)
            if to_remove:
                changed = True
                for rem_idx in to_remove:
                    coverage_count[neighbors[rem_idx]] -= 1
                    selected_set.discard(rem_idx)
        
        return list(selected_set)

    def local_search(selected_indices, deadline, seed=42):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(seed)
        
        # Build reverse mapping: for each point, which codewords cover it
        # Precompute: for each point, list of codeword indices that cover it
        # This is a 6561 x 486-ish sparse structure
        point_to_covers = [[] for _ in range(total_words)]
        for i, idx in enumerate(selected_list):
            for p in neighbors[idx]:
                point_to_covers[p].append(i)
        
        stuck_counter = 0
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # With some probability, try a 2-swap move (remove 2, add 1)
            if len(selected_list) >= 2 and rng.random() < 0.3:
                # Pick two random codewords to try removing
                i1, i2 = rng.choice(len(selected_list), size=2, replace=False)
                c1 = selected_list[i1]
                c2 = selected_list[i2]
                
                # Tentatively remove both
                coverage_count[neighbors[c1]] -= 1
                coverage_count[neighbors[c2]] -= 1
                
                # Find points that are now uncovered
                uncovered_after_removal = np.where(coverage_count == 0)[0]
                
                if len(uncovered_after_removal) == 0:
                    # Both were redundant! Remove them
                    # Remove the higher index first to keep indices valid
                    if i1 > i2:
                        i1, i2 = i2, i1
                    selected_list.pop(i1)
                    selected_list.pop(i2)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    # Rebuild point_to_covers
                    point_to_covers = [[] for _ in range(total_words)]
                    for i, idx in enumerate(selected_list):
                        for p in neighbors[idx]:
                            point_to_covers[p].append(i)
                    stuck_counter = 0
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    continue
                
                # Try to find a single codeword that covers all uncovered points
                if len(uncovered_after_removal) <= 20:
                    # Find intersection of covers for all uncovered points
                    # Use the first uncovered point's covers as candidate set
                    target_points = uncovered_after_removal
                    # Start with covers of first point
                    candidate_set = set(point_to_covers[target_points[0]])
                    # Intersect with covers of other points
                    for tp in target_points[1:]:
                        if not candidate_set:
                            break
                        candidate_set &= set(point_to_covers[tp])
                    
                    # Also consider new codewords not in the set
                    found = False
                    # Try candidates from intersection
                    for cand_idx in candidate_set:
                        cand = selected_list[cand_idx]
                        coverage_count[neighbors[cand]] += 1
                        if not np.any(coverage_count == 0):
                            # Accept: remove c1, c2, keep cand
                            # Need to update selected_list
                            # Remove c1 and c2, keep cand at its position
                            # This is complex; let's just rebuild
                            new_list = []
                            for j, x in enumerate(selected_list):
                                if x != c1 and x != c2:
                                    new_list.append(x)
                            selected_list = new_list
                            selected_set = set(selected_list)
                            # Rebuild point_to_covers
                            point_to_covers = [[] for _ in range(total_words)]
                            for i, idx in enumerate(selected_list):
                                for p in neighbors[idx]:
                                    point_to_covers[p].append(i)
                            found = True
                            break
                        else:
                            coverage_count[neighbors[cand]] -= 1
                    
                    # If not found, try random new codewords
                    if not found:
                        for _ in range(30):
                            candidate = rng.integers(total_words)
                            if candidate in selected_set:
                                continue
                            # Check if candidate covers all uncovered points
                            covers_all = True
                            for tp in target_points:
                                if tp not in neighbors[candidate]:
                                    covers_all = False
                                    break
                            if covers_all:
                                coverage_count[neighbors[candidate]] += 1
                                if not np.any(coverage_count == 0):
                                    new_list = []
                                    for j, x in enumerate(selected_list):
                                        if x != c1 and x != c2:
                                            new_list.append(x)
                                    new_list.append(candidate)
                                    selected_list = new_list
                                    selected_set = set(selected_list)
                                    point_to_covers = [[] for _ in range(total_words)]
                                    for i, idx in enumerate(selected_list):
                                        for p in neighbors[idx]:
                                            point_to_covers[p].append(i)
                                    found = True
                                    break
                                else:
                                    coverage_count[neighbors[candidate]] -= 1
                    
                    if found:
                        stuck_counter = 0
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                    else:
                        # Revert
                        coverage_count[neighbors[c1]] += 1
                        coverage_count[neighbors[c2]] += 1
                        stuck_counter += 1
                    continue
                
                # Revert if too many uncovered
                coverage_count[neighbors[c1]] += 1
                coverage_count[neighbors[c2]] += 1
                stuck_counter += 1
                continue
            
            # Single removal move
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            
            # Tentatively remove c
            coverage_count[neighbors[c]] -= 1
            
            # Find points that are now uncovered
            uncovered_after_removal = np.where(coverage_count == 0)[0]
            
            if len(uncovered_after_removal) == 0:
                # c was redundant! Remove it
                selected_list.pop(c_idx)
                selected_set.discard(c)
                # Rebuild point_to_covers
                point_to_covers = [[] for _ in range(total_words)]
                for i, idx in enumerate(selected_list):
                    for p in neighbors[idx]:
                        point_to_covers[p].append(i)
                stuck_counter = 0
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue
            
            # Try to find a single codeword that covers all uncovered points
            if len(uncovered_after_removal) <= 20:
                # Find intersection of covers for all uncovered points
                target_points = uncovered_after_removal
                candidate_set = set(point_to_covers[target_points[0]])
                for tp in target_points[1:]:
                    if not candidate_set:
                        break
                    candidate_set &= set(point_to_covers[tp])
                
                found = False
                for cand_idx in candidate_set:
                    cand = selected_list[cand_idx]
                    coverage_count[neighbors[cand]] += 1
                    if not np.any(coverage_count == 0):
                        selected_list[c_idx] = cand
                        selected_set.discard(c)
                        selected_set.add(cand)
                        # Rebuild point_to_covers
                        point_to_covers = [[] for _ in range(total_words)]
                        for i, idx in enumerate(selected_list):
                            for p in neighbors[idx]:
                                point_to_covers[p].append(i)
                        found = True
                        break
                    else:
                        coverage_count[neighbors[cand]] -= 1
                
                if not found:
                    # Try random new codewords
                    for _ in range(30):
                        candidate = rng.integers(total_words)
                        if candidate in selected_set:
                            continue
                        covers_all = True
                        for tp in target_points:
                            if tp not in neighbors[candidate]:
                                covers_all = False
                                break
                        if covers_all:
                            coverage_count[neighbors[candidate]] += 1
                            if not np.any(coverage_count == 0):
                                selected_list[c_idx] = candidate
                                selected_set.discard(c)
                                selected_set.add(candidate)
                                point_to_covers = [[] for _ in range(total_words)]
                                for i, idx in enumerate(selected_list):
                                    for p in neighbors[idx]:
                                        point_to_covers[p].append(i)
                                found = True
                                break
                            else:
                                coverage_count[neighbors[candidate]] -= 1
                
                if found:
                    stuck_counter = 0
                else:
                    coverage_count[neighbors[c]] += 1
                    stuck_counter += 1
            else:
                coverage_count[neighbors[c]] += 1
                stuck_counter += 1
            
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        selected = greedy_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search(best_code, deadline, seed=42)
    
    # Random restarts for local search to escape local minima
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() >= deadline - 5:
                break
            
            # Start from a copy of the best code
            rc_code = best_code[:]
            rc_size = len(rc_code)
            
            # Run local search with a different seed
            rc_code = local_search(rc_code, deadline - 5, seed=42 + restart * 1000)
            
            if len(rc_code) < best_len:
                best_len = len(rc_code)
                best_code = rc_code
    
    # Final Validation
    if not is_valid_code(best_code):
        # Fallback to the last known valid code from greedy
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            # Absolute fallback: full space (valid but large)
            best_code = list(range(total_words))
            
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
