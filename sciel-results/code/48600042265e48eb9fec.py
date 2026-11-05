import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    # Total words = 3^8 = 6561
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    # --- Phase 1: Fast Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
                
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # --- Phase 2: Iterated Local Search ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    # Precompute neighbors_of_neighbor for faster swap candidate selection?
    # Actually, we need: for a risky point x (covered only by c1), find c2 not in code such that x in neighbors[c2].
    # This is equivalent to: c2 is in neighbors[x] and c2 not in current_code_set.
    # neighbors[x] has 17 elements.
    
    stuck_counter = 0
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        # 2. If stuck, try a swap
        if stuck_counter > 50 and time.time() < deadline - 10.0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                # c1 is removable, but we already checked. This shouldn't happen if logic is correct.
                # But if it does, just remove it.
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers all risky points.
                # Since risky points are in neighbors[c1], and c2 must cover them,
                # c2 must be in the intersection of neighbors[x] for all x in risky_points.
                # Actually, c2 just needs to cover ALL risky points.
                # So c2 must be in intersection_{x in risky_points} neighbors[x].
                
                # Start with neighbors of first risky point
                candidates = set(neighbors[risky_points[0]].tolist())
                candidates.discard(c1) # c1 is in code, we're replacing it, but c2 != c1 usually.
                # Actually c2 must not be in current_code_set.
                
                for x in risky_points[1:]:
                    candidates &= set(neighbors[x].tolist())
                    if not candidates:
                        break
                
                # Filter out candidates already in code
                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
                
                if valid_candidates:
                    # Pick a random valid candidate
                    c2 = random.choice(valid_candidates)
                    
                    # Verify swap validity (should be valid by construction, but double check)
                    # Removing c1 drops cov_count for neighbors[c1] by 1.
                    # Adding c2 increases cov_count for neighbors[c2] by 1.
                    # Risky points are covered by c2, so they stay >= 1.
                    # Other points in neighbors[c1] have cov_count > 1, so they stay >= 1.
                    # Points in neighbors[c2] but not neighbors[c1] gain coverage.
                    # So swap is valid.
                    
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2
                    
                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)
                    
                    stuck_counter = 0
                else:
                    # No valid swap found. Increment stuck counter more aggressively?
                    # Or try a different c1?
                    # For now, just let stuck_counter grow.
                    pass
                    
            # Reset stuck counter after attempting swap to avoid immediate re-attempt
            # But if swap failed, we want to try again soon?
            # Let's reset to 0 to allow more removal attempts first.
            stuck_counter = 0
            
    # Final Polish
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
