import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    def random_poly():
        positions = random.sample(range(41), 12)
        positions.sort()
        values = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        poly = [0] * 41
        for i, v in zip(positions, values):
            poly[i] = v
        # Ensure final coefficient is non-zero (it is, as positions are distinct and within 0..40)
        # But what if the last position is 0? 
        # positions are from 0..40.
        # If 40 is not in positions, coeffs[-1] (index 40) is 0.
        # validate requires coeffs[-1] != 0.
        # So we must ensure 40 is in positions?
        # Or we can trim the list to the last non-zero term.
        # "Return coefficient list... nonzero final coefficient".
        # If I return a list of length 41 with last element 0, it's invalid.
        # So I should trim the list to the last non-zero element.
        
        last_nz = max(positions)
        poly = poly[:last_nz+1]
        return poly

    def local_search(curr, max_iters=3000):
        curr = curr.copy()
        curr_score = evaluate(curr)
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            move_type = random.choice(['change_val', 'move_pos', 'swap_pos', 'replace'])
            
            if move_type == 'change_val':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if not non_zero_indices: continue
                idx = random.choice(non_zero_indices)
                old_val = curr[idx]
                # Try a few new values
                found_improve = False
                for _ in range(5):
                    new_val = random.randint(-64, 64)
                    if new_val == 0 or new_val == old_val:
                        continue
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        found_improve = True
                        break
                    curr[idx] = old_val
                if not found_improve:
                    no_improve += 1

            elif move_type == 'move_pos':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if not non_zero_indices: continue
                idx = random.choice(non_zero_indices)
                old_val = curr[idx]
                new_idx = random.randint(0, 40)
                
                # We can extend the list if new_idx >= len(curr)
                # But we must maintain length <= 41.
                # If new_idx > 40, skip.
                if new_idx > 40:
                    continue
                
                # If new_idx is within current length and occupied, skip
                if new_idx < len(curr) and curr[new_idx] != 0:
                    continue
                
                # Perform move
                if new_idx >= len(curr):
                    # Extend list with zeros
                    curr.extend([0] * (new_idx - len(curr) + 1))
                
                curr[idx] = 0
                curr[new_idx] = old_val
                
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                    # Trim trailing zeros?
                    # Not strictly necessary for evaluate, but keeps length small.
                    # However, if we trim, we change the list.
                    # Let's trim trailing zeros to keep "nonzero final coefficient" property easy.
                    while curr and curr[-1] == 0:
                        curr.pop()
                else:
                    # Revert
                    curr[new_idx] = 0
                    curr[idx] = old_val
                    # If we extended, we might have trailing zeros.
                    # Revert extension?
                    # It's safer to just work with fixed length 41 internally 
                    # and trim at the end.
                    # Let's switch to fixed length 41 for local search.
                    
            # ... This is getting messy with dynamic lengths.
            # Let's use fixed length 41 for the search, and trim at the end.
            
            break # Break out of this complex draft, rewrite properly.

    # Rewritten Local Search with Fixed Length 41
    
    def local_search_fixed(curr, max_iters=3000):
        curr = curr.copy() # curr is length 41
        curr_score = evaluate(curr)
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            move_type = random.choice(['change_val', 'move_pos', 'swap_pos', 'replace'])
            
            if move_type == 'change_val':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if not non_zero_indices: continue
                idx = random.choice(non_zero_indices)
                old_val = curr[idx]
                found_improve = False
                for _ in range(5):
                    new_val = random.randint(-64, 64)
                    if new_val == 0 or new_val == old_val:
                        continue
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        found_improve = True
                        break
                    curr[idx] = old_val
                if not found_improve:
                    no_improve += 1

            elif move_type == 'move_pos':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if not non_zero_indices: continue
                idx = random.choice(non_zero_indices)
                old_val = curr[idx]
                new_idx = random.randint(0, 40)
                if new_idx == idx:
                    continue
                if curr[new_idx] != 0:
                    continue
                
                curr[idx] = 0
                curr[new_idx] = old_val
                
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                else:
                    curr[new_idx] = 0
                    curr[idx] = old_val
                    no_improve += 1

            elif move_type == 'swap_pos':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if len(non_zero_indices) < 2:
                    continue
                idx1, idx2 = random.sample(non_zero_indices, 2)
                v1, v2 = curr[idx1], curr[idx2]
                curr[idx1], curr[idx2] = v2, v1
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                else:
                    curr[idx1], curr[idx2] = v1, v2
                    no_improve += 1

            elif move_type == 'replace':
                non_zero_indices = [i for i, c in enumerate(curr) if c != 0]
                if not non_zero_indices: continue
                idx_remove = random.choice(non_zero_indices)
                old_val_remove = curr[idx_remove]
                curr[idx_remove] = 0
                
                idx_add = random.randint(0, 40)
                if curr[idx_add] != 0:
                    curr[idx_remove] = old_val_remove
                    continue
                    
                val_add = random.choice([-2, -1, 1, 2])
                curr[idx_add] = val_add
                
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                else:
                    curr[idx_add] = 0
                    curr[idx_remove] = old_val_remove
                    no_improve += 1

            if no_improve > 100:
                break

        return curr, curr_score

    # Main Loop
    for i in range(300):
        if time.time() > deadline - 10:
            break
            
        # Generate random poly of length 41
        positions = random.sample(range(41), 12)
        values = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        curr = [0] * 41
        for i, v in zip(positions, values):
            curr[i] = v
            
        curr_score = evaluate(curr)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr.copy()
            
        # Local Search
        curr, curr_score = local_search_fixed(curr, max_iters=2000)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr.copy()

    # Final Polish: Trim
    if best_poly is not None:
        # Trim leading zeros
        first_nz = 0
        while first_nz < len(best_poly) and best_poly[first_nz] == 0:
            first_nz += 1
        best_poly = best_poly[first_nz:]
        
        # Trim trailing zeros (should be none if we ensured non-zero at end, 
        # but random positions might not include 40)
        while best_poly and best_poly[-1] == 0:
            best_poly.pop()
            
        if not best_poly:
             best_poly = [1]

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
