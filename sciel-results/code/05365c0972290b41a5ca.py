import time
import random
import itertools

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute helper functions
    def index_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        return digits

    def digits_to_index(digits):
        idx = 0
        for i, d in enumerate(reversed(digits)):
            idx += d * (q ** i)
        return idx

    # Pre-compute spheres (indices and masks)
    sphere_indices = []
    sphere_masks = []
    for idx in range(total_words):
        digits = index_to_digits(idx)
        inds = [idx]
        for i in range(n):
            original_digit = digits[i]
            for new_digit in range(q):
                if new_digit != original_digit:
                    new_digits = digits.copy()
                    new_digits[i] = new_digit
                    new_idx = digits_to_index(new_digits)
                    inds.append(new_idx)
        sphere_indices.append(inds)
        
        mask = 0
        for w in inds:
            mask |= (1 << w)
        sphere_masks.append(mask)

    # Initialize code with tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_digits = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code_digits.append(h + list(tail))
    
    current_code = []
    in_code = [False] * total_words
    cover_count = [0] * total_words
    
    # Initialize uncovered_mask with all bits set
    all_ones = (1 << total_words) - 1
    uncovered_mask = all_ones
    
    for cw_digits in initial_code_digits:
        idx = digits_to_index(cw_digits)
        if not in_code[idx]:
            current_code.append(idx)
            in_code[idx] = True
            for w in sphere_indices[idx]:
                cover_count[w] += 1
                if cover_count[w] == 1:
                    uncovered_mask &= ~(1 << w)

    best_size = len(current_code)
    best_code = current_code[:]
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Local search
    while time.time() < deadline:
        # 1. Greedy Removal
        random.shuffle(current_code)
        # Iterate backwards to safely remove
        for i in range(len(current_code) - 1, -1, -1):
            c_idx = current_code[i]
            if not in_code[c_idx]:
                continue
            
            # Try remove
            in_code[c_idx] = False
            valid = True
            for w in sphere_indices[c_idx]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for w in sphere_indices[c_idx]:
                    cover_count[w] += 1
                    if cover_count[w] == 1:
                        uncovered_mask &= ~(1 << w)
                in_code[c_idx] = True
            else:
                # Keep removed
                current_code.pop(i)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
        
        # 2. Perturbation
        if current_code:
            rem_idx_pos = random.randint(0, len(current_code) - 1)
            c_remove = current_code[rem_idx_pos]
            
            # Remove it
            in_code[c_remove] = False
            current_code.pop(rem_idx_pos)
            for w in sphere_indices[c_remove]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    uncovered_mask |= (1 << w)
            
            # Repair
            while uncovered_mask != 0:
                best_score = -1
                best_c = -1
                for c in range(total_words):
                    s = (sphere_masks[c] & uncovered_mask).bit_count()
                    if s > best_score:
                        best_score = s
                        best_c = c
                
                if best_c == -1 or best_score == 0:
                    break
                
                in_code[best_c] = True
                current_code.append(best_c)
                for w in sphere_indices[best_c]:
                    cover_count[w] += 1
                    if cover_count[w] == 1:
                        uncovered_mask &= ~(1 << w)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]

    result = []
    for idx in best_code:
        result.append(index_to_digits(idx))
        
    return result
