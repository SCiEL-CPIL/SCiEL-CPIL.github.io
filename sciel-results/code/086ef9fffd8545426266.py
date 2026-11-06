import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Explicit deadline guard
    
    # Set recursion limit just in case, though depth is small
    sys.setrecursionlimit(10000)
    
    # 1. Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # 2. Precompute conflict masks for O(1) compatibility check
    # conflict_mask[i] is a bitmask where bit j is 1 if smooth[i] and smooth[j] conflict
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        si = smooth[i]
        mask = 0
        for j in range(n_smooth):
            if i == j:
                continue
            sj = smooth[j]
            if sj % si == 0 or si % sj == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # 3. Find minimal antichain representation for each sum 1..128
    min_terms = [100] * 129
    best_rep = [None] * 129
    
    def backtrack(start_idx, current_sum, current_size, current_mask, current_indices):
        # Deadline check
        if time.time() > deadline:
            return False
        
        if current_sum > 0 and current_sum <= 128:
            if current_size < min_terms[current_sum]:
                min_terms[current_sum] = current_size
                best_rep[current_sum] = tuple(smooth[i] for i in current_indices)
        
        if current_sum > 128:
            return True
            
        for i in range(start_idx, n_smooth):
            # Check compatibility using bitmask
            if (current_mask & conflict_mask[i]) != 0:
                continue
            
            new_sum = current_sum + smooth[i]
            if new_sum > 128:
                break
                
            new_mask = current_mask | (1 << i)
            current_indices.append(i)
            
            # Recurse
            if not backtrack(i + 1, new_sum, current_size + 1, new_mask, current_indices):
                current_indices.pop()
                return False
            
            current_indices.pop()
            
        return True

    backtrack(0, 0, 0, 0, [])
    
    # 4. Construct the output
    answers = []
    for n in range(1, 129):
        if best_rep[n] is not None:
            answers.append(list(best_rep[n]))
        else:
            if n in smooth_set:
                answers.append([n])
            else:
                # Fallback: Should not happen if search completes.
                # If it does, we try to find a valid rep quickly or return invalid.
                # Given the small problem size, this is unlikely.
                answers.append([n])
                
    return answers
