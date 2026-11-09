import time
import random
import numpy as np

def count_nonzero(coeffs):
    """Count nonzero terms in a coefficient list."""
    if isinstance(coeffs, list):
        return sum(1 for c in coeffs if c != 0)
    return np.count_nonzero(coeffs)

def compute_square(coeffs):
    """Compute the exact square of a polynomial using numpy convolution."""
    arr = np.array(coeffs, dtype=np.int64)
    return np.convolve(arr, arr)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline (5s safety margin)
    
    # Initial candidate: P(x) = 1 + x + ... + x^11
    # Square has 23 nonzero terms
    best_coeffs = [1] * 12
    best_square = compute_square(best_coeffs)
    best_nonzero = count_nonzero(best_square)
    
    # Try a few specific analytical candidates that might have cancellations
    candidates = [
        [1] * 11 + [-1],
        [1] * 6 + [-1] * 6,
        [1 if i % 2 == 0 else -1 for i in range(12)],
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
    
    for cand in candidates:
        if len(cand) <= 41 and count_nonzero(cand) == 12 and cand[-1] != 0:
            sq = compute_square(cand)
            nz = count_nonzero(sq)
            if nz < best_nonzero:
                best_nonzero = nz
                best_coeffs = cand[:]
                best_square = sq

    random.seed(42)
    valid_values = [i for i in range(-64, 65) if i != 0]
    
    current_coeffs = best_coeffs[:]
    current_nonzero = best_nonzero
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Random restart
            if random.random() < 0.5:
                current_coeffs = best_coeffs[:]
                current_nonzero = best_nonzero
            else:
                # Generate new random valid candidate
                L = random.randint(12, 41)
                last_idx = L - 1
                other_indices = random.sample(range(L - 1), 11)
                non_zero_indices = set(other_indices) | {last_idx}
                current_coeffs = [0] * L
                for idx in non_zero_indices:
                    current_coeffs[idx] = random.choice(valid_values)
                current_square = compute_square(current_coeffs)
                current_nonzero = count_nonzero(current_square)
                if current_nonzero < best_nonzero:
                    best_nonzero = current_nonzero
                    best_coeffs = current_coeffs[:]
                    best_square = current_square
                    no_improve_count = 0
            no_improve_count = 0
            continue
        
        prev_coeffs = current_coeffs[:]
        
        # Mutation
        mutation_type = random.choice([0, 1, 2])
        
        if mutation_type == 0:
            # Change value of a random non-zero term
            non_zero_indices = [i for i, c in enumerate(current_coeffs) if c != 0]
            if not non_zero_indices:
                no_improve_count += 1
                continue
            idx = random.choice(non_zero_indices)
            old_val = current_coeffs[idx]
            new_val = random.choice(valid_values)
            if new_val == old_val:
                no_improve_count += 1
                continue
            current_coeffs[idx] = new_val
            
        elif mutation_type == 1:
            # Move a term: set one non-zero to 0, set one zero to non-zero
            non_zero_indices = [i for i, c in enumerate(current_coeffs) if c != 0]
            zero_indices = [i for i, c in enumerate(current_coeffs) if c == 0]
            if not non_zero_indices or not zero_indices:
                no_improve_count += 1
                continue
            
            idx_to_remove = random.choice(non_zero_indices)
            idx_to_add = random.choice(zero_indices)
            
            val_to_move = current_coeffs[idx_to_remove]
            
            current_coeffs[idx_to_remove] = 0
            current_coeffs[idx_to_add] = val_to_move
            
            # Validate: final coefficient must be non-zero
            if current_coeffs[-1] == 0:
                current_coeffs = prev_coeffs[:]
                no_improve_count += 1
                continue
            
            # Validate: exactly 12 non-zero terms
            if count_nonzero(current_coeffs) != 12:
                current_coeffs = prev_coeffs[:]
                no_improve_count += 1
                continue
                
        else:
            # Shift exponent: move a term to a new random position
            non_zero_indices = [i for i, c in enumerate(current_coeffs) if c != 0]
            if not non_zero_indices:
                no_improve_count += 1
                continue
            
            # Do not move the term at the last index if it's the only one ensuring final coeff is non-zero?
            # Actually, if we move the term at L-1, L-1 becomes 0.
            # So we must not move the term at L-1 unless we put something else there.
            # Simplest: pick a term that is NOT at L-1.
            movable_indices = [i for i in non_zero_indices if i != len(current_coeffs) - 1]
            if not movable_indices:
                # All non-zero terms are at L-1? Impossible with 12 terms and L>=12.
                # Unless L=12 and only one term is at L-1 and others are... wait.
                # If L=12, indices 0..11. Only 11 is L-1.
                # So movable_indices will have 11 elements.
                no_improve_count += 1
                continue
                
            idx = random.choice(movable_indices)
            val = current_coeffs[idx]
            current_coeffs[idx] = 0
            
            new_idx = random.randint(0, len(current_coeffs) - 1)
            
            current_coeffs[new_idx] = val
            
            # If we moved to L-1, that's fine.
            # If we moved from somewhere else, L-1 is untouched (was non-zero).
            # So final coeff is still non-zero.
            
            if count_nonzero(current_coeffs) != 12:
                current_coeffs = prev_coeffs[:]
                no_improve_count += 1
                continue

        # Evaluate
        current_square = compute_square(current_coeffs)
        current_nonzero = count_nonzero(current_square)
        
        if current_nonzero < best_nonzero:
            best_nonzero = current_nonzero
            best_coeffs = current_coeffs[:]
            best_square = current_square
            no_improve_count = 0
        else:
            no_improve_count += 1
            current_coeffs = prev_coeffs[:]
            current_nonzero = best_nonzero # Reset to best if we reverted? 
            # Actually, if we revert, current_coeffs is prev_coeffs.
            # prev_coeffs was the one that led to current_nonzero (which was >= best).
            # So we just keep current_coeffs as prev_coeffs.
            
    return best_coeffs
