import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_count = 23  # AP gives 23 as baseline
    best_poly = None
    
    # Start with a known good candidate: arithmetic progression
    # P = 1 + x + x^2 + ... + x^11 -> square has 23 terms
    # But try structured patterns
    
    # Try the "two-block" pattern: terms at 0,1,2,3, 6,7,8,9, 12,13,14,15
    # This might create cancellations if coefficients alternate
    test_patterns = []
    
    # Pattern 1: AP with all 1s
    test_patterns.append(list(range(12)))
    
    # Pattern 2: two blocks
    test_patterns.append([0,1,2,3, 6,7,8,9, 12,13,14,15])
    
    # Pattern 3: geometric-like
    test_patterns.append([0,1,2,4,5,6,8,9,10,12,13,14])
    
    # Pattern 4: spread out
    test_patterns.append([0,1,3,4,6,7,9,10,12,13,15,16])
    
    def count_nonzero_square(positions, coeffs):
        """Compute number of nonzero terms in P^2."""
        # Build coefficient array
        if not positions:
            return 0, []
        max_deg = max(positions)
        arr = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for p, c in zip(positions, coeffs):
            arr[p] = c
        sq = np.convolve(arr, arr)
        return np.count_nonzero(sq), sq
    
    # Evaluate initial patterns with all coefficients = 1
    for positions in test_patterns:
        if time.time() > deadline:
            break
        coeffs = [1] * 12
        cnt, _ = count_nonzero_square(positions, coeffs)
        if cnt < best_count:
            best_count = cnt
            best_poly = (positions, coeffs)
    
    # Random search: try random positions and coefficients
    # Coefficients from {-2, -1, 1, 2}
    coeff_choices = [-2, -1, 1, 2]
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Generate random 12-term polynomial
        # Strategy: sometimes use structured positions, sometimes random
        if iteration % 3 == 0:
            # Structured: two or three blocks
            n_blocks = random.choice([2, 3])
            if n_blocks == 2:
                # Two blocks of 6
                gap = random.randint(1, 8)
                base = random.randint(0, 5)
                positions = list(range(base, base + 6)) + list(range(base + 6 + gap, base + 12 + gap))
            elif n_blocks == 3:
                # Three blocks of 4
                gap1 = random.randint(1, 5)
                gap2 = random.randint(1, 5)
                base = random.randint(0, 3)
                positions = list(range(base, base + 4)) + \
                           list(range(base + 4 + gap1, base + 8 + gap1)) + \
                           list(range(base + 8 + gap1 + gap2, base + 12 + gap1 + gap2))
            else:
                positions = list(range(12))
        else:
            # Random positions, ensure they're distinct and sorted
            positions = sorted(random.sample(range(0, 40), 12))
        
        # Random coefficients
        coeffs = [random.choice(coeff_choices) for _ in range(12)]
        
        # Ensure at least one coefficient is nonzero (always true with our choices)
        # Ensure the leading coefficient is nonzero (it is)
        
        cnt, _ = count_nonzero_square(positions, coeffs)
        
        if cnt < best_count:
            best_count = cnt
            best_poly = (positions, coeffs)
            
            # If we found something great, do local search around it
            if best_count <= 14:
                # Local search: try changing one coefficient
                for i in range(12):
                    for new_c in coeff_choices:
                        if new_c == coeffs[i]:
                            continue
                        new_coeffs = coeffs.copy()
                        new_coeffs[i] = new_c
                        cnt2, _ = count_nonzero_square(positions, new_coeffs)
                        if cnt2 < best_count:
                            best_count = cnt2
                            best_poly = (positions, new_coeffs)
                            coeffs = new_coeffs
                # Try moving one position
                for i in range(12):
                    for delta in [-2, -1, 1, 2]:
                        new_pos = positions[i] + delta
                        if new_pos < 0 or new_pos >= 40:
                            continue
                        new_positions = positions.copy()
                        new_positions[i] = new_pos
                        if len(set(new_positions)) != 12:
                            continue
                        new_positions.sort()
                        cnt2, _ = count_nonzero_square(new_positions, coeffs)
                        if cnt2 < best_count:
                            best_count = cnt2
                            best_poly = (new_positions, coeffs)
                            positions = new_positions
    
    # Build the final coefficient list
    if best_poly is None:
        # Fallback: simple AP
        return [1] * 12
    
    positions, coeffs = best_poly
    max_deg = max(positions)
    result = [0] * (max_deg + 1)
    for p, c in zip(positions, coeffs):
        result[p] = c
    
    # Trim trailing zeros (shouldn't be any since last position has nonzero coeff)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Validate: exactly 12 nonzero terms
    assert sum(1 for x in result if x != 0) == 12, f"Expected 12 nonzero terms, got {sum(1 for x in result if x != 0)}"
    assert result[-1] != 0, "Final coefficient must be nonzero"
    assert len(result) <= 41, f"Length {len(result)} exceeds 41"
    assert all(-64 <= x <= 64 for x in result), "Coefficients out of range"
    
    return result
