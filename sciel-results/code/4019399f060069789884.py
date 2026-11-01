import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # Vectorized sumset and diffset
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    def lfsr_sequence(polynomial, initial_state, length):
        """Generate an LFSR sequence using the given polynomial and initial state."""
        degree = polynomial.bit_length() - 1
        state = initial_state & ((1 << degree) - 1)
        if state == 0:
            state = 1  # Avoid all-zero state
        sequence = []
        for _ in range(length):
            sequence.append(state)
            # Compute feedback bit
            feedback = 0
            for i in range(degree):
                if (polynomial >> i) & 1:
                    feedback ^= (state >> i) & 1
            # Shift and insert feedback
            state = ((state >> 1) | (feedback << (degree - 1))) & ((1 << degree) - 1)
        return sequence
    
    def de_bruijn_positions(polynomial, initial_state, k, window_size):
        """Extract positions from de Bruijn-like sequence where k-bit window has specific property."""
        degree = polynomial.bit_length() - 1
        state = initial_state & ((1 << degree) - 1)
        if state == 0:
            state = 1
        positions = [0]
        period = (1 << degree) - 1
        for i in range(1, period + 1):
            # Check if current window has exactly one 1
            window = state & ((1 << window_size) - 1)
            if bin(window).count('1') == 1:
                positions.append(i)
            # Update state
            feedback = 0
            for j in range(degree):
                if (polynomial >> j) & 1:
                    feedback ^= (state >> j) & 1
            state = ((state >> 1) | (feedback << (degree - 1))) & ((1 << degree) - 1)
        return sorted(set(positions))
    
    # Phase 1: Try known good Golomb rulers as initial candidates
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 31, 34],
        [0, 1, 4, 10, 18, 23, 31, 42, 45],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 54],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 65],
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline - 15:
            break
        try_set(gr)
    
    # Phase 2: Exhaustive search for small M
    for M in range(1, 14):
        if time.time() > deadline - 15:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 15:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 3: LFSR-based structured search
    # Try various LFSR configurations to generate structured sets
    lfsr_configs = []
    for degree in range(4, 12):
        for poly in range(1 << degree, 1 << (degree + 1)):
            if (poly & 1) == 0:  # Must be odd (has constant term)
                continue
            # Check if polynomial is primitive (simplified check)
            # For small degrees, we can test by generating the sequence
            lfsr_configs.append((degree, poly))
    
    # Limit number of LFSR configs to try
    random.shuffle(lfsr_configs)
    lfsr_configs = lfsr_configs[:200]
    
    for degree, poly in lfsr_configs:
        if time.time() > deadline - 20:
            break
        for window_size in range(2, min(degree + 1, 8)):
            if time.time() > deadline - 20:
                break
            for initial_state in range(1, min(1 << degree, 50)):
                if time.time() > deadline - 20:
                    break
                positions = de_bruijn_positions(poly, initial_state, window_size, window_size)
                if len(positions) >= 2:
                    # Scale and offset to create valid set
                    for scale in [1, 2, 3]:
                        cand = [p * scale for p in positions]
                        try_set(cand)
    
    # Phase 4: Heuristic Search with multiple strategies
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current) if current else 0
        
        # Strategy 1: Try adding an element - more targeted search
        if not improved:
            candidates_to_add = set()
            # Nearby positions
            for i in range(1, max_cur + 50):
                if i not in current:
                    candidates_to_add.add(i)
            # Larger jumps based on current structure
            if len(current) > 2:
                for _ in range(50):
                    candidates_to_add.add(random.randint(max_cur + 1, max_cur + 300))
            # Try positions that create interesting difference patterns
            for _ in range(30):
                candidates_to_add.add(random.randint(1, max_cur + 100))
                
            for new_el in sorted(candidates_to_add):
                if time.time() > deadline - 10:
                    break
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 2: Try replacing an element with better neighborhood search
        if not improved and len(current) >= 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Wider neighborhood for replacement
                for delta in range(-50, 51):
                    if time.time() > deadline - 10:
                        break
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Strategy 3: Try removing an element
        if not improved and len(current) > 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 4: Try swapping two elements
        if not improved and len(current) > 3:
            for i in range(1, len(current)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 10:
                        break
                    cand = list(current)
                    cand[i], cand[j] = cand[j], cand[i]
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Strategy 5: LFSR-inspired local moves - replace subset with LFSR-generated positions
        if not improved and len(current) >= 4:
            # Try replacing a contiguous block of elements with LFSR-generated positions
            for start_idx in range(1, len(current) - 2):
                if time.time() > deadline - 10:
                    break
                for block_size in [2, 3, 4]:
                    if start_idx + block_size > len(current):
                        break
                    # Generate LFSR positions for this block
                    for degree in [4, 5, 6]:
                        for poly in [0b11011, 0b10111, 0b11101, 0b10011, 0b11010, 0b10110]:
                            if degree != poly.bit_length() - 1:
                                continue
                            positions = de_bruijn_positions(poly, 1, 2, 2)
                            if len(positions) >= block_size:
                                # Scale to fit in the range
                                old_elements = current[start_idx:start_idx + block_size]
                                if old_elements:
                                    min_old = min(old_elements)
                                    max_old = max(old_elements)
                                    if max_old > min_old:
                                        scale = (max_old - min_old) / max(len(positions) - 1, 1)
                                        new_elements = [int(min_old + p * scale) for p in positions[:block_size]]
                                        # Ensure uniqueness and validity
                                        new_elements = sorted(set([max(1, e) for e in new_elements]))
                                        if len(new_elements) == block_size:
                                            cand = current[:start_idx] + new_elements + current[start_idx + block_size:]
                                            cand = sorted(set(cand))
                                            if 0 in cand and len(cand) == len(current):
                                                s = compute_score(cand)
                                                if s > best_score:
                                                    best_score = s
                                                    best_set = list(cand)
                                                    current = list(cand)
                                                    improved = True
                                                    no_improve = 0
                                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local optima
            if random.random() < 0.7:
                perturbation_type = random.random()
                if perturbation_type < 0.25:
                    # Swap two random elements
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        j = random.randint(1, len(current)-1)
                        if i != j:
                            cand = list(current)
                            cand[i], cand[j] = cand[j], cand[i]
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                elif perturbation_type < 0.5:
                    # Add a random element
                    new_el = random.randint(1, max_cur + 150)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
                elif perturbation_type < 0.75:
                    # Replace a random element
                    if len(current) > 1:
                        i = random.randint(1, len(current)-1)
                        new_el = random.randint(1, max_cur + 80)
                        if new_el not in current:
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                else:
                    # Remove a random element
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        cand = [x for j, x in enumerate(current) if j != i]
                        try_set(cand)
                        current = list(best_set)
    
    return list(best_set)
