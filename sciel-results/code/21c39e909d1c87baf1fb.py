import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def compute_square_nonzero_count(support, signs, mags):
        """
        Fast computation of nonzero count in square.
        support: sorted list of 12 ints (degrees)
        signs: list of 12 ints (+1 or -1)
        mags: list of 12 positive ints
        """
        deg_sum = {}
        n = len(support)
        # Diagonal terms (i,i)
        for i in range(n):
            d = support[i] + support[i]
            c = signs[i] * mags[i]
            c2 = c * c
            if d in deg_sum:
                deg_sum[d] += c2
            else:
                deg_sum[d] = c2
        
        # Off-diagonal terms (i,j) where i < j
        for i in range(n):
            ci = signs[i] * mags[i]
            for j in range(i + 1, n):
                cj = signs[j] * mags[j]
                d = support[i] + support[j]
                val = 2 * ci * cj
                if d in deg_sum:
                    deg_sum[d] += val
                else:
                    deg_sum[d] = val
        
        count = 0
        for v in deg_sum.values():
            if v != 0:
                count += 1
        return count

    def state_to_coeffs(support, signs, mags):
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for i in range(len(support)):
            coeffs[support[i]] = signs[i] * mags[i]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    # Initialize with a reasonable structured guess
    # Try a few initial configurations and pick the best
    initial_configs = []
    
    # Config 1: Consecutive
    support1 = list(range(12))
    signs1 = [1] * 12
    mags1 = [1] * 12
    initial_configs.append((support1, signs1, mags1))
    
    # Config 2: Even spaced
    support2 = list(range(0, 24, 2))
    signs2 = [1] * 12
    mags2 = [1] * 12
    initial_configs.append((support2, signs2, mags2))
    
    # Config 3: Random
    random.seed(42)
    support3 = sorted(random.sample(range(41), 12))
    signs3 = [random.choice([-1, 1]) for _ in range(12)]
    mags3 = [random.randint(1, 5) for _ in range(12)]
    initial_configs.append((support3, signs3, mags3))
    
    best_init_score = float('inf')
    best_init_state = None
    for sup, sig, mag in initial_configs:
        score = compute_square_nonzero_count(sup, sig, mag)
        if score < best_init_score:
            best_init_score = score
            best_init_state = (sup, sig, mag)
            
    # Simulated Annealing
    current_support, current_signs, current_mags = best_init_state
    current_score = best_init_score
    
    best_support = current_support[:]
    best_signs = current_signs[:]
    best_mags = current_mags[:]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    steps = 0
    max_steps = 500000  # Upper bound, but we stop by time
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        r = random.random()
        
        if r < 0.5:
            # Move a position
            i = random.randint(0, 11)
            old_pos = current_support[i]
            # Try to move to a nearby position or random position
            if random.random() < 0.7:
                new_pos = old_pos + random.randint(-3, 3)
            else:
                new_pos = random.randint(0, 40)
            
            if 0 <= new_pos <= 40:
                # Check if new_pos is already occupied by another term
                if new_pos not in current_support or new_pos == old_pos:
                    new_support = current_support[:]
                    new_support[i] = new_pos
                    new_support.sort()
                    # Ensure no duplicates
                    if len(set(new_support)) == 12:
                        new_score = compute_square_nonzero_count(new_support, current_signs, current_mags)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                            current_support = new_support
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_support = current_support[:]
                                best_signs = current_signs[:]
                                best_mags = current_mags[:]
        elif r < 0.8:
            # Flip a sign
            i = random.randint(0, 11)
            new_signs = current_signs[:]
            new_signs[i] = -new_signs[i]
            new_score = compute_square_nonzero_count(current_support, new_signs, current_mags)
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_signs = new_signs
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_support = current_support[:]
                    best_signs = current_signs[:]
                    best_mags = current_mags[:]
        else:
            # Change a magnitude
            i = random.randint(0, 11)
            new_mag = random.randint(1, 64)
            if new_mag != current_mags[i]:
                new_mags = current_mags[:]
                new_mags[i] = new_mag
                new_score = compute_square_nonzero_count(current_support, current_signs, new_mags)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_mags = new_mags
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_support = current_support[:]
                        best_signs = current_signs[:]
                        best_mags = current_mags[:]
        
        T *= cooling_rate
        steps += 1
        
        # If we've found a very good score, cool down faster
        if best_score <= 12:
            T *= 0.99  # Extra cooling

    # Final local search polish
    if time.time() < deadline - 5:
        # Greedy sign flip
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                new_signs = best_signs[:]
                new_signs[i] = -new_signs[i]
                new_score = compute_square_nonzero_count(best_support, new_signs, best_mags)
                if new_score < best_score:
                    best_signs = new_signs
                    best_score = new_score
                    improved = True
        # Greedy magnitude adjustment (try 1-10)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for new_m in range(1, 11):
                    if new_m == best_mags[i]:
                        continue
                    new_mags = best_mags[:]
                    new_mags[i] = new_m
                    new_score = compute_square_nonzero_count(best_support, best_signs, new_mags)
                    if new_score < best_score:
                        best_mags = new_mags
                        best_score = new_score
                        improved = True
                        break
        # Greedy position move (try nearby positions)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                old_pos = best_support[i]
                for offset in range(-5, 6):
                    if offset == 0:
                        continue
                    new_pos = old_pos + offset
                    if 0 <= new_pos <= 40 and new_pos not in best_support:
                        new_support = best_support[:]
                        new_support[i] = new_pos
                        new_support.sort()
                        if len(set(new_support)) == 12:
                            new_score = compute_square_nonzero_count(new_support, best_signs, best_mags)
                            if new_score < best_score:
                                best_support = new_support
                                best_score = new_score
                                improved = True
                                break

    # Construct final result
    coeffs = state_to_coeffs(best_support, best_signs, best_mags)
    
    # Validate
    if not is_valid(coeffs):
        # Fallback
        coeffs = [1] * 12
    
    return coeffs
