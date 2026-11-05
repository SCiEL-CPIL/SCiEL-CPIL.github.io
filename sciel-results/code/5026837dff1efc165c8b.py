import numpy as np
import time
import random

def _count_nonzero_square(coeffs, max_degree=40):
    """Compute the square of a polynomial and count nonzero terms."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Use FFT for speed
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.zeros(size, dtype=np.complex128)
    f[:n] = coeffs
    f2 = np.fft.fft(f)
    f2 *= f2
    result = np.fft.ifft(f2).real
    # Round to nearest integer
    result = np.round(result).astype(np.int64)
    result = result[:2*n-1]
    return np.count_nonzero(result)

def _square_nonzero_count(coeffs):
    """Count nonzero terms in the square, handling the exact integer case."""
    n = len(coeffs)
    if n == 0:
        return 0
    # For small n, direct convolution might be faster and exact
    if n <= 30:
        result = np.convolve(coeffs, coeffs)
        return np.count_nonzero(result)
    else:
        return _count_nonzero_square(coeffs, 2*n-1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: 85s out of 90s
    
    best_count = 23  # Seed value: (1+x+...+x^11)^2 has 23 nonzero terms
    best_poly = [1]*12
    
    # Helper: generate a random 12-term polynomial
    def random_poly():
        # Choose 12 distinct positions in [0, 40]
        positions = sorted(random.sample(range(41), 12))
        coeffs = np.zeros(41, dtype=np.int64)
        for p in positions:
            coeffs[p] = random.choice([-1, 1, 2, -2, 3, -3, 4, -4])
        # Ensure last coefficient is nonzero
        if coeffs[positions[-1]] == 0:
            coeffs[positions[-1]] = 1
        # Trim trailing zeros
        last_nonzero = 40
        while last_nonzero > 0 and coeffs[last_nonzero] == 0:
            last_nonzero -= 1
        return coeffs[:last_nonzero+1]
    
    # Helper: get list of nonzero positions
    def nonzero_positions(coeffs):
        return [i for i, c in enumerate(coeffs) if c != 0]
    
    # Evaluate a polynomial
    def evaluate(coeffs):
        nz = nonzero_positions(coeffs)
        if len(nz) != 12:
            return 1000  # Invalid
        if coeffs[-1] == 0:
            return 1000  # Invalid
        for c in coeffs:
            if c < -64 or c > 64:
                return 1000
        if len(coeffs) > 41:
            return 1000
        return _square_nonzero_count(coeffs)
    
    # Initial random sampling
    random.seed(42)
    np.random.seed(42)
    
    for i in range(200000):
        if time.time() > deadline:
            break
        coeffs = random_poly()
        count = evaluate(coeffs)
        if count < best_count:
            best_count = count
            best_poly = list(coeffs)
    
    # Simulated annealing from best
    if best_count < 23:
        current = np.array(best_poly, dtype=np.int64)
        current_count = best_count
        T = 2.0
        cooling = 0.9999
        
        for i in range(500000):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
            
            # Perturbation: change one coefficient or move one term
            action = random.random()
            nz_pos = nonzero_positions(current)
            
            if action < 0.4 and len(nz_pos) > 0:
                # Change value of one term
                idx = nz_pos[random.randint(0, len(nz_pos)-1)]
                old_val = current[idx]
                new_val = old_val + random.choice([-2,-1,1,2])
                if -64 <= new_val <= 64 and new_val != 0:
                    current[idx] = new_val
                    new_count = _square_nonzero_count(current)
                    delta = new_count - current_count
                    if delta <= 0 or random.random() < np.exp(-delta/T):
                        current_count = new_count
                        if new_count < best_count:
                            best_count = new_count
                            best_poly = list(current)
                    else:
                        current[idx] = old_val
            elif action < 0.6:
                # Move a term to a new position
                idx = nz_pos[random.randint(0, len(nz_pos)-1)]
                val = current[idx]
                current[idx] = 0
                # Find a new position
                new_pos = random.randint(0, 40)
                if current[new_pos] == 0:
                    current[new_pos] = val
                    # Re-trim
                    last_nz = 40
                    while last_nz > 0 and current[last_nz] == 0:
                        last_nz -= 1
                    if len(nonzero_positions(current)) == 12 and current[-1] != 0:
                        new_count = _square_nonzero_count(current)
                        delta = new_count - current_count
                        if delta <= 0 or random.random() < np.exp(-delta/T):
                            current_count = new_count
                            if new_count < best_count:
                                best_count = new_count
                                best_poly = list(current)
                        else:
                            current[new_pos] = 0
                            current[idx] = val
                    else:
                        current[new_pos] = 0
                        current[idx] = val
                else:
                    pass
            elif action < 0.8:
                # Swap two terms
                if len(nz_pos) >= 2:
                    i1, i2 = random.sample(range(len(nz_pos)), 2)
                    p1, p2 = nz_pos[i1], nz_pos[i2]
                    v1, v2 = current[p1], current[p2]
                    current[p1], current[p2] = v2, v1
                    new_count = _square_nonzero_count(current)
                    delta = new_count - current_count
                    if delta <= 0 or random.random() < np.exp(-delta/T):
                        current_count = new_count
                        if new_count < best_count:
                            best_count = new_count
                            best_poly = list(current)
                    else:
                        current[p1], current[p2] = v1, v2
            else:
                # Add/remove term (keep exactly 12)
                if len(nz_pos) == 12:
                    # Remove one, add another
                    idx = nz_pos[random.randint(0, len(nz_pos)-1)]
                    val = current[idx]
                    current[idx] = 0
                    new_pos = random.randint(0, 40)
                    if current[new_pos] == 0:
                        current[new_pos] = random.choice([-1, 1, 2, -2, 3, -3])
                        if len(nonzero_positions(current)) == 12 and current[-1] != 0:
                            new_count = _square_nonzero_count(current)
                            delta = new_count - current_count
                            if delta <= 0 or random.random() < np.exp(-delta/T):
                                current_count = new_count
                                if new_count < best_count:
                                    best_count = new_count
                                    best_poly = list(current)
                            else:
                                current[new_pos] = 0
                                current[idx] = val
                        else:
                            current[new_pos] = 0
                            current[idx] = val
                    else:
                        current[idx] = val
            
            T *= cooling
    
    # Final polish: local search around best
    if best_count < 23:
        current = np.array(best_poly, dtype=np.int64)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            nz_pos = nonzero_positions(current)
            for idx in nz_pos:
                for delta_val in [-3,-2,-1,1,2,3]:
                    old_val = current[idx]
                    new_val = old_val + delta_val
                    if -64 <= new_val <= 64 and new_val != 0:
                        current[idx] = new_val
                        new_count = _square_nonzero_count(current)
                        if new_count < best_count:
                            best_count = new_count
                            best_poly = list(current)
                            improved = True
                        else:
                            current[idx] = old_val
            # Try moving terms
            for idx in nz_pos:
                val = current[idx]
                current[idx] = 0
                for new_pos in range(41):
                    if current[new_pos] == 0:
                        current[new_pos] = val
                        if len(nonzero_positions(current)) == 12 and current[-1] != 0:
                            new_count = _square_nonzero_count(current)
                            if new_count < best_count:
                                best_count = new_count
                                best_poly = list(current)
                                improved = True
                current[new_pos] = 0
                current[idx] = val
    
    # Return best polynomial as list
    return best_poly
