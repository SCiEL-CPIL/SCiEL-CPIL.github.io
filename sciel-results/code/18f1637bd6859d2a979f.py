import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    # Precompute all arithmetic progressions
    # An AP is defined by (start, step, length) where all elements are in [1, n]
    # We represent each AP as a tuple of 0-indexed positions
    # We want to compute signed sums for all APs efficiently

    # Generate all APs
    aps = []
    for step in range(1, n + 1):
        for start in range(n):  # 0-indexed start
            # Maximum length: start + (len-1)*step < n
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                positions = list(range(start, start + length * step, step))
                aps.append(positions)
    
    # Convert to numpy array for efficient computation
    # Each AP has variable length, so we'll handle them individually
    # But we can vectorize the sum computation

    # For each AP, we need to compute sum(signs[i] for i in ap)
    # Let's precompute indices for all APs
    ap_indices = [np.array(ap) for ap in aps]
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute signed sum over all APs."""
        max_abs = 1  # at least 1 (singleton)
        for idx in ap_indices:
            s = np.sum(signs[idx])
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = int(abs_s)
        return max_abs

    # Start with a random assignment
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n)
    current_max = compute_max_abs_sum(signs)
    best_signs = signs.copy()
    best_max = current_max

    # Simulated annealing
    temp = 5.0
    cooling = 0.995
    iterations = 0

    while time.time() < deadline:
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        old_sign = signs[pos]
        new_sign = -old_sign
        signs[pos] = new_sign

        # Compute new max
        new_max = compute_max_abs_sum(signs)

        if new_max <= current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_max - current_max)/temp)
            delta = new_max - current_max
            if delta < temp * 10:  # avoid underflow
                prob = np.exp(-delta / temp)
                if rng.random() < prob:
                    current_max = new_max
                else:
                    signs[pos] = old_sign
            else:
                signs[pos] = old_sign

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

        iterations += 1
        # Check deadline more frequently
        if iterations % 100 == 0 and time.time() > deadline - 2.0:
            break

    # Final polish: try to improve by checking all single flips
    for _ in range(3):
        improved = False
        for pos in range(n):
            old_sign = signs[pos]
            signs[pos] = -old_sign
            new_max = compute_max_abs_sum(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                signs[pos] = old_sign
        if not improved:
            break

    return list(best_signs.astype(int))
