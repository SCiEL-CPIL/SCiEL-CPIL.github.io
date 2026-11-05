import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)

    def fast_objective(x_bin):
        """
        Compute peak representation multiplicity using convolution.
        x_bin is a binary numpy array of length 129.
        Returns (peak, is_valid).
        """
        if not np.any(x_bin):
            return float('inf'), False
        
        # Convolution: (x * x)[s] counts ordered pairs (i,j) with i+j=s
        # For unordered pairs with equal summands allowed:
        # count_unordered[s] = (ordered_count[s] + (s even ? x[s/2] : 0)) / 2
        # But we need to be careful: ordered counts (i,j) and (j,i) as separate
        # When i=j, it's counted once in ordered, but should count once in unordered
        # When i≠j, ordered counts both (i,j) and (j,i), unordered counts once
        
        # Actually, let's think more carefully:
        # ordered_conv[s] = sum_i x[i] * x[s-i] for all i
        # This counts (i,j) where i+j=s, including both (i,j) and (j,i) when i≠j
        # For i=j, it's counted once
        
        # Unordered count = (ordered_conv[s] + diagonal[s]) / 2
        # where diagonal[s] = x[s/2] if s is even, else 0
        # Wait, that's not quite right either.
        
        # Let me reconsider:
        # ordered_conv[s] = number of ordered pairs (a,b) in A×A with a+b=s
        # = sum_{a in A} [s-a in A]
        # 
        # For unordered pairs with repetition:
        # If a < b: counted once in unordered, twice in ordered
        # If a = b: counted once in unordered, once in ordered
        # 
        # So: unordered[s] = (ordered[s] + diag[s]) / 2
        # where diag[s] = 1 if s is even and s/2 in A, else 0
        # Because: ordered = 2*(#pairs with a<b) + (#pairs with a=b)
        #           unordered = (#pairs with a<b) + (#pairs with a=b)
        #           So unordered = (ordered + #pairs with a=b) / 2
        
        conv = np.convolve(x_bin, x_bin, mode='full')
        # conv[s] = ordered pairs summing to s, for s from 0 to 256
        
        # Extract target range
        conv_target = conv[min_sum:max_sum+1].astype(np.float64)
        
        # Add diagonal correction
        diag = np.zeros(max_sum - min_sum + 1)
        for idx, s in enumerate(target_sums):
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n:
                    diag[idx] = x_bin[mid]
        
        unordered_counts = (conv_target + diag) / 2.0
        unordered_counts = np.round(unordered_counts).astype(np.int32)
        
        # Check coverage
        if np.any(unordered_counts == 0):
            return float('inf'), False
        
        peak = int(np.max(unordered_counts))
        return peak, True

    def compute_peak(A_set):
        """Compute peak from a set."""
        x_bin = np.zeros(129, dtype=np.float64)
        for x in A_set:
            x_bin[x] = 1.0
        peak, valid = fast_objective(x_bin)
        return peak

    # Initialize with a good starting point
    # Try structured initializations
    best_initial_peak = float('inf')
    best_initial_A = None

    # Try two-interval structures
    for a in range(0, 65):
        for b in range(64, 129):
            if a >= b:
                continue
            A = set(range(0, a + 1)) | set(range(b, 129))
            peak = compute_peak(A)
            if peak < best_initial_peak:
                best_initial_peak = peak
                best_initial_A = A.copy()

    # Also try single interval
    for a in range(0, 65):
        for b in range(64, 129):
            A = set(range(a, b + 1))
            peak = compute_peak(A)
            if peak < best_initial_peak:
                best_initial_peak = peak
                best_initial_A = A.copy()

    if best_initial_A is None:
        best_initial_A = set(range(64, 129))
        best_initial_peak = compute_peak(best_initial_A)

    # Simulated Annealing
    current_A = best_initial_A.copy()
    current_peak = best_initial_peak
    best_A = current_A.copy()
    best_peak = current_peak

    random.seed(42)
    np.random.seed(42)

    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    max_iters = 200000

    iter_count = 0
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
        iter_count += 1

        # Generate a move
        move_type = random.random()

        if move_type < 0.3:
            # Add an element
            candidates_add = [x for x in range(129) if x not in current_A]
            if not candidates_add:
                continue
            new_elem = random.choice(candidates_add)
            new_A = current_A.copy()
            new_A.add(new_elem)

        elif move_type < 0.6:
            # Remove an element
            if len(current_A) <= 5:
                continue
            new_elem = random.choice(list(current_A))
            new_A = current_A.copy()
            new_A.discard(new_elem)

        else:
            # Swap: remove one, add one
            if len(current_A) <= 5:
                continue
            remove_elem = random.choice(list(current_A))
            candidates_add = [x for x in range(129) if x not in current_A]
            if not candidates_add:
                continue
            add_elem = random.choice(candidates_add)
            new_A = current_A.copy()
            new_A.discard(remove_elem)
            new_A.add(add_elem)

        new_peak = compute_peak(new_A)

        # Acceptance criterion
        if new_peak < current_peak:
            current_A = new_A
            current_peak = new_peak
            no_improve = 0
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A.copy()
        else:
            # Accept with probability exp(-(new_peak - current_peak) / T)
            delta = new_peak - current_peak
            if T > T_min and random.random() < np.exp(-delta / T):
                current_A = new_A
                current_peak = new_peak
            no_improve += 1

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Return the best solution found
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))

    return sorted(best_A)
