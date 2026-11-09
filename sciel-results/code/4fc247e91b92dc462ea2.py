import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def count_ordered(A_mask):
        """Return numpy array of ordered pair counts for sums 0..256."""
        f = np.fft.rfft(A_mask, n=512)
        conv = np.fft.irfft(f * f, n=512)[:257].round().astype(np.int64)
        return conv

    def count_unordered(conv, A_mask):
        """Convert ordered counts to unordered pair counts for sums 128..256."""
        reps = np.empty(RANGE_SIZE, dtype=np.int64)
        for i in range(RANGE_SIZE):
            s = N_LOW + i
            if s % 2 == 0:
                half = s // 2
                diag = 1 if (half <= MAX_VAL and A_mask[half]) else 0
                reps[i] = (conv[s] + diag) // 2
            else:
                reps[i] = conv[s] // 2
        return reps

    def is_covered(reps):
        return bool(np.all(reps > 0))

    def get_peak(reps):
        return int(np.max(reps))

    def calc_reps(A_mask):
        conv = count_ordered(A_mask)
        return count_unordered(conv, A_mask)

    # Phase 1: Build a valid covering set using a greedy approach
    # We need every sum in [128, 256] to have at least one representation.
    # Start with {0, 128} which covers sum 128 (0+128) and sum 256 (128+128).
    # Then greedily add elements to cover remaining sums.
    
    A_mask = np.zeros(129, dtype=np.int8)
    A_mask[0] = 1
    A_mask[128] = 1
    
    # Find which sums are not yet covered
    def uncovered_sums(A_mask):
        conv = count_ordered(A_mask)
        reps = count_unordered(conv, A_mask)
        return np.where(reps == 0)[0]  # indices into [128..256]
    
    # Greedily add elements that cover the most uncovered sums
    while True:
        uncov = uncovered_sums(A_mask)
        if len(uncov) == 0:
            break
        if time.time() > deadline - 10:
            break
        best_elem = -1
        best_count = 0
        for x in range(129):
            if A_mask[x]:
                continue
            # How many uncovered sums does adding x cover?
            # A sum s is covered if there exists a in A such that s-a in A
            # After adding x, new representations involve x paired with existing elements
            # s = x + a for a in A, or s = a + x (same)
            # So sums newly covered are {x + a for a in A} intersect uncovered
            covered_new = 0
            for idx in uncov:
                s = N_LOW + idx
                other = s - x
                if 0 <= other <= 128 and A_mask[other]:
                    covered_new += 1
            if covered_new > best_count:
                best_count = covered_new
                best_elem = x
        if best_elem >= 0:
            A_mask[best_elem] = 1
        else:
            # Cannot cover any more, add random
            break
    
    # Verify coverage; if not fully covered, add all missing
    uncov = uncovered_sums(A_mask)
    while len(uncov) > 0 and time.time() < deadline - 5:
        # Add element that covers most
        best_elem = -1
        best_count = 0
        for x in range(129):
            if A_mask[x]:
                continue
            covered_new = 0
            for idx in uncov:
                s = N_LOW + idx
                other = s - x
                if 0 <= other <= 128 and A_mask[other]:
                    covered_new += 1
            if covered_new > best_count:
                best_count = covered_new
                best_elem = x
        if best_elem < 0:
            break
        A_mask[best_elem] = 1
        uncov = uncovered_sums(A_mask)
    
    # If still not covered, use dense set as fallback
    if len(uncovered_sums(A_mask)) > 0:
        A_mask = np.zeros(129, dtype=np.int8)
        A_mask[64:] = 1  # [64..128]
    
    current_reps = calc_reps(A_mask)
    current_peak = get_peak(current_reps)
    best_peak = current_peak
    best_mask = A_mask.copy()
    
    if time.time() > deadline - 3:
        return list(np.where(best_mask)[0])
    
    # Phase 2: Local search to reduce peak
    # We'll use a targeted approach: find sums with max multiplicity and try to reduce them.
    
    rng = np.random.default_rng(42)
    
    # Precompute for fast evaluation: given current mask, compute reps
    # For local moves, we can update incrementally
    
    def reps_from_mask(mask):
        conv = count_ordered(mask)
        return count_unordered(conv, mask)
    
    # Try to reduce peak by removing elements that contribute to high-multiplicity sums
    # and adding elements that don't increase peak
    
    T = 1.0
    cooling = 0.999
    min_T = 0.05
    stuck = 0
    
    while time.time() < deadline - 2 and T > min_T:
        # Pick a random element to try removing
        elems = np.where(A_mask)[0]
        if len(elems) <= 20:
            break
        x = int(rng.choice(elems))
        
        # Try removing x
        A_mask[x] = 0
        test_reps = calc_reps(A_mask)
        
        if is_covered(test_reps):
            test_peak = get_peak(test_reps)
            delta = test_peak - current_peak
            if delta <= 0 or (rng.random() < np.exp(-delta / max(T, 0.01))):
                current_reps = test_reps
                current_peak = test_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_mask = A_mask.copy()
                stuck = 0
            else:
                A_mask[x] = 1
                stuck += 1
        else:
            A_mask[x] = 1
            stuck += 1
        
        # Occasionally try adding an element
        if rng.random() < 0.15:
            y = int(rng.integers(0, 129))
            if not A_mask[y]:
                A_mask[y] = 1
                test_reps = calc_reps(A_mask)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    delta = test_peak - current_peak
                    if delta <= 0 or (rng.random() < np.exp(-delta / max(T, 0.01))):
                        current_reps = test_reps
                        current_peak = test_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_mask = A_mask.copy()
                        stuck = 0
                    else:
                        A_mask[y] = 0
                        stuck += 1
                else:
                    A_mask[y] = 0
                    stuck += 1
        
        T *= cooling
        
        if stuck > 300:
            # Reset to best
            A_mask = best_mask.copy()
            current_reps = calc_reps(A_mask)
            current_peak = get_peak(current_reps)
            stuck = 0
            T = max(T, 0.3)
    
    # Phase 3: Greedy removal pass
    if time.time() < deadline - 1:
        final_mask = best_mask.copy()
        final_reps = calc_reps(final_mask)
        final_peak = get_peak(final_reps)
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            elems = list(np.where(final_mask)[0])
            rng.shuffle(elems)
            for x in elems:
                if len(elems) <= 20:
                    break
                final_mask[x] = 0
                test_reps = calc_reps(final_mask)
                if is_covered(test_reps) and get_peak(test_reps) <= final_peak:
                    final_reps = test_reps
                    final_peak = get_peak(test_reps)
                    changed = True
                    if final_peak < best_peak:
                        best_peak = final_peak
                        best_mask = final_mask.copy()
                    break
                else:
                    final_mask[x] = 1
    
    return list(np.where(best_mask)[0])
