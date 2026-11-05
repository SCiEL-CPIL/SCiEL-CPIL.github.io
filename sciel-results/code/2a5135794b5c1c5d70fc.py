"""Additive bases with few representations – max-flow based optimization."""
import numpy as np
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = 128
    # Target sums: [128, 256], 129 values
    # A is a subset of [0, 128], 129 possible elements
    # Unordered pairs (a,b) with a<=b, a,b in A, a+b in [128,256]

    def compute_counts(A):
        """Compute multiplicity of each sum in [128, 256] using FFT convolution."""
        if len(A) == 0:
            return np.zeros(129, dtype=np.int64)
        # Create indicator array
        arr = np.zeros(257, dtype=np.float64)
        for a in A:
            arr[a] = 1.0
        # Convolution: (arr * arr)[s] = number of ordered pairs (a,b) with a+b=s
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s
        # For unordered pairs with a<=b:
        #   if s is even: unordered_count = (conv[s] + 1) // 2  (the diagonal a=b=s/2 adds 1 ordered pair)
        #   if s is odd:  unordered_count = conv[s] // 2
        # Actually: conv[s] = sum_{a+b=s} 1 = number of ordered pairs
        # Unordered with a<=b: for each pair {a,b} with a<b, it contributes 2 to conv[s]
        # For a=b=s/2 (only if s even), it contributes 1 to conv[s]
        # So unordered_count = (conv[s] + (1 if s even and s/2 in A else 0)) / 2
        # Simpler: unordered = (conv[s] + diag) / 2 where diag = 1 if s/2 in A and s even
        counts = np.zeros(129, dtype=np.int64)
        for i, s in enumerate(range(128, 257)):
            ordered = int(round(conv[s]))
            # Diagonal contribution: a=b=s/2, only if s is even and s/2 in A
            diag = 0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and half in A_set:
                    diag = 1
            counts[i] = (ordered + diag) // 2
        return counts

    def compute_counts_fast(A, A_set):
        """Fast count computation using precomputed structure."""
        if len(A) == 0:
            return np.zeros(129, dtype=np.int64)
        arr = np.zeros(257, dtype=np.float64)
        for a in A:
            arr[a] = 1.0
        conv = np.convolve(arr, arr, mode='full')
        counts = np.zeros(129, dtype=np.int64)
        for i, s in enumerate(range(128, 257)):
            ordered = int(round(conv[s]))
            diag = 0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and half in A_set:
                    diag = 1
            counts[i] = (ordered + diag) // 2
        return counts

    def is_valid(A):
        A_set = set(A)
        counts = compute_counts_fast(A, A_set)
        return bool(np.all(counts > 0))

    def peak_multiplicity(A):
        A_set = set(A)
        counts = compute_counts_fast(A, A_set)
        return int(counts.max())

    def objective(A):
        A_set = set(A)
        counts = compute_counts_fast(A, A_set)
        uncovered = int(np.sum(counts == 0))
        peak = int(counts.max())
        return peak + 1000 * uncovered

    # Maximum flow based feasibility check
    # Given a candidate peak T, check if we can assign pairs to cover all sums
    # with each sum having at most T representations, and each element used
    # in a bounded way.
    # 
    # Actually, the max flow approach: 
    # We want to find if there exists a set A such that every sum in [128,256]
    # is covered with multiplicity <= T.
    # 
    # Alternative: given A, the peak is determined. So we use max flow to
    # guide the search: find A that minimizes peak.
    # 
    # Better approach: Use the structure to find good A directly.
    # Key insight: symmetric sets work well. If A is symmetric around 64,
    # i.e., x in A iff 128-x in A, then sums are symmetric around 128.
    # 
    # Let's try: A = {a_1, ..., a_k} where we choose elements to balance coverage.

    # Strategy: 
    # 1. Use max-flow to find minimum T such that a valid A exists
    # 2. Then find such A using local search

    # For max flow: 
    # Left nodes: target sums s_1=128, ..., s_129=256
    # Right nodes: elements x_1=0, ..., x_129=128
    # Edge (s, x) exists if s-x in [0,128] and s-x <= x (to avoid double counting)
    #   Actually for unordered: edge if there exists y in [0,128] with y>=x and x+y=s
    #   i.e., x <= s/2 and s-x <= 128
    # Capacity on left nodes: 1 (each sum needs at least 1 representation)
    # Capacity on right nodes: T (each element can be in at most T pairs? No...)
    # 
    # This formulation is tricky. Let me think differently.
    #
    # The peak multiplicity of sum s is the number of pairs (a,b) in A with a<=b and a+b=s.
    # We want to minimize max_s (number of such pairs).
    #
    # For a given A, this is fixed. So the optimization is over A.
    #
    # Let's use a different approach: 
    # 1. Start with a good initial A
    # 2. Use local search to minimize peak
    # 3. Use the max flow idea to check if we can do better

    # Good initial solution: use a structured set
    # If A = {0, 1, ..., m} union {128-m, ..., 128}, sums cover a range
    # Let's try A with elements spread across [0, 128]

    # Initialize with a heuristic set
    def heuristic_init():
        """Create an initial set that covers [128, 256]."""
        # Use a grid-like structure
        A = set()
        # Add elements at regular intervals
        for i in range(0, 129, 8):
            A.add(i)
        # Add some to fill gaps
        for i in range(4, 129, 8):
            A.add(i)
        return sorted(A)

    # Local search with efficient evaluation
    def local_search(A, max_time=30.0):
        """Improve A by local moves."""
        A = sorted(set(A))
        A_set = set(A)
        best_obj = objective(A)
        best_A = A[:]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try random swaps
            for _ in range(50):
                if time.time() > max_time:
                    break
                idx = random.randint(0, len(A) - 1)
                candidate = random.randint(0, 128)
                if candidate in A_set:
                    continue
                new_A = A[:idx] + [candidate] + A[idx+1:]
                new_A_set = A_set - {A[idx]} | {candidate}
                new_obj = objective(new_A)
                if new_obj < best_obj:
                    A = new_A
                    A_set = new_A_set
                    best_obj = new_obj
                    best_A = A[:]
                    improved = True
                    break
            # Try random additions
            for _ in range(20):
                if time.time() > max_time:
                    break
                candidate = random.randint(0, 128)
                if candidate in A_set:
                    continue
                new_A = A + [candidate]
                new_obj = objective(new_A)
                if new_obj < best_obj:
                    A = new_A
                    A_set = set(A)
                    best_obj = new_obj
                    best_A = A[:]
                    improved = True
                    break
            # Try random removals (if we have more than minimum)
            if len(A) > 18:
                for _ in range(10):
                    if time.time() > max_time:
                        break
                    idx = random.randint(0, len(A) - 1)
                    new_A = A[:idx] + A[idx+1:]
                    new_obj = objective(new_A)
                    if new_obj < best_obj:
                        A = new_A
                        A_set = set(A)
                        best_obj = new_obj
                        best_A = A[:]
                        improved = True
                        break
        return best_A, best_obj

    # Simulated annealing
    def simulated_annealing(A, time_limit):
        """SA to minimize peak multiplicity."""
        A = sorted(set(A))
        A_set = set(A)
        best_A = A[:]
        best_obj = objective(A)
        current_obj = best_obj
        T = 5.0
        cooling = 0.9995
        min_T = 0.01

        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            if move_type < 0.3 and len(A) > 18:
                idx = random.randint(0, len(A) - 1)
                new_A = A[:idx] + A[idx+1:]
            elif move_type < 0.55:
                candidate = random.randint(0, 128)
                if candidate in A_set:
                    T *= cooling
                    continue
                new_A = A + [candidate]
            else:
                if len(A) < 1:
                    T *= cooling
                    continue
                idx = random.randint(0, len(A) - 1)
                candidate = random.randint(0, 128)
                if candidate in A_set:
                    T *= cooling
                    continue
                new_A = A[:idx] + [candidate] + A[idx+1:]

            new_obj = objective(new_A)
            delta = new_obj - current_obj

            if delta <= 0 or random.random() < np.exp(-delta / T):
                A = new_A
                A_set = set(A)
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = A[:]

            T *= cooling

        return best_A, best_obj

    # Run optimization
    best_A = None
    best_score = float('inf')

    # Try multiple initializations
    for restart in range(3):
        if time.time() > deadline - 20:
            break

        # Different initial strategies
        if restart == 0:
            A = heuristic_init()
        elif restart == 1:
            # More spread out
            A = sorted(random.sample(range(0, 129), 25))
        else:
            # Symmetric around 64
            half = sorted(random.sample(range(0, 65), 15))
            A = sorted(set(half + [128 - x for x in half]))

        A_set = set(A)

        # Ensure validity
        if not is_valid(A):
            # Repair
            counts = compute_counts_fast(A, A_set)
            for _ in range(100):
                uncovered = np.where(counts == 0)[0] + 128
                if len(uncovered) == 0:
                    break
                target = random.choice(list(uncovered))
                for _ in range(30):
                    candidate = random.randint(0, 128)
                    if candidate not in A_set:
                        new_A = A + [candidate]
                        new_A_set = A_set | {candidate}
                        new_counts = compute_counts_fast(new_A, new_A_set)
                        if np.sum(new_counts == 0) < np.sum(counts == 0):
                            A = new_A
                            A_set = new_A_set
                            counts = new_counts
                            break

        if not is_valid(A):
            continue

        # Local search
        sa_time = min(20, deadline - time.time() - 10)
        if sa_time > 5:
            A, obj = simulated_annealing(A, time.time() + sa_time)
        else:
            A, obj = local_search(A, max_time=sa_time)

        if obj < best_score:
            best_score = obj
            best_A = A[:]

    # Final polish
    if best_A is not None and time.time() < deadline - 5:
        best_A, final_obj = local_search(best_A, max_time=deadline - time.time() - 3)
        if final_obj < best_score:
            best_score = final_obj

    # Fallback
    if best_A is None or not is_valid(best_A):
        # Return a known valid solution
        A = list(range(64, 129))
        return A

    return sorted(best_A)
