import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n

    # Precompute FFT size
    fft_size = 1 << (2 * n + 1).bit_length()

    def compute_counts_fft(A):
        """Compute pair-sum counts using FFT. A is a boolean array of length n+1."""
        f = np.zeros(fft_size, dtype=np.float64)
        f[:len(A)] = A
        f_conv = np.fft.ifft(np.fft.fft(f) ** 2).real
        # f_conv[s] = sum_{a+b=s} f[a]*f[b] = ordered count
        # For unordered: count_unordered[s] = (ordered[s] + (1 if s even and f[s//2] else 0)) / 2
        # Actually: ordered[s] = 2 * unordered[s] - (1 if s even and f[s//2]==1 else 0)
        # So unordered[s] = (ordered[s] + diag[s]) / 2 where diag[s] = 1 if s even and f[s//2]==1
        diag = np.zeros(max_sum + 1)
        for s in range(0, max_sum + 1, 2):
            if s // 2 <= n and A[s // 2]:
                diag[s] = 1.0
        counts = (f_conv[:max_sum + 1] + diag) / 2.0
        return np.round(counts).astype(int)

    def compute_counts_direct(A_set):
        """Direct O(k^2) count computation for small sets."""
        counts = np.zeros(max_sum + 1, dtype=int)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if s <= max_sum:
                    counts[s] += 1
        return counts

    def get_peak(counts):
        """Get peak multiplicity in target range."""
        return int(np.max(counts[target_lo:target_hi + 1]))

    def get_uncovered(counts):
        """Count uncovered targets."""
        return int(np.sum(counts[target_lo:target_hi + 1] == 0))

    def score_tuple(counts):
        """Return (uncovered, peak, sum_sq) for comparison."""
        t = counts[target_lo:target_hi + 1]
        unc = int(np.sum(t == 0))
        pk = int(np.max(t))
        sq = int(np.sum(t * t))
        return (unc, pk, sq)

    # Phase 1: Deterministic greedy minimax construction
    # Start with a small seed, then greedily add elements to minimize peak
    def greedy_minimax_build(seed_size=10, max_size=80):
        """
        Deterministic greedy: at each step, try all candidates and pick the one
        that gives the best (uncovered, peak) score.
        """
        A_bool = np.zeros(n + 1, dtype=np.float64)
        A_set = set()

        # Initial seed: spread out for coverage
        # Use a structured seed: every other element
        for i in range(0, n + 1, 2):
            if len(A_set) < seed_size:
                A_set.add(i)
                A_bool[i] = 1.0

        # Also add some from the other parity
        for i in range(1, n + 1, 2):
            if len(A_set) < seed_size:
                A_set.add(i)
                A_bool[i] = 1.0

        counts = compute_counts_direct(A_set)
        curr_score = score_tuple(counts)

        for step in range(max_size - len(A_set)):
            if time.time() > deadline - 20.0:
                break

            best_y = None
            best_score = curr_score

            # Try all candidates
            for y in range(n + 1):
                if y in A_set:
                    continue
                # Quick check: will this help coverage?
                A_set.add(y)
                A_bool[y] = 1.0
                counts_new = compute_counts_direct(A_set)
                score_new = score_tuple(counts_new)
                A_set.remove(y)
                A_bool[y] = 0.0

                if score_new < best_score:
                    best_score = score_new
                    best_y = y

            if best_y is not None:
                A_set.add(best_y)
                A_bool[best_y] = 1.0
                counts = compute_counts_direct(A_set)
                curr_score = best_score
            else:
                break

        return A_set, curr_score

    # Phase 2: Deterministic hill-climbing polish
    def hill_climb_polish(A_set, max_rounds=50):
        """
        Deterministic local search: repeatedly try all single swaps
        (remove one, add another) and accept if it improves the score.
        """
        A_set = set(A_set)
        counts = compute_counts_direct(A_set)
        curr_score = score_tuple(counts)
        best_score = curr_score
        best_A = set(A_set)

        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break

            improved = False
            A_list = sorted(A_set)

            for x in A_list:
                if time.time() > deadline - 5.0:
                    break

                # Try removing x and adding each y
                for y in range(n + 1):
                    if y == x or y in A_set:
                        continue

                    # Simulate swap
                    A_set.remove(x)
                    A_set.add(y)
                    counts_new = compute_counts_direct(A_set)
                    score_new = score_tuple(counts_new)

                    if score_new < curr_score:
                        curr_score = score_new
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_A = set(A_set)
                        break  # Accept first improvement, restart scan
                    else:
                        A_set.remove(y)
                        A_set.add(x)

            if not improved:
                break

        return best_A, best_score

    # Phase 3: Multi-start with different seeds
    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))

    # Start 1: Greedy minimax from structured seed
    A1, s1 = greedy_minimax_build(seed_size=12, max_size=75)
    if s1 < best_score:
        best_score = s1
        best_A = A1

    # Start 2: Greedy from arithmetic progression seed
    A2_set = set(range(0, n + 1, 2))  # All evens
    A2_bool = np.zeros(n + 1, dtype=np.float64)
    for i in A2_set:
        A2_bool[i] = 1.0
    # Greedy add from here
    counts2 = compute_counts_direct(A2_set)
    curr2 = score_tuple(counts2)
    for step in range(30):
        if time.time() > deadline - 30.0:
            break
        best_y = None
        best_s = curr2
        for y in range(n + 1):
            if y in A2_set:
                continue
            A2_set.add(y)
            c_new = compute_counts_direct(A2_set)
            s_new = score_tuple(c_new)
            A2_set.remove(y)
            if s_new < best_s:
                best_s = s_new
                best_y = y
        if best_y is not None:
            A2_set.add(best_y)
            curr2 = best_s
        else:
            break
    if curr2 < best_score:
        best_score = curr2
        best_A = A2_set

    # Start 3: Random-ish structured seed (every 3rd)
    A3_set = set(range(0, n + 1, 3))
    counts3 = compute_counts_direct(A3_set)
    curr3 = score_tuple(counts3)
    for step in range(40):
        if time.time() > deadline - 30.0:
            break
        best_y = None
        best_s = curr3
        for y in range(n + 1):
            if y in A3_set:
                continue
            A3_set.add(y)
            c_new = compute_counts_direct(A3_set)
            s_new = score_tuple(c_new)
            A3_set.remove(y)
            if s_new < best_s:
                best_s = s_new
                best_y = y
        if best_y is not None:
            A3_set.add(best_y)
            curr3 = best_s
        else:
            break
    if curr3 < best_score:
        best_score = curr3
        best_A = A3_set

    # Phase 4: Polish the best
    if best_A is not None and time.time() < deadline - 10.0:
        A_polished, s_polished = hill_climb_polish(best_A, max_rounds=30)
        if s_polished < best_score:
            best_score = s_polished
            best_A = A_polished

    # Phase 5: Final quick polish
    if best_A is not None and time.time() < deadline - 3.0:
        A_final, s_final = hill_climb_polish(best_A, max_rounds=15)
        if s_final < best_score:
            best_score = s_final
            best_A = A_final

    if best_A is None:
        return list(range(0, n + 1))

    return list(best_A)
