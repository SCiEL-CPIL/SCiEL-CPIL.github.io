import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    rng = random.Random(42)

    def compute_m(A_set):
        """Compute unordered pair-sum multiplicities for targets 128..256."""
        A = np.array(sorted(A_set), dtype=np.int32)
        if len(A) == 0:
            return np.zeros(229, dtype=np.int32)
        indicator = np.zeros(129, dtype=np.float64)
        indicator[A] = 1.0
        conv = np.convolve(indicator, indicator, mode='full')
        m = np.zeros(229, dtype=np.int32)
        for i, s in enumerate(range(128, 257)):
            c = conv[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and indicator[half] > 0:
                    c += 1
            m[i] = int(c) // 2
        return m

    def evaluate(A_set):
        """Returns (peak, valid, m_array)."""
        m = compute_m(A_set)
        if np.any(m == 0):
            return float('inf'), False, m
        return int(np.max(m)), True, m

    def greedy_reduce(A_set, end_time):
        """Remove elements one at a time if it doesn't break coverage."""
        A = set(A_set)
        # Sort by how "essential" they might be (elements that appear in few pairs)
        # Try removing in order of least contribution
        changed = True
        while changed and time.time() < end_time:
            changed = False
            # Try removing each element, pick the one that keeps coverage and reduces peak most
            best_remove = None
            best_peak = float('inf')
            current_peak, valid, _ = evaluate(A)
            if not valid:
                break
            for el in list(A):
                if time.time() > end_time:
                    break
                candidate = A - {el}
                if len(candidate) < 2:
                    continue
                peak_c, valid_c, m_c = evaluate(candidate)
                if valid_c and peak_c < best_peak:
                    best_peak = peak_c
                    best_remove = el
            if best_remove is not None and best_peak <= current_peak:
                A.discard(best_remove)
                changed = True
        return A

    def targeted_local_search(A_set, peak, end_time):
        """Local search that targets high-multiplicity sums."""
        A = set(A_set)
        best_peak = peak
        best_A = set(A)
        stall = 0

        while time.time() < end_time:
            current_peak, valid, m = evaluate(A)
            if not valid:
                break

            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(A)
                stall = 0
            else:
                stall += 1
                if stall > 20:
                    break

            # Find the worst target(s)
            worst_targets = np.where(m == current_peak)[0]
            if len(worst_targets) == 0:
                break

            # For each worst target, try to reduce its multiplicity
            improved = False
            for t_idx in rng.sample(list(worst_targets), min(len(worst_targets), 5)):
                t = 128 + t_idx
                # Find all pairs in A that sum to t
                pairs = []
                A_list = sorted(A)
                for i in range(len(A_list)):
                    for j in range(i, len(A_list)):
                        if A_list[i] + A_list[j] == t:
                            pairs.append((A_list[i], A_list[j]))
                        elif A_list[i] + A_list[j] > t:
                            break

                if not pairs:
                    continue

                # Try swapping out one element from a pair
                for a, b in pairs[:5]:
                    # Try removing a and adding something else
                    for new_el in rng.sample(range(129), min(15, 129 - len(A))):
                        if new_el in A:
                            continue
                        candidate = A - {a} | {new_el}
                        peak_c, valid_c, _ = evaluate(candidate)
                        if valid_c and peak_c < current_peak:
                            A = candidate
                            current_peak = peak_c
                            if peak_c < best_peak:
                                best_peak = peak_c
                                best_A = set(A)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                # Try random perturbation
                k = rng.randint(1, 3)
                candidate = A.copy()
                for _ in range(k):
                    if len(candidate) > 2:
                        candidate.discard(rng.choice(list(candidate)))
                available = [i for i in range(129) if i not in candidate]
                if available:
                    for _ in range(rng.randint(1, k)):
                        candidate.add(rng.choice(available))
                peak_c, valid_c, _ = evaluate(candidate)
                if valid_c and peak_c < current_peak:
                    A = candidate
                    stall = 0

        return best_A, best_peak

    # Start with a dense valid set
    A = set(range(64, 129))
    peak, valid, _ = evaluate(A)
    if not valid:
        # Fallback: use full range
        A = set(range(0, 129))
        peak, valid, _ = evaluate(A)

    # Phase 1: Greedy reduction
    A = greedy_reduce(A, time.time() + 30)
    peak, valid, _ = evaluate(A)
    if not valid:
        A = set(range(64, 129))
        peak, valid, _ = evaluate(A)

    # Phase 2: Targeted local search
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        A, peak = targeted_local_search(A, peak, time.time() + remaining)

    # Final validation
    final_peak, final_valid, _ = evaluate(A)
    if not final_valid:
        # Emergency fallback
        A = set(range(64, 129))
        final_peak, final_valid, _ = evaluate(A)

    return list(A)
