"""Meet-in-the-middle exact solver for closest strict underpacking of unit fractions."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    denominators = list(range(2, 49))  # [2, 3, ..., 48], 47 items
    n = len(denominators)
    mid = n // 2  # 23

    A = denominators[:mid]       # first 23
    B = denominators[mid:]       # last 24

    def enumerate_subsets(dens):
        """Return list of (sum_fraction, tuple_of_local_indices)."""
        result = [(Fraction(0), ())]
        for i, d in enumerate(dens):
            # Double the list by adding 1/d to each existing subset
            current_len = len(result)
            for k in range(current_len):
                s, idx = result[k]
                ns = s + Fraction(1, d)
                nidx = idx + (i,)
                result.append((ns, nidx))
        return result

    subsA = enumerate_subsets(A)
    if time.time() - start_time > DEADLINE:
        return _greedy(denominators)

    subsB = enumerate_subsets(B)
    if time.time() - start_time > DEADLINE:
        return _greedy(denominators)

    # Sort both by sum ascending
    subsA.sort(key=lambda x: x[0])
    subsB.sort(key=lambda x: x[0])

    best_sum = Fraction(0)
    best_indices = None
    one = Fraction(1)

    # Two-pointer: j starts at end of B, moves left as A sum increases
    j = len(subsB) - 1
    for sumA, idxA in subsA:
        if time.time() - start_time > DEADLINE:
            break
        while j >= 0 and sumA + subsB[j][0] >= 1:
            j -= 1
        if j < 0:
            break
        sumB, idxB = subsB[j]
        total = sumA + sumB
        if total > best_sum:
            best_sum = total
            global_idx = list(idxA) + [mid + i for i in idxB]
            best_indices = global_idx

    if not best_indices:
        return _greedy(denominators)

    return best_indices

def _greedy(denominators):
    from fractions import Fraction
    selected, total = [], Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
# EVOLVE-BLOCK-END
