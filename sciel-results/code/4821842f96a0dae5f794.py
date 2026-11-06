"""Combined program: meet-in-the-middle with known solution fallback and efficient k-search."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70.0  # safety margin before 90s kill
    N = 5040
    TARGET = 4979

    # Compute all positive divisors of 5040
    divisors = [d for d in range(1, N + 1) if N % d == 0]
    divisors.sort(reverse=True)  # larger first for pruning

    n_div = len(divisors)
    div_set = set(divisors)

    # Known solution: {2520, 1680, 560, 210, 9} sums to 4979
    known_solution = [2520, 1680, 560, 210, 9]
    if sum(known_solution) == TARGET and len(set(known_solution)) == len(known_solution):
        if all(N % d == 0 for d in known_solution):
            # This is a valid 5-term solution. Try to find fewer terms.
            best = known_solution
        else:
            best = None
    else:
        best = None

    # k=1
    if TARGET in div_set:
        return [TARGET]

    # k=2
    for i in range(n_div):
        d1 = divisors[i]
        d2 = TARGET - d1
        if d2 in div_set and d2 != d1:
            return [d1, d2]

    # k=3: use pair sums
    pair_sums = {}
    for i in range(n_div):
        for j in range(i + 1, n_div):
            s = divisors[i] + divisors[j]
            if s not in pair_sums:
                pair_sums[s] = (divisors[i], divisors[j])

    for d in divisors:
        rem = TARGET - d
        if rem in pair_sums:
            d1, d2 = pair_sums[rem]
            if d1 != d and d2 != d and d1 != d2:
                return [d, d1, d2]

    # k=4: meet-in-the-middle with 2+2
    half = n_div // 2
    first_half = divisors[:half]
    second_half = divisors[half:]

    first_pairs = []
    for i in range(len(first_half)):
        for j in range(i + 1, len(first_half)):
            first_pairs.append((first_half[i] + first_half[j], first_half[i], first_half[j]))

    second_pair_dict = {}
    for i in range(len(second_half)):
        for j in range(i + 1, len(second_half)):
            s = second_half[i] + second_half[j]
            if s not in second_pair_dict:
                second_pair_dict[s] = (second_half[i], second_half[j])

    for s, d1, d2 in first_pairs:
        rem = TARGET - s
        if rem in second_pair_dict:
            d3, d4 = second_pair_dict[rem]
            if len({d1, d2, d3, d4}) == 4:
                return [d1, d2, d3, d4]

    # k=5: meet-in-the-middle with 2+3 or 3+2
    # Try 3 from first half + 2 from second half
    first_triples = []
    for i in range(len(first_half)):
        for j in range(i + 1, len(first_half)):
            for k in range(j + 1, len(first_half)):
                s = first_half[i] + first_half[j] + first_half[k]
                first_triples.append((s, first_half[i], first_half[j], first_half[k]))

    for s, d1, d2, d3 in first_triples:
        rem = TARGET - s
        if rem in second_pair_dict:
            d4, d5 = second_pair_dict[rem]
            if len({d1, d2, d3, d4, d5}) == 5:
                return [d1, d2, d3, d4, d5]

    # Try 2 from first half + 3 from second half
    second_triple_dict = {}
    for i in range(len(second_half)):
        for j in range(i + 1, len(second_half)):
            for k in range(j + 1, len(second_half)):
                s = second_half[i] + second_half[j] + second_half[k]
                if s not in second_triple_dict:
                    second_triple_dict[s] = (second_half[i], second_half[j], second_half[k])

    for s, d1, d2 in first_pairs:
        rem = TARGET - s
        if rem in second_triple_dict:
            d3, d4, d5 = second_triple_dict[rem]
            if len({d1, d2, d3, d4, d5}) == 5:
                return [d1, d2, d3, d4, d5]

    # If we get here, no solution found for k<=5.
    # Return the known 5-term solution if it's valid
    if best is not None:
        return best

    # Fallback: greedy
    remainder = TARGET
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    if remainder == 0:
        return chosen

    # Last resort
    return []
# EVOLVE-BLOCK-END
