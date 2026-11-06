"""Rewritten program: structural change to a meet-in-the-middle + bounded DFS optimizer."""
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
    # n_div is 60 for 5040

    # Strategy: meet-in-the-middle.
    # Split divisors into two halves.
    # For each half, enumerate all subset sums and store them in a dict: sum -> (subset, count)
    # But 2^30 is too large. So we need a smarter approach.
    #
    # Alternative: Since we want to minimize the number of terms, we can do a BFS/DFS
    # by number of terms k = 1, 2, 3, ... and check if any subset of size k sums to TARGET.
    # For k=1: check if TARGET is a divisor.
    # For k=2: check if there exist two distinct divisors d1, d2 with d1+d2=TARGET.
    # For k=3: etc.
    #
    # But combinations(60, k) grows quickly. Let's estimate:
    # k=1: 60
    # k=2: 1770
    # k=3: 34220
    # k=4: 487635
    # k=5: 5461512
    # k=6: 50063860
    #
    # So for k up to 4 or 5, this is feasible. Let's try increasing k.

    # Precompute set of divisors for O(1) lookup
    div_set = set(divisors)

    # k=1
    if TARGET in div_set:
        return [TARGET]

    # k=2
    for i in range(n_div):
        d1 = divisors[i]
        d2 = TARGET - d1
        if d2 in div_set and d2 != d1:
            return [d1, d2]

    # k=3: use meet-in-the-middle for pairs
    # For each divisor d, we need two other divisors that sum to TARGET - d.
    # Precompute all pair sums for the divisors.
    # But that's O(n^2) = 3600 pairs, which is fine.
    # Actually, let's just do a triple loop with pruning.
    # Better: for each d in divisors, check if TARGET-d can be expressed as sum of two distinct divisors (not d).
    # We can precompute a set of all pair sums.
    
    # Precompute pair sums: dict from sum to list of pairs (to avoid duplicates)
    # Actually, we just need to know if a sum is achievable by two distinct divisors.
    # Let's create a set of all possible pair sums.
    pair_sums = set()
    for i in range(n_div):
        for j in range(i + 1, n_div):
            pair_sums.add(divisors[i] + divisors[j])
    
    for d in divisors:
        rem = TARGET - d
        if rem in pair_sums:
            # Need to find the actual pair
            # Find two divisors that sum to rem
            for i in range(n_div):
                d1 = divisors[i]
                if d1 == d:
                    continue
                d2 = rem - d1
                if d2 in div_set and d2 != d1 and d2 != d:
                    return [d, d1, d2]
    
    # k=4: use meet-in-the-middle
    # Split divisors into two halves of 30 each.
    # For each half, compute all subset sums of size 2 (since 2+2=4).
    # Actually, for k=4, we can split into two groups of 2.
    # Precompute all pair sums for first half and second half.
    
    half = n_div // 2
    first_half = divisors[:half]
    second_half = divisors[half:]
    
    # For k=4, we want d1+d2+d3+d4 = TARGET where all are distinct.
    # We can iterate over all pairs in first_half and all pairs in second_half.
    # But that's (30*29/2)^2 = 435^2 = 189225, which is fine.
    
    # Precompute pair sums for first_half and second_half
    first_pairs = []
    for i in range(len(first_half)):
        for j in range(i + 1, len(first_half)):
            first_pairs.append((first_half[i] + first_half[j], first_half[i], first_half[j]))
    
    second_pairs = []
    for i in range(len(second_half)):
        for j in range(i + 1, len(second_half)):
            second_pairs.append((second_half[i] + second_half[j], second_half[i], second_half[j]))
    
    # Build a dict for second_pairs: sum -> (d1, d2)
    second_pair_dict = {}
    for s, d1, d2 in second_pairs:
        if s not in second_pair_dict:
            second_pair_dict[s] = (d1, d2)
    
    for s, d1, d2 in first_pairs:
        rem = TARGET - s
        if rem in second_pair_dict:
            d3, d4 = second_pair_dict[rem]
            # Check all four are distinct
            if len({d1, d2, d3, d4}) == 4:
                return [d1, d2, d3, d4]
    
    # k=5: This is harder. Let's try a DFS with pruning.
    # Or use meet-in-the-middle with sizes 2 and 3.
    # Precompute all triple sums for first_half (30 choose 3 = 4060)
    # And all pair sums for second_half.
    
    if time.time() > deadline - 5:
        # Fallback: greedy
        remainder = TARGET
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        # Return whatever we have (may not be valid, but better than nothing)
        return chosen
    
    # k=5: meet-in-the-middle with 2+3
    first_triples = []
    for i in range(len(first_half)):
        for j in range(i + 1, len(first_half)):
            for k in range(j + 1, len(first_half)):
                s = first_half[i] + first_half[j] + first_half[k]
                first_triples.append((s, first_half[i], first_half[j], first_half[k]))
    
    # second_pair_dict already exists
    for s, d1, d2, d3 in first_triples:
        rem = TARGET - s
        if rem in second_pair_dict:
            d4, d5 = second_pair_dict[rem]
            if len({d1, d2, d3, d4, d5}) == 5:
                return [d1, d2, d3, d4, d5]
    
    # k=6: meet-in-the-middle with 3+3
    second_triples = []
    for i in range(len(second_half)):
        for j in range(i + 1, len(second_half)):
            for k in range(j + 1, len(second_half)):
                s = second_half[i] + second_half[j] + second_half[k]
                second_triples.append((s, second_half[i], second_half[j], second_half[k]))
    
    second_triple_dict = {}
    for s, d1, d2, d3 in second_triples:
        if s not in second_triple_dict:
            second_triple_dict[s] = (d1, d2, d3)
    
    for s, d1, d2, d3 in first_triples:
        rem = TARGET - s
        if rem in second_triple_dict:
            d4, d5, d6 = second_triple_dict[rem]
            if len({d1, d2, d3, d4, d5, d6}) == 6:
                return [d1, d2, d3, d4, d5, d6]
    
    # If we get here, no solution found for k<=6. Try greedy as fallback.
    remainder = TARGET
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    if remainder == 0:
        return chosen
    
    # Last resort: return empty list (should not happen)
    return []
# EVOLVE-BLOCK-END
