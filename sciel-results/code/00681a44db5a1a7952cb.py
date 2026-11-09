"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    P = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, P)

    # Precompute all subset sums
    # There are 2^10 - 1 = 1023 non-empty subsets
    subset_info = []  # list of (residue, popcount, terms_list)
    for mask in range(1, 1 << N):
        terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % P
        popcount = len(terms)
        subset_info.append((residue, popcount, terms))

    # Group subsets by residue
    # For each residue, we want to find the minimum popcount subset that can represent it
    residue_to_subsets = [{} for _ in range(P)]  # residue -> {popcount: [terms, ...]}
    for residue, popcount, terms in subset_info:
        if popcount not in residue_to_subsets[residue]:
            residue_to_subsets[residue][popcount] = []
        residue_to_subsets[residue][popcount].append(terms)

    # For each residue, sort by popcount
    for r in range(P):
        if residue_to_subsets[r]:
            residue_to_subsets[r] = dict(sorted(residue_to_subsets[r].items()))

    # Check: can we cover all residues with max length k?
    # Since denominators can recur across rows, each residue just needs ONE subset of size <= k
    # So the minimum max row length is the minimum k such that every residue has at least one subset of size <= k

    # Find the minimum k
    min_k = N
    for k in range(1, N + 1):
        all_covered = True
        for r in range(P):
            if not residue_to_subsets[r] or min(residue_to_subsets[r].keys()) > k:
                all_covered = False
                break
        if all_covered:
            min_k = k
            break

    # Assign each residue the smallest valid subset (by popcount, then lexicographic)
    result = []
    for r in range(P):
        if residue_to_subsets[r]:
            min_pop = min(residue_to_subsets[r].keys())
            candidates = residue_to_subsets[r][min_pop]
            # Pick the lexicographically smallest for determinism
            candidates.sort()
            result.append(candidates[0])
        else:
            # This shouldn't happen based on our check, but fallback
            result.append([1])

    return result
# EVOLVE-BLOCK-END
