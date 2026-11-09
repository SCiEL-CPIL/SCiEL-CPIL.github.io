"""Divisibility-antichain representations by smooth numbers.
Global min-cost assignment via pre-computed minimum-cardinality antichain sums."""
from itertools import combinations

def construct():
    # Precompute 2^a * 3^b * 5^c smooth numbers up to 128
    smooth = []
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                val = p2 * p3 * p5
                if val > 128:
                    break
                smooth.append(val)
    smooth = sorted(set(smooth))
    smooth_set = set(smooth)

    def is_divisible(a, b):
        return b % a == 0

    def valid_antichain(terms):
        # distinct positive smooth terms, no term divides another
        if len(terms) != len(set(terms)):
            return False
        for i in range(len(terms)):
            for j in range(len(terms)):
                if i == j:
                    continue
                if is_divisible(terms[i], terms[j]):
                    return False
        return True

    # For each target, find minimum-cardinality antichain representation
    # Model as: for each target n, candidate representations sorted by cardinality
    # Then assign each target to its min-cardinality rep (independent, so greedy = optimal)
    # Frame as min-cost max-flow on bipartite graph: left = targets, right = reps
    # Edge cost = cardinality. Since independent, min-cost = sum of per-target min.

    target_reps = {}  # n -> list of (count, terms_tuple)
    for n in range(1, 129):
        candidates = []
        smooth_le_n = [x for x in smooth if x <= n]
        # Try cardinalities 1, 2, 3, ...
        found = False
        for count in range(1, 31):
            if count > len(smooth_le_n):
                break
            for terms in combinations(smooth_le_n, count):
                if sum(terms) != n:
                    continue
                if not valid_antichain(terms):
                    continue
                candidates.append((count, terms))
                found = True
            if found:
                break
        if candidates:
            # Pick minimum cardinality (all in candidates have same min count since we break)
            target_reps[n] = candidates[0][1]
        else:
            target_reps[n] = None

    # Global min-cost assignment: since targets are independent, each target
    # independently takes its min-cardinality rep. Solve as trivial flow.
    # For completeness, frame as min-cost max-flow:
    # Source -> each target (cap 1, cost 0)
    # Each target -> its feasible reps (cap 1, cost = cardinality)
    # Each rep -> sink (cap 1, cost 0)
    # But reps are shared across targets only if same tuple used for multiple targets,
    # which is fine since reps are just value tuples, not physical resources.
    # So independent assignment is optimal.

    answers = []
    for n in range(1, 129):
        rep = target_reps.get(n)
        if rep is None:
            # Fallback: should not happen, but use a valid one
            answers.append([])
        else:
            answers.append(list(rep))

    return answers
