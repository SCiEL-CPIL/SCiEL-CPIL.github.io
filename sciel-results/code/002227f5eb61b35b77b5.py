from fractions import Fraction
from itertools import combinations


def _is_subsum_free(selected, candidate):
    """Check that no reciprocal of a selected element equals the sum of
    reciprocals of a nonempty subset of the OTHER selected elements.
    'candidate' is the element to add; we verify the full set
    selected + [candidate].
    """
    S = selected + [candidate]
    n = len(S)
    recip = {x: Fraction(1, x) for x in S}

    for i in range(n):
        target = recip[S[i]]
        others = [S[j] for j in range(n) if j != i]
        # check all nonempty subsets of 'others'
        m = len(others)
        for mask in range(1, 1 << m):
            total = Fraction(0, 1)
            for k in range(m):
                if (mask >> k) & 1:
                    total += recip[others[k]]
                    if total > target:
                        break
            if total == target:
                return False
    return True


def construct():
    # Greedy: consider candidates in descending order; add if the set stays
    # reciprocal-subsum-free. Larger elements first tends to keep reciprocals
    # small and distinct, helping maximise cardinality.
    selected = []
    for cand in range(24, 0, -1):
        if _is_subsum_free(selected, cand):
            selected.append(cand)
        if len(selected) == 0:
            # ensure nonempty; first candidate is always fine
            selected.append(cand)
            break
    # Fallback: guarantee a valid nonempty set
    if not selected:
        return [1]
    return list(selected)
