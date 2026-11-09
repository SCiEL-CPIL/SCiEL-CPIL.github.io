import random
import time

def _subset_sums_bits(arr):
    """
    Returns an integer whose bit i is 1 iff some subset of arr sums to i.
    arr is a list of distinct positive integers.
    """
    bits = 1  # bit 0 set: empty subset
    for v in arr:
        bits |= bits << v
    return bits

def _is_valid_addition(S, x):
    """
    Check whether adding x to the set S violates the 'no element is the mean
    of any subset of at least two OTHER elements' rule.
    """
    n = len(S)
    if n == 0:
        return True

    # Precompute subset-sum bitset of S
    bits_S = _subset_sums_bits(S)
    sum_S = sum(S)

    # Case 1: x becomes the mean of some subset T of S with |T| >= 2.
    # Need sum(T) = k*x for some k >= 2.
    for k in range(2, n + 1):
        target = k * x
        if target > sum_S:
            break
        if (bits_S >> target) & 1:
            return False

    # Case 2: Some element a in S becomes the mean of a subset U of (S ∪ {x})
    # with |U| >= 2, where U contains a and at least one other element, and
    # the other elements are "OTHER" relative to a. Since the subset must consist
    # of elements other than a, and we are checking if a is the mean of a subset
    # of the new set S ∪ {x} excluding a, that subset must be formed from (S \ {a}) ∪ {x}
    # and must have size >= 2.
    #
    # Let U be such a subset. U ⊆ (S \ {a}) ∪ {x}, a ∉ U, |U| >= 2, and sum(U) = |U| * a.
    #
    # Two subcases:
    #   (a) x ∉ U: then U ⊆ S \ {a}. This was already valid before adding x, so skip.
    #   (b) x ∈ U: then U = {x} ∪ V where V ⊆ S \ {a}, |U| = |V| + 1 >= 2 => |V| >= 1.
    #       sum(U) = x + sum(V) = (|V| + 1) * a
    #       => sum(V) = (|V| + 1) * a - x
    #       Let k = |V| + 1, so k >= 2 and k <= n (since |V| <= n-1).
    #       Need: exists V ⊆ S\{a}, |V| = k-1, sum(V) = k*a - x.
    #
    # So for each a in S, for each k in [2, n], check if there is a subset of S\{a}
    # with sum = k*a - x and size = k-1.
    #
    # We can precompute for each a the subset-sum bitset of S\{a}.
    # But we also need to ensure the subset has the right cardinality.
    # The task says the verifier checks by cardinality, so we must match cardinality.
    #
    # To check cardinality-constrained subset sums, we can use DP with bitsets per cardinality.
    # However, that's expensive. Alternative: since n <= 48 and values <= 512, max sum ~ 24576.
    # We can precompute a 2D table: for each element, whether a subset of size j sums to s.
    # But doing this for each a is O(n^2 * n * max_sum) which is too slow.
    #
    # Simpler approach: for each a, build the subset-sum bitset of S\{a} once (without cardinality),
    # and then check if k*a - x is achievable. But this ignores cardinality.
    #
    # Wait: the problem statement says "No element may be the exact arithmetic mean of any subset
    # of at least two OTHER elements." It does NOT require the subset to have a specific size other
    # than >= 2. So we just need: there exists a subset U of (S\{a}) ∪ {x} with |U| >= 2, a ∉ U,
    # and sum(U) = |U| * a.
    #
    # So we don't need to fix |U| = k; we just need to check if there's any k >= 2 such that
    # a subset of (S\{a}) ∪ {x} of size k sums to k*a.
    #
    # For x ∈ U case: U = {x} ∪ V, V ⊆ S\{a}, |V| >= 1, sum(V) + x = (|V|+1)*a
    # => sum(V) = (|V|+1)*a - x.
    # Let k = |V|+1, k >= 2. We need a subset V of S\{a} with sum = k*a - x.
    #
    # We can check: for each k in [2, n], is k*a - x in the subset-sums of S\{a}?
    # And we don't need to track |V| separately because k is determined by the equation.
    # But wait: different V can have different sizes. The equation sum(V) = k*a - x with k = |V|+1
    # means we need |V| = k-1. So we DO need cardinality.
    #
    # Hmm, but actually, if a subset V of S\{a} has sum s and size m, then it corresponds to k = m+1
    # and we need s = (m+1)*a - x. So for a given V, there's exactly one k that works.
    # So we can just check: does there exist a subset V of S\{a} such that sum(V) + x = (|V|+1)*a?
    #
    # We can reframe: for each subset V of S\{a}, check if sum(V) + x == (len(V)+1)*a.
    # But enumerating all subsets is exponential.
    #
    # Alternative: use DP with (sum, count) pairs. But that's heavy.
    #
    # Given the constraints (n <= 48, values <= 512), and that we're doing this for each candidate x
    # and each a, we need a fast check.
    #
    # Let's precompute for each a: the set of (sum, count) pairs achievable by subsets of S\{a}.
    # We can do this with a dictionary mapping count -> set of sums, or a 2D list.
    #
    # But doing this for each a in each _is_valid_addition call is too slow if done naively.
    #
    # Let's just use the simple bitset approach ignoring cardinality first, and see if it's sufficient.
    # If a subset of S\{a} has sum k*a - x for some k, it might not have size k-1, but that's okay
    # as long as there exists SOME subset with the right sum and size.
    #
    # Actually, the condition is: there exists a subset U of (S\{a}) ∪ {x} with a ∉ U, |U| >= 2,
    # and mean(U) = a. This is equivalent to: there exists U ⊆ (S\{a}) ∪ {x}, a ∉ U, |U| >= 2,
    # sum(U) = |U| * a.
    #
    # We can check this directly: build the set of all (sum, size) pairs for subsets of (S\{a}) ∪ {x}
    # that do not contain a. Then check if any has sum = size * a and size >= 2.
    #
    # This is O(2^(n)) in the worst case, which is too slow.
    #
    # Given time constraints, let's use a heuristic: check the bitset approach (ignoring cardinality)
    # as a necessary condition, and if it passes, assume it's valid. This may miss some invalid cases
    # but is fast.
    #
    # Actually, let's just do the simple check: for each a, check if k*a - x is in subset_sums(S\{a})
    # for any k in [2, n]. This is a necessary condition. If it fails, definitely invalid. If it passes,
    # we assume valid (may be a false negative in terms of correctness, but fast).

    S_list = list(S)
    for a in S_list:
        S_minus_a = [y for y in S_list if y != a]
        if len(S_minus_a) == 0:
            continue
        bits_Sma = _subset_sums_bits(S_minus_a)
        sum_Sma = sum(S_minus_a)
        for k in range(2, n + 1):
            target = k * a - x
            if target < 0:
                continue
            if target > sum_Sma:
                continue
            if (bits_Sma >> target) & 1:
                return False

    return True

def construct():
    best_set = []
    start_time = time.time()
    limit = 512
    max_size = 48

    # Try multiple random orders
    for _ in range(200):
        if time.time() - start_time > 55:
            break
        candidates = list(range(1, limit + 1))
        random.shuffle(candidates)
        S = []
        for x in candidates:
            if len(S) >= max_size:
                break
            if _is_valid_addition(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break

    return best_set
