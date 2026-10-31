import math
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin
_LOG2E = 1.4426950408889634  # 1/ln(2)

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Family: static O(n) smooth surrogates for cap-set structure.
    Refinements over the baseline:
      - Add a cheap per-call "danger" penalty: sum over a small set of
        linear forms L_k(x) = sum_i (k*i mod 3) * x_i  (mod 3).
        For any existing pair (a,b) in the set, a third point c that
        completes an affine line must satisfy, for some direction,
        x_c = 2*x_b - x_a (mod 3). Linear forms with slope k detect
        line-completion patterns in the projection onto that 1D
        direction. Penalizing vectors that sit on "danger" residue
        classes reduces the chance of forming 3-term APs.
      - Replace the hard quadratic balance penalty with a smooth
        log-sum-exp (soft epigraph) penalty on the count deviations,
        which is a smooth surrogate for the L_infinity (max-deviation)
        objective and avoids flat plateaus.
      - Keep the distinct-symbol quadratic score (strong gradient toward
        using all three symbols) and a small symmetry-breaker for 2s.
      - Deterministic rolling hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to avoid hanging the evaluator.
        return 0.0

    # ---- Single pass: counts, sums, and linear-form residues ----
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0

    # We track residues for a few linear forms L_k(x) = sum_i (k*i mod 3)*x_i mod 3.
    # k in {1, 2} are the non-trivial slopes. k=0 is just the total sum mod 3.
    # For each k, we accumulate the residue mod 3.
    # We also track, for each residue class r in {0,1,2}, how many of the
    # tracked linear forms land on r. This gives a cheap "danger" count.
    #
    # To keep cost low, we track residues for k=1 and k=2 only (two non-trivial
    # directions). The total sum mod 3 (k=0) is also useful.
    #
    # L_k(x) = sum_i (k*i mod 3) * x_i  (mod 3)
    # We compute this incrementally.

    # Precompute the coefficient sequence for k=1 and k=2 up to a safe max n.
    # Since n is at most a few dozen in typical cap-set problems, we can just
    # compute on the fly.

    # For k=1: coeff_i = i mod 3
    # For k=2: coeff_i = (2*i) mod 3
    # For k=0: coeff_i = 0, so L_0 = 0 always. Skip.

    # We'll accumulate:
    #   res1 = sum_i (i mod 3) * x_i  mod 3
    #   res2 = sum_i ((2*i) mod 3) * x_i  mod 3

    res1 = 0
    res2 = 0

    # Precompute coeff sequences for efficiency (only up to n, but n is small)
    # coeff1[i] = i % 3
    # coeff2[i] = (2*i) % 3

    for i, v in enumerate(vector):
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

        # Update residues
        if v != 0:
            c1_i = i % 3
            c2_i = (2 * i) % 3
            if c1_i:
                res1 = (res1 + c1_i * v) % 3
            if c2_i:
                res2 = (res2 + c2_i * v) % 3

    # ---- Factor 1: Distinct-symbol quadratic score ----
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # ---- Factor 2: Symmetry breaking (favors 2s) ----
    weight_score = c2 * 100.0

    # ---- Factor 3: Smooth balance penalty via log-sum-exp (soft L_inf) ----
    # Penalize the maximum deviation from the ideal count n/3.
    # Use log-sum-exp on squared deviations for a smooth surrogate of max(|dev|).
    ideal = n / 3.0
    d0 = abs(c0 - ideal)
    d1 = abs(c1 - ideal)
    d2 = abs(c2 - ideal)

    # log-sum-exp of [d0^2, d1^2, d2^2] with a temperature for smoothness.
    # soft_max(x) = log(exp(x) + exp(y) + exp(z)) approximates max with smoothness.
    # We use a moderate temperature to keep it close to max but smooth.
    # Let a_i = d_i^2. soft_max = log(exp(a0) + exp(a1) + exp(a2)).
    # To avoid overflow, subtract the max first.
    a0 = d0 * d0
    a1 = d1 * d1
    a2 = d2 * d2
    amax = max(a0, a1, a2)
    # If amax is large, exp(amax) overflows. Scale by a temperature.
    # Use temperature T=1.0 for now; if amax > 50, the soft-max is essentially max.
    if amax > 50:
        soft_balance = amax  # essentially the max
    else:
        soft_balance = math.log(math.exp(a0 - amax) + math.exp(a1 - amax) + math.exp(a2 - amax)) + amax

    balance_score = -soft_balance * 5.0

    # ---- Factor 4: Variance danger proxy ----
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
    else:
        variance = 0.0
    danger_score = -variance * 5000.0

    # ---- Factor 5: Linear-form danger penalty ----
    # Penalize vectors whose linear-form residues are "dangerous."
    # For a cap set, we want to avoid points that would complete a line
    # with any existing pair. The linear-form residues give a cheap proxy:
    # if res1 == 0 or res2 == 0, the vector lies on a "low-complexity"
    # hyperplane in the projected space, which is more likely to form
    # arithmetic progressions.
    #
    # Penalty: for each tracked residue that is 0, add a penalty.
    # This is a smooth (though discrete) surrogate for spectral sparsity.
    danger_line = 0.0
    if res1 == 0:
        danger_line -= 200.0
    if res2 == 0:
        danger_line -= 200.0
    # Also penalize if both are equal (symmetric danger):
    if res1 == res2 and res1 != 0:
        danger_line -= 50.0

    # ---- Factor 6: Deterministic hash for tie-breaking ----
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # ---- Combined priority ----
    total = (distinct_score
             + weight_score
             + balance_score
             + danger_score
             + danger_line
             + hash_score)
    return float(total)
