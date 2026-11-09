import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 270.0  # Safety margin before 300s hard limit


def priority(vector, n):
    # Deadline guard: if we've overshot, return a neutral low score
    # so the evaluator can still finish its sort/insertion pass.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0

    # Count symbols 0, 1, 2
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1

    if n <= 0:
        return 0.0

    # ---- Probabilistic collision density model ----
    # For a vector v, estimate the probability that it forms a 3-AP
    # with two other random vectors from a "typical" cap set of density d.
    #
    # Key insight: A 3-AP (a, b, c) exists iff for each coordinate i:
    #   a_i + c_i ≡ 2*b_i (mod 3)
    # Given b = v, the number of pairs (a, c) that complete a 3-AP
    # with v as the center is 3^(k) where k = number of coordinates
    # where v_i = 1 (since a_i, c_i can be any of (0,2), (1,1), (2,0)).
    # When v_i ≠ 1, there are fewer valid (a_i, c_i) pairs.
    #
    # More precisely, for each coordinate:
    #   v_i = 0: valid (a_i, c_i) pairs are (0,0), (1,2), (2,1) -> 3 pairs
    #   v_i = 1: valid (a_i, c_i) pairs are (0,2), (1,1), (2,0) -> 3 pairs  
    #   v_i = 2: valid (a_i, c_i) pairs are (0,1), (1,0), (2,2) -> 3 pairs
    #
    # So every coordinate contributes a factor of 3, giving 3^n total
    # pairs. But not all of these are in the cap set. The collision
    # probability depends on how "structured" v is.
    #
    # Better model: The local neighborhood size around v in the 3-AP graph.
    # v can be the CENTER of a 3-AP with any pair (a, c) where
    # a + c = 2v (mod 3). The number of such pairs is 3^n (one per coordinate
    # choice), but many will be invalid or already excluded.
    #
    # We estimate collision density as:
    #   P_collision(v) ∝ (number of potential AP partners) × (local density)
    #
    # The number of potential AP partners where v is the center:
    # For each coordinate i, if v_i = x, then a_i + c_i ≡ 2x (mod 3).
    # The valid (a_i, c_i) pairs are:
    #   x=0: (0,0), (1,2), (2,1) → 3 choices
    #   x=1: (0,2), (1,1), (2,0) → 3 choices  
    #   x=2: (0,1), (1,0), (2,2) → 3 choices
    # So always 3^n total pairs.
    #
    # But the KEY difference is in how many of these pairs are LIKELY to be
    # in a typical cap set. Vectors with more 1s have their AP partners
    # more "spread out" in the cube, while vectors with extreme compositions
    # (all 0s, all 2s) have AP partners that cluster in structured subspaces.
    #
    # Refined model: Use the entropy of the coordinate distribution as a
    # proxy for how "generic" v is. High entropy = generic = lower collision.
    p0 = n0 / n
    p1 = n1 / n
    p2 = n2 / n

    # Shannon entropy of the composition (max = ln(3) when balanced)
    entropy = 0.0
    for p in [p0, p1, p2]:
        if p > 1e-12:
            entropy -= p * math.log(p)
    # Normalize: entropy in [0, ln(3)]
    max_entropy = math.log(3.0)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0  # in [0, 1]

    # Midpoint concentration: v with many 1s is a "natural" center for many APs
    # The fraction of coordinates where v is the midpoint (value 1)
    midpoint_frac = n1 / n

    # Endpoint concentration: v with many 0s or 2s tends to be an endpoint
    endpoint_frac = (n0 + n2) / n

    # Asymmetry: imbalance between 0s and 2s
    asymmetry = abs(n0 - n2) / n

    # Concentration: how close to a "pure" vector
    max_frac = max(p0, p1, p2)
    concentration = max_frac  # in [1/3, 1]

    # ---- Collision density estimate ----
    # We model the expected number of 3-APs that v participates in,
    # weighted by the local density of the cap set.
    #
    # Heuristic: The collision probability is higher when:
    # 1. v has low entropy (structured, close to a face of the cube)
    # 2. v is concentrated near a "midpoint" (many 1s) OR near an endpoint
    # 3. v is asymmetric (more 0s than 2s or vice versa)
    #
    # Use a smooth surrogate: collision_density = f(entropy, midpoint_frac, concentration)
    #
    # The "independent set" approximation: treat each potential 3-AP as an
    # edge in a graph. v is "safe" if it has few edges to other likely cap set members.
    #
    # Weighted risk:
    low_entropy_risk = (1.0 - norm_entropy) ** 2  # high when structured
    midpoint_risk = midpoint_frac ** 2  # high when many 1s
    endpoint_risk = endpoint_frac ** 2  # high when few 1s
    asym_risk = asymmetry ** 2  # high when imbalanced
    conc_risk = (concentration - 1.0/3.0) ** 2  # high when concentrated

    # Combine with weights that emphasize the most important factors.
    # Based on the insight that balanced compositions route through the
    # high-density center first, we penalize both low entropy AND extreme
    # midpoint/endpoint fractions.
    collision_density = (
        2.5 * low_entropy_risk
        + 1.8 * midpoint_risk
        + 1.2 * endpoint_risk
        + 0.9 * asym_risk
        + 1.5 * conc_risk
    )

    # ---- Deterministic tie-breaker via FNV-1a hash ----
    h = 0x811c9dc5
    for v in vector:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    hash_score = (h % 1000003) / 1000003.0  # small perturbation

    # Return: higher priority for lower collision density.
    # Scale so the hash only breaks ties.
    return -collision_density * 1000.0 + hash_score
