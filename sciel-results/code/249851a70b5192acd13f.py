import zlib
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Structural priority for cap set greedy construction.
    # Key insight: cap sets are subsets of {0,1,2}^n with no 3-term arithmetic progression.
    # A 3-AP exists if there are a, b, c such that a+c = 2b (coordinate-wise, mod 3).
    # This happens when a_i + c_i = 2*b_i for all i.
    #
    # Strategy: Prioritize vectors that are "far" from forming APs.
    # 1. Balance score: vectors with balanced counts of 0,1,2 are less likely to be
    #    part of short APs because they don't cluster around a single value.
    # 2. Spread score: vectors with more distinct values are better.
    # 3. AP-structure penalty: penalize vectors that have many coordinates equal,
    #    since if many coordinates are equal, it's easier to find a and c that
    #    satisfy a_i + c_i = 2*b_i for those coordinates.
    #
    # We compute:
    #   - c0, c1, c2: counts of 0, 1, 2
    #   - balance = c0 * c1 * c2 (maximized when balanced)
    #   - max_count = max(c0, c1, c2) (minimized when balanced)
    #   - For AP avoidance: count how many coordinates are "redundant" for APs.
    #     If b_i = 0, then a_i + c_i = 0 mod 3 means (a_i, c_i) in {(0,0),(1,2),(2,1)}
    #     If b_i = 1, then a_i + c_i = 2 mod 3 means (a_i, c_i) in {(0,2),(1,1),(2,0)}
    #     If b_i = 2, then a_i + c_i = 4 = 1 mod 3 means (a_i, c_i) in {(0,1),(1,0),(2,2)}
    #     In all cases, there are 3 choices for each coordinate, so the structure is
    #     symmetric. The key is that if many coordinates of b are the same value,
    #     the constraint a_i + c_i = 2*b_i is the same for those coordinates.
    #
    # A better proxy: vectors with more "variation" in their values are preferred.
    # We use the number of distinct adjacent value pairs as a spread measure.
    #
    # Final score:
    #   primary = balance (c0*c1*c2)
    #   secondary = -max_count (prefer balanced)
    #   tertiary = CRC32 hash for deterministic tie-breaking

    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    balance = float(c0 * c1 * c2)
    max_count = max(c0, c1, c2)

    # Penalty for having a dominant value (more likely to be in APs)
    # The penalty should be smooth: (max_count - n/3)^2
    n_float = float(n)
    deviation = max_count - n_float / 3.0
    penalty = deviation * deviation

    # Combine: balance is the primary driver, penalty is secondary
    score = balance - 0.5 * penalty

    # Deterministic tie-breaker
    b = bytes(vector)
    h = zlib.crc32(b)
    tie_breaker = (h % 1000000) * 1e-6

    return score + tie_breaker
# EVOLVE-BLOCK-END
