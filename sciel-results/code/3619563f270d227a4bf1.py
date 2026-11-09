"""Cap set priority: structural F_3^n cap-set awareness."""
import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0  # 50s safety margin before 330s kill

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Structural approach refined:
    - In F_3^n, a 3-term AP {a,b,c} satisfies a+c ≡ 2b (mod 3) coordinate-wise.
    - For each coordinate, the triple (a_i, b_i, c_i) must be either:
      * all equal: (0,0,0), (1,1,1), (2,2,2) — no constraint
      * a permutation of (0,1,2) — b_i is the midpoint
    - A vector with many 1s is a "midpoint-rich" point: it can serve as b
      in many potential APs, making it harder to keep the set cap.
    - Corners (all 0s or all 2s) are safest as midpoints.
    
    Strategy:
    1. PRIMARY: penalize deviation from uniform symbol distribution (n/3 each).
       This avoids extreme corners that cluster in low-dimensional subspaces.
    2. SECONDARY: penalize c1 (number of 1s) since 1s make a point a better
       midpoint candidate, increasing AP risk.
    3. TERTIARY: pairwise coordinate correlation — vectors where 0s and 2s
       are spread across different coordinates (not correlated) are safer.
    4. QUATERNARY: deterministic hash for stable tie-breaking.
    """
    if time.time() > _DEADLINE:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # PRIMARY: balance toward n/3 per symbol.
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    score = -penalty

    # SECONDARY: penalize 1s — they make the point a midpoint in more
    # coordinate-wise APs. Weight scaled to be secondary to balance.
    score -= 0.05 * c1

    # TERTIARY: measure how "spread" the 0s and 2s are.
    # For each pair of coordinates (i,j), if both are 0 or both are 2,
    # that's a correlation that increases the chance of forming APs with
    # other points. We want 0s and 2s in different coordinate patterns.
    # Approximate: count of (0,0) pairs + (2,2) pairs among coordinates.
    # Use a cheaper proxy: variance of the "0-vs-2" indicator.
    # Let s_i = 1 if v_i==0, -1 if v_i==2, 0 if v_i==1.
    # Sum of s_i^2 = c0 + c2. Sum of s_i = c0 - c2.
    # Variance-like: (c0 - c2)^2 / n measures imbalance between 0s and 2s.
    imbalance_02 = (c0 - c2)
    score -= 0.02 * imbalance_02 * imbalance_02 / max(n, 1)

    # QUATERNARY: deterministic position-weighted hash for stable tie-break.
    tie = 0.0
    for i, v in enumerate(vector):
        w = math.sin((i + 1) * 0.6180339887)
        tie += w * v
    score += tie * 1e-6

    return score
