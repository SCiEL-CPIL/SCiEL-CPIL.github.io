"""Cap set greedy priority: probabilistic local 3-AP collision density model."""
import time
import math
from itertools import product

# Hard wall-clock guard (evaluator kills at ~330s; keep safety margin)
_DEADLINE = 330.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _hamming_weight(v):
    return sum(1 for x in v if x != 0)


def _coord_entropy(v):
    # Entropy of the distribution over coordinates where v[i] != 0.
    # For a cap set heuristic, coordinates with more 'active' (non-zero) entries
    # are more likely to participate in 3-APs with other vectors.
    n = len(v)
    if n == 0:
        return 0.0
    non_zero = sum(1 for x in v if x != 0)
    if non_zero == 0 or non_zero == n:
        return 0.0
    p = non_zero / n
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _local_collision_estimate(v, n):
    """
    Estimate the local 3-AP collision probability for vector v in {0,1,2}^n.
    
    A 3-AP in F_3^n consists of a, a+d, a+2d (mod 3). For a given vector v,
    the number of potential 3-APs it can participate in depends on its
    structure. We use a heuristic based on:
    
    1. Hamming weight: vectors with weight n/2 (balanced) sit in the densest
       region of the cube and have the most potential partners.
    2. Coordinate-wise entropy: measures how "spread out" the non-zero entries
       are, which correlates with the number of directions d such that
       v+d and v+2d are also in the cube.
    
    Higher estimated collision density => lower priority (we want to avoid
    vectors that block many others).
    """
    hw = _hamming_weight(v)
    # Balanced weight (n/2) maximizes potential 3-AP participation.
    # Use a quadratic penalty: collision ~ (hw - n/2)^2 is bad, so we want
    # to AVOID high collision. But actually, for greedy insertion, we want
    # vectors that are "easy to insert" i.e., low collision with existing set.
    # Since we don't know the existing set at priority time, we estimate
    # the *intrinsic* collision propensity.
    
    # The number of directions d such that v, v+d, v+2d are all in {0,1,2}^n
    # is maximized when v has entries that are "central" (i.e., 1) because
    # 1+d and 1+2d mod 3 are always in {0,1,2}. If v[i]=0, then d[i] must be
    # 0 for v+d to be valid. If v[i]=2, then d[i] must be 0 as well (since
    # 2+d mod 3 = 0 only if d=1, but then 2+2d=2+2=1 mod 3, wait let me think).
    
    # Actually, for a fixed v, the number of d such that v, v+d, v+2d are all
    # in {0,1,2}^n: for each coordinate i:
    # - if v[i] = 0: d[i] can be 0 (v+d[i]=0, v+2d[i]=0) or 1 (v+d[i]=1, v+2d[i]=2) or 2 (v+d[i]=2, v+2d[i]=1). So 3 choices.
    # - if v[i] = 1: d[i] can be 0 (1,1,1) or 1 (2,0,1) or 2 (0,2,1). So 3 choices.
    # - if v[i] = 2: d[i] can be 0 (2,2,2) or 1 (0,1,2) or 2 (1,0,2). So 3 choices.
    # Wait, that's always 3^n? That can't be right for collision counting.
    
    # Let me reconsider. The issue is not just whether v+d is in the cube, but
    # whether v+d and v+2d are *distinct* from v and from each other, and whether
    # they collide with *other* vectors in the set. For a greedy heuristic, we
    # approximate the expected number of conflicts.
    
    # Simpler heuristic: vectors with more 1s are more "central" and thus more
    # likely to be part of many 3-APs. Vectors with all 0s or all 2s are on the
    # boundary and less likely to conflict.
    
    # Let's use: collision_score = number of coordinates where v[i] == 1
    # (since 1 is the "middle" value and most flexible for forming 3-APs)
    ones = sum(1 for x in v if x == 1)
    twos = sum(1 for x in v if x == 2)
    
    # Vectors with many 1s are more "dangerous" (higher collision propensity)
    # Vectors with many 0s or 2s are on the boundary, less dangerous
    # But we also want to balance: a vector that is all 0s is very safe but
    # doesn't help much with size. Let's use a composite score.
    
    # Heuristic: priority should be HIGH for vectors that are "safe" (low collision)
    # Safe vectors: boundary vectors (many 0s or 2s, few 1s)
    # But we also want to eventually fill the set, so maybe a moderate preference.
    
    # Let's try: score = - (number of 1s) / n  (higher = fewer 1s = safer)
    # Plus a small bonus for balanced weight to keep the set dense.
    
    balanced_bonus = 1.0 - abs(hw - n / 2.0) / (n / 2.0)  # 1 at center, 0 at edges
    safety = 1.0 - ones / n  # 1 if no 1s, 0 if all 1s
    
    # Combine: we want high safety AND reasonable balance
    return safety * 0.7 + balanced_bonus * 0.3


def priority(vector, n):
    """
    Priority for greedy cap set construction.
    Higher priority = inserted earlier.
    """
    if _elapsed() > _DEADLINE - 5.0:
        # Safety margin: return a default to avoid timeout
        return 0.0
    
    # Convert to tuple if needed
    v = tuple(vector)
    
    # Compute local collision estimate
    collision_density = _local_collision_estimate(v, n)
    
    # Higher collision density => lower priority
    # Invert so that safer vectors have higher priority
    return -collision_density


# Pre-compute nothing; the greedy loop calls priority for each vector.
# The evaluator handles the greedy insertion; we just provide priorities.
