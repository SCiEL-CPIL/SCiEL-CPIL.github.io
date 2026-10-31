import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Entropy-based balance score - measures how uniformly the symbols
       are distributed. Higher entropy = more balanced = safer against affine lines.
    2. Secondary: Mod-3 coordinate-sum penalty - vectors with sum ≡ 0 mod 3 are
       more likely to complete affine lines with complementary pairs.
    3. Tertiary: Quadratic distinct symbol score - smooth reward for using all 3 symbols.
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Entropy-based balance score
    # Shannon entropy of the symbol distribution: H = -sum(p_i * log(p_i))
    # Maximum entropy (log(3)) when all symbols equally present
    # Low entropy when distribution is skewed
    counts = [c0, c1, c2]
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Normalize entropy to [0, 1] range where 1 = perfectly balanced
    max_entropy = math.log(3)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Higher entropy = better (more balanced) = higher priority
    entropy_score = 20000.0 * entropy_norm

    # Factor 2: Smooth geometric conflict proxy
    # In F_3^n, a coordinate value v is "dangerous" if it can easily complete lines.
    # Value 1 is most dangerous (1+1+1≡0), values 0 and 2 need complementary pairs.
    # Per-coordinate danger: d(v) = 1 if v==1, 0.5 if v in {0,2}
    # Total danger = sum of per-coordinate dangers, normalized.
    # Lower danger = higher priority.
    # Use smooth quadratic: danger = (c1 - 0.5*c0 - 0.5*c2) as imbalance toward 1
    # Plus a global sum-mod-3 smooth penalty via cos(2*pi*sum/3)
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    # Smooth penalty: max at sum≡0, min at sum≡1 or 2
    # cos(0)=1, cos(2pi/3)=-0.5, cos(4pi/3)=-0.5
    smooth_sum_penalty = math.cos(2.0 * math.pi * coord_sum / 3.0)
    # smooth_sum_penalty: 1.0 for sum≡0, -0.5 for sum≡1 or 2
    # Map to score: lower penalty = higher priority
    sum_score = -8000.0 * smooth_sum_penalty  # range [-4000, 8000]
    # Additional: per-coordinate 1-bias penalty (smooth)
    one_bias = (c1 - 0.5 * (c0 + c2)) / n  # positive if skewed toward 1
    one_bias_penalty = -3000.0 * one_bias * one_bias  # quadratic, always <= 0
    sum_score += one_bias_penalty

    # Factor 3: Quadratic distinct symbol score (smooth landscape)
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 4: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = entropy_score + sum_score + distinct_score + hash_score
    return float(total)
