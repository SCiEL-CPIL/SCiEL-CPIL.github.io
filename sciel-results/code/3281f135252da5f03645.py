"""Cap set greedy builder using smooth quadratic landscape with Z3-correlation penalty."""
import time
import hashlib

_start_time = None
_DEADLINE = 300.0 * 0.95  # 285s safety margin

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Structural design:
    1. Smooth quadratic distinct-symbol score (continuous gradient, no cliffs).
    2. L2 balance penalty from uniform (n/3, n/3, n/3) distribution.
    3. Z3 pairwise correlation penalty: for each pair of coordinates, penalize
       configurations that are "aligned" in Z3 (sum ≡ 0 mod 3), which are the
       configurations most likely to complete affine lines. This is a smooth
       proxy for the vector's contribution to the set's additive energy.
    4. Symmetry breaking: favor vectors with more 2s to explore full space.
    5. Deterministic hash for tie-breaking.
    
    Higher priority = inserted earlier.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast single pass: counts, sum, sum-of-squares
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Smooth quadratic distinct-symbol score.
    # 0 distinct -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: L2 balance penalty.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Z3 pairwise correlation penalty.
    # For each unordered pair of coordinates (i,j), compute (v_i + v_j) mod 3.
    # If the sum is 0 mod 3, the pair is "aligned" and more likely to be part
    # of a line. Count how many pairs have sum ≡ 0, 1, 2 mod 3.
    # The penalty is based on the excess of 0-mod-3 pairs over the expected count.
    # Expected number of 0-mod-3 pairs for random vector: n*(n-1)/6 (since P(a+b≡0) = 1/3)
    # We compute the actual count and penalize deviation above expected.
    # For efficiency, we use the identity: sum_{i<j} (v_i + v_j) mod 3 can be
    # computed from the counts. Actually, let's use a direct but efficient method.
    
    # Count pairs by their sum mod 3 using the counts:
    # Pairs (0,0): sum=0, count = c0*(c0-1)//2
    # Pairs (0,1): sum=1, count = c0*c1
    # Pairs (0,2): sum=2, count = c0*c2
    # Pairs (1,1): sum=2, count = c1*(c1-1)//2
    # Pairs (1,2): sum=0, count = c1*c2
    # Pairs (2,2): sum=1, count = c2*(c2-1)//2
    
    pairs_sum0 = c0*(c0-1)//2 + c1*c2
    pairs_sum1 = c0*c1 + c2*(c2-1)//2
    pairs_sum2 = c0*c2 + c1*(c1-1)//2
    
    total_pairs = n*(n-1)//2
    # Expected fraction of sum-0 pairs is 1/3 for uniform random
    expected_sum0 = total_pairs / 3.0
    
    # Penalize excess of sum-0 pairs (aligned pairs that form lines)
    excess_sum0 = max(0.0, pairs_sum0 - expected_sum0)
    
    # Also penalize if the distribution is too skewed (low diversity in pair sums)
    # Use entropy-like measure: penalize if one pair-sum class dominates
    if total_pairs > 0:
        p0 = pairs_sum0 / total_pairs
        p1 = pairs_sum1 / total_pairs
        p2 = pairs_sum2 / total_pairs
        # Penalty for non-uniformity of pair-sum distribution
        skew_penalty = (p0 - 1/3)**2 + (p1 - 1/3)**2 + (p2 - 1/3)**2
    else:
        skew_penalty = 0.0
    
    # Use a smooth max (log-sum-exp style) to combine penalties
    # This creates a smoother landscape than hard max
    import math
    alpha = 5.0  # temperature parameter for soft-max
    if excess_sum0 > 0 or skew_penalty > 0:
        max_val = max(excess_sum0, skew_penalty * 500.0)
        # Soft-max approximation: smooth the transition
        corr_combined = max_val * (1.0 + 0.1 * math.exp(-alpha * max_val / max(1.0, total_pairs)))
    else:
        corr_combined = 0.0
    
    correlation_score = -corr_combined * 2.0

    # Factor 5: Variance penalty (smooth danger proxy).
    # Use a smooth max-based approach: penalize high variance more aggressively
    mean = s / n
    variance = max(0.0, s2 / n - mean * mean)
    # Smooth the danger: use a soft-threshold to avoid hard cliffs
    threshold = 0.5  # variance below this is "safe"
    if variance > threshold:
        excess_var = variance - threshold
        danger_score = -excess_var * 3000.0 - variance * 2000.0
    else:
        danger_score = -variance * 1000.0

    # Factor 6: SHA256-based hash for better uniformity in tie-breaking.
    # Convert vector to a string and hash it for a 64-bit slice
    vec_str = ''.join(str(v) for v in vector)
    h = int(hashlib.sha256(vec_str.encode()).hexdigest()[:16], 16)
    hash_score = (h & 0xFFFFFFFF) / 0xFFFFFFFF

    # Factor 7: Product-based complexity score (rewards full-rank algebraic structure)
    # Count how many distinct values appear in each "block" of coordinates
    # This rewards vectors with richer internal structure
    if n >= 3:
        # Split into 3 roughly equal blocks and check diversity in each
        block_size = n // 3
        diversity_bonus = 0.0
        for b in range(3):
            start = b * block_size
            end = start + block_size if b < 2 else n
            block_vals = set(vector[start:end])
            diversity_bonus += len(block_vals)
        # Reward higher diversity across blocks
        complexity_score = diversity_bonus * 50.0
    else:
        complexity_score = 0.0
    
    # Combined priority
    total = distinct_score + weight_score + balance_score + correlation_score + danger_score + hash_score + complexity_score
    return float(total)
