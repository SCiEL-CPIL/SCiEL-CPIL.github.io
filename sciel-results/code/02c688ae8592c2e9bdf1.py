# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] for sorted support.
    Uses prefix sums of weights to avoid O(n^2) inner loop.
    
    P = sum_{k,l} w_k w_l * F(2*s_l - s_k - epsilon)
    where F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i w_j
    
    For sorted support, for fixed k,l, we can compute F(2*s_l - s_k) by
    iterating over i and using binary search to find max j such that s_j < 2*s_l - s_k - s_i.
    """
    n = len(support)
    w = weights / weights.sum()
    # Precompute prefix sums of weights
    w_prefix = np.concatenate([[0.0], np.cumsum(w)])
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # For each i, find largest j such that support[j] < threshold - support[i]
            # i.e., support[j] < threshold - support[i]
            inner = 0.0
            for i in range(n):
                target = threshold - support[i]
                # Find number of j with support[j] < target
                # Since support is sorted, use searchsorted
                j_max = np.searchsorted(support, target, side='left') - 1
                if j_max >= 0:
                    inner += w[i] * w_prefix[j_max + 1]
            P += w[k] * w[l] * inner
    return P


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for discrete measure."""
    n = len(support)
    w = weights / weights.sum()
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            pair_sums = support[:, None] + support[None, :]
            mask = pair_sums < threshold
            inner = np.sum(w[:, None] * w[None, :] * mask)
            P += w[k] * w[l] * inner
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using a mixture-of-baselines structural approach.
    
    Key insight: P[X1+X2+X3 < 2*X4] can be maximized by concentrating mass
    at a few well-chosen points. We explore:
    1. Two-point distributions {0, M} with optimized weight ratio
    2. Three-point distributions {0, a, M} with grid-searched weights
    3. Four-point distributions with geometric spacing
    
    The two-point case has a clean analytical structure: if X ∈ {0, M} with
    P(X=M)=q, then X1+X2+X3 ∈ {0, M, 2M, 3M} and 2X4 ∈ {0, 2M}.
    The condition X1+X2+X3 < 2X4 holds when:
    - X4=M (prob q) and X1+X2+X3 < 2M, i.e., sum ∈ {0, M} (prob 1-(1-q)^3*... )
    
    We do a focused search over small n with fine-grained weight optimization.
    """
    deadline = time.time() + 570  # safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # ===== Two-point analysis: X ∈ {0, M}, P(X=M) = q =====
    # X1+X2+X3 takes values: 0 (prob (1-q)^3), M (prob 3q(1-q)^2), 2M (prob 3q^2(1-q)), 3M (prob q^3)
    # 2X4 takes values: 0 (prob 1-q), 2M (prob q)
    # Condition X1+X2+X3 < 2X4:
    #   If 2X4 = 0: need X1+X2+X3 < 0, impossible (all >= 0). Contribution: 0
    #   If 2X4 = 2M: need X1+X2+X3 < 2M, i.e., sum ∈ {0, M}
    #     P(sum ∈ {0, M}) = (1-q)^3 + 3q(1-q)^2 = (1-q)^2((1-q) + 3q) = (1-q)^2(1+2q)
    # So P = q * (1-q)^2 * (1+2q)
    # Maximize f(q) = q(1-q)^2(1+2q) over q ∈ [0,1]
    # f(q) = q(1-2q+q^2)(1+2q) = q(1 + 2q - 2q - 4q^2 + q^2 + 2q^3) = q(1 - 3q^2 + 2q^3) = q - 3q^3 + 2q^4
    # f'(q) = 1 - 9q^2 + 8q^3 = 0
    # 8q^3 - 9q^2 + 1 = 0
    # Try q=1: 8-9+1=0. So (q-1) is a factor.
    # 8q^3 - 9q^2 + 1 = (q-1)(8q^2 - q - 1) = (q-1)(q-1)(8q+1)... let me check
    # (q-1)(8q^2 - q - 1) = 8q^3 - q^2 - q - 8q^2 + q + 1 = 8q^3 - 9q^2 + 1. Yes!
    # 8q^2 - q - 1 = 0 => q = (1 ± sqrt(1+32))/16 = (1 ± sqrt(33))/16
    # q = (1 + sqrt(33))/16 ≈ (1+5.7446)/16 ≈ 0.4215
    # or q = (1 - sqrt(33))/16 < 0, reject.
    # So optimal q = (1 + sqrt(33))/16 ≈ 0.4215
    # f(q*) = ?
    q_opt = (1.0 + np.sqrt(33.0)) / 16.0
    P_two_point = q_opt * (1 - q_opt)**2 * (1 + 2 * q_opt)
    
    if P_two_point > best_score:
        best_score = P_two_point
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_opt, q_opt], dtype=np.float64)
    
    # Verify with direct computation
    support_2p = np.array([0.0, 1.0], dtype=np.float64)
    weights_2p = np.array([1 - q_opt, q_opt], dtype=np.float64)
    P_verify = _compute_P(support_2p, weights_2p)
    if abs(P_verify - P_two_point) > 1e-10:
        # Use the verified value
        if P_verify > best_score:
            best_score = P_verify
            best_support = support_2p.copy()
            best_weights = weights_2p.copy()
    
    # ===== Three-point: X ∈ {0, a, M} =====
    # Try a few values of a/M ratio and optimize weights
    for ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
        if time.time() > deadline:
            break
        M = 1.0
        a = ratio * M
        support_3p = np.array([0.0, a, M], dtype=np.float64)
        # Grid search over p1, p2 (p0 = 1 - p1 - p2)
        for p1 in np.linspace(0.0, 0.9, 20):
            if time.time() > deadline:
                break
            for p2 in np.linspace(0.0, 0.9, 20):
                p0 = 1.0 - p1 - p2
                if p0 < 1e-6 or p1 < 1e-6 or p2 < 1e-6:
                    continue
                weights_3p = np.array([p0, p1, p2], dtype=np.float64)
                try:
                    P = _compute_P(support_3p, weights_3p)
                    if P > best_score:
                        best_score = P
                        best_support = support_3p.copy()
                        best_weights = weights_3p.copy()
                except Exception:
                    continue
    
    # ===== Four-point: geometric spacing =====
    for n in [4, 5, 6]:
        if time.time() > deadline:
            break
        for r in [2.0, 3.0, 5.0, 10.0]:
            if time.time() > deadline:
                break
            support_4p = np.array([r ** i for i in range(n)], dtype=np.float64)
            # Try geometric weight decay
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                if time.time() > deadline:
                    break
                weights_4p = np.array([alpha ** i for i in range(n)], dtype=np.float64)
                try:
                    P = _compute_P(support_4p, weights_4p)
                    if P > best_score:
                        best_score = P
                        best_support = support_4p.copy()
                        best_weights = weights_4p.copy()
                except Exception:
                    continue
    
    # ===== Try concentrated mass at large value with small mass at 0 =====
    # This is essentially the two-point case but let's also try asymmetric three-point
    for q_large in [0.3, 0.4, 0.42, 0.45, 0.5, 0.6]:
        if time.time() > deadline:
            break
        for q_mid in [0.0, 0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            q_small = 1.0 - q_large - q_mid
            if q_small < 1e-6:
                continue
            for M in [1.0, 10.0, 100.0]:
                if time.time() > deadline:
                    break
                support = np.array([0.0, 0.5 * M, M], dtype=np.float64)
                weights = np.array([q_small, q_mid, q_large], dtype=np.float64)
                try:
                    P = _compute_P(support, weights)
                    if P > best_score:
                        best_score = P
                        best_support = support.copy()
                        best_weights = weights.copy()
                except Exception:
                    continue
    
    # Final validation
    if best_support is None or best_weights is None:
        # Fallback to two-point optimal
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_opt, q_opt], dtype=np.float64)
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Clamp tiny negatives
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 1e-9
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_opt, q_opt], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
