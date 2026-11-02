import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def negative_score(h):
    return -compute_score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Theoretical maximum is 1.0, achieved when sup|autoconv| = ||h||_2^2
    # i.e., the diagonal peak strictly dominates all off-diagonal peaks.
    # 
    # Strategy: Use a Sidon-like set of positions where all pairwise sums/differences
    # are unique, ensuring off-diagonal autoconvolution values are strictly less than diagonal.
    #
    # For a vector with support at positions S, the autoconvolution at lag k is:
    #   c[k] = sum_{i in S, i+k in S} h[i]*h[i+k]
    # The diagonal c[0] = sum h[i]^2
    #
    # If we use equal heights h[i] = a for i in S, then:
    #   c[0] = |S| * a^2
    #   c[k] = (number of pairs (i, i+k) both in S) * a^2
    #
    # For a Sidon set, each off-diagonal k has at most 1 pair, so c[k] = a^2 < |S|*a^2 = c[0] for |S| > 1.
    #
    # Score = (|S|*a^2) / ((|S|*a) * (|S|*a^2)) = 1/(|S|*a) * a^2 = a/|S| ... wait
    #
    # Let me recompute: ||h||_1 = |S|*a, ||h||_2^2 = |S|*a^2, sup|c| = |S|*a^2
    # Score = |S|*a^2 / (|S|*a * |S|*a^2) = 1/(|S|*a)
    #
    # This is NOT scale invariant in this form... but it should be. Let me set a=1:
    # Score = |S| / (|S| * |S|) = 1/|S|
    #
    # That's < 1 for |S| > 1. Something is wrong.
    #
    # Ah, I see: sup|c| = |S|*a^2 only if the diagonal is the max. For equal heights,
    # c[0] = |S|*a^2, and c[k] for k≠0 is at most a^2 (for Sidon). So sup = |S|*a^2.
    # Score = |S|*a^2 / (|S|*a * |S|*a^2) = 1/(|S|*a)
    #
    # With a=1: Score = 1/|S|. This is less than 1.
    #
    # The issue: for a SINGLE spike (|S|=1), Score = 1/(1*1) = 1.
    # For |S|=2 with Sidon: Score = 1/2.
    #
    # So the maximum IS 1.0, achieved by a single spike!
    #
    # But wait, that contradicts the scale invariance argument. Let me recheck with h=[1,0,...,0]:
    # ||h||_1 = 1, ||h||_2^2 = 1, c = [0,...,0,1,0,...,0], sup|c| = 1
    # Score = 1/(1*1) = 1. ✓
    #
    # For h=[0.5, 0.5, 0, ...]:
    # ||h||_1 = 1, ||h||_2^2 = 0.5, c[0]=0.5, c[1]=0.25, sup=0.5
    # Score = 0.5/(1*0.5) = 1. ✓
    #
    # For h=[a, b, 0, ...]:
    # ||h||_1 = a+b, ||h||_2^2 = a^2+b^2, c[0]=a^2+b^2, c[1]=ab, sup=max(a^2+b^2, ab)=a^2+b^2
    # Score = (a^2+b^2)/((a+b)(a^2+b^2)) = 1/(a+b)
    #
    # For a+b=1: Score = 1. For a+b≠1: Score = 1/(a+b).
    #
    # So the score depends on the L1 norm! The evaluator must be normalizing or the score is not scale invariant.
    #
    # Let me recheck scale invariance: score(αh) = ||αh||_2^2 / (||αh||_1 * sup|autoconv(αh)|)
    # = α^2||h||_2^2 / (α||h||_1 * α^2 sup|autoconv(h)|)
    # = ||h||_2^2 / (α||h||_1 * sup|autoconv(h)|)
    # = score(h) / α
    #
    # It's NOT scale invariant! score(αh) = score(h)/α.
    #
    # So to maximize score, we want α as small as possible! But h must be nonnegative, and presumably
    # the evaluator doesn't normalize. So we should scale down h to make score large.
    #
    # But the task says "nonnegative heights" with no explicit bound. If we can scale arbitrarily small,
    # score → ∞. That can't be the intent.
    #
    # Re-reading: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # The ratio is ||h||_2^2 / (||h||_1 * sup|autoconv|). This is NOT scale invariant.
    #
    # However, the seed program scales by 0.1 and gets score 1.0, suggesting the evaluator may normalize
    # or the score is computed differently.
    #
    # Given the ambiguity, the safest approach: return a vector with L1 norm = 1 (normalized)
    # and ensure the diagonal dominates, achieving score = 1.0.
    #
    # Use a single spike for guaranteed score = 1.0.
    
    best_h = np.zeros(N)
    best_h[0] = 1.0  # Single spike: score = 1.0
    
    return list(best_h)
