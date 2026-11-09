# EVOLVE-BLOCK-START
import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Approach: Use a two-point (or few-point) distribution with a heavy-tailed component.
    The key insight is that if X4 is very large with some probability, then 2*X4 will
    be much larger than X1+X2+X3 (which are bounded by the support). So we want a
    distribution where there's a high-probability small value and a low-probability
    very large value.

    Let's try a two-point distribution: {a, b} with probabilities {p, 1-p} where a << b.
    
    For the inequality X1+X2+X3 < 2*X4:
    - If X4 = b (prob 1-p), then 2*b is very large, so X1+X2+3 < 2*b is almost always true
      (as long as 3*b_max < 2*b, which is true if b is large enough relative to a)
    - If X4 = a (prob p), then 2*a is small, so we need X1+X2+X3 < 2*a. Since X1,X2,X3 
      are each either a or b, and b >> a, the only way this works is if all three are a.
      That happens with probability p^3.

    So P ≈ (1-p) * 1 + p * p^3 = 1-p + p^4

    Wait, that's not quite right. Let me think more carefully.

    Actually, when X4 = b, 2*X4 = 2b. The sum X1+X2+X3 can be at most 3b. So if 2b > 3b, 
    that's never true. But 2b < 3b for b > 0. So we need 2b > X1+X2+X3.
    
    X1+X2+X3 ranges from 3a to 3b. So 2b > X1+X2+X3 iff X1+X2+X3 < 2b.
    The maximum possible value of X1+X2+X3 is 3b (when all are b). 2b < 3b, so it's not
    always true when X4=b.
    
    Let's compute exactly for two points {0, B} with probabilities {p, 1-p}:
    
    X4 = 0 (prob p): 2*X4 = 0. Need X1+X2+X3 < 0. Impossible since X_i >= 0. So prob 0.
    X4 = B (prob 1-p): 2*X4 = 2B. Need X1+X2+X3 < 2B.
    X1+X2+X3 can be 0, B, 2B, or 3B.
    - 0 < 2B: always true (prob p^3)
    - B < 2B: true (prob 3*p^2*(1-p))
    - 2B < 2B: false (strict inequality)
    - 3B < 2B: false
    
    So P(X1+X2+X3 < 2B | X4=B) = p^3 + 3*p^2*(1-p) = p^2*(p + 3 - 3p) = p^2*(3 - 2p)
    
    Total P = (1-p) * p^2 * (3-2p) + p * 0 = (1-p)*p^2*(3-2p)
    
    Let f(p) = (1-p)*p^2*(3-2p) = p^2*(3-2p)*(1-p) = p^2*(3 - 3p - 2p + 2p^2) = p^2*(3 - 5p + 2p^2)
    f(p) = 3p^2 - 5p^3 + 2p^4
    
    Maximize f(p) for p in [0,1]:
    f'(p) = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    8p^2 - 15p + 6 = 0
    p = (15 ± sqrt(225 - 192))/16 = (15 ± sqrt(33))/16
    sqrt(33) ≈ 5.744
    p = (15 - 5.744)/16 ≈ 9.256/16 ≈ 0.5785 or p = (15+5.744)/16 ≈ 1.3 (out of range)
    
    So p* ≈ 0.5785
    f(p*) = 3*(0.5785)^2 - 5*(0.5785)^3 + 2*(0.5785)^4
    ≈ 3*0.3347 - 5*0.1936 + 2*0.1120
    ≈ 1.0041 - 0.9680 + 0.2240
    ≈ 0.2601
    
    Hmm, that's about 0.26. The baseline uniform on {0,...,31} probably gives a higher value.
    
    Let me think of a better approach. What about using more points?
    
    Actually, let's try a distribution concentrated near 0 with a small tail. The key is
    to make 2*X4 large relative to X1+X2+X3 when X4 is large, but also have X4 be small
    often enough that the "bad" case doesn't dominate.
    
    Let me try: support = [0, 1, 2, ..., n-1] but with non-uniform weights.
    Or better: use a geometric-like distribution.
    
    Actually, let me just compute the baseline score first to understand what we're competing against.
    
    For uniform on {0, 1, ..., 31}, let's estimate P[X1+X2+X3 < 2*X4].
    This is hard to compute analytically, but the score is 1.0 which means it's the seed.
    
    Let me try a different strategy: use a distribution that puts most mass at 0 and a little
    at a large value, but with more than 2 points to fine-tune.
    
    Let me try support = [0, 1, 2, 3, 100] with carefully chosen weights.
    
    Actually, let me think about this differently. The optimal strategy likely involves:
    1. A point at 0 (or very small) with significant probability
    2. A spread of intermediate values
    3. Possibly a large value
    
    Let me try a simple 3-point distribution: {0, 1, M} with probabilities {p0, p1, pM}
    where p0 + p1 + pM = 1.
    
    For the inequality X1+X2+X3 < 2*X4:
    
    Case X4=0: 2*0=0, need X1+X2+X3<0, impossible. Prob 0.
    Case X4=1: 2*1=2, need X1+X2+X3<2. Possible sums: 0,1,2,3,...
      X1+X2+X3 can be 0 (all 0), 1 (two 0s and one 1), 2 (various), etc.
      Sums < 2: 0 and 1.
      Prob(sum=0) = p0^3
      Prob(sum=1) = 3*p0^2*p1
      So P(sum<2 | X4=1) = p0^3 + 3*p0^2*p1 = p0^2*(p0+3*p1)
    Case X4=M: 2*M is large. Need X1+X2+X3 < 2M.
      Max sum is 3M. So not always true.
      Sums < 2M: all combinations where not all three are M.
      Actually, sum can be 0,1,2,3,...,M, M+1, etc.
      Sums < 2M: everything except cases where sum >= 2M.
      Sum >= 2M requires at least two M's (since 2M >= 2M, and M+M+0=2M which is not < 2M).
      So sum >= 2M iff at least two of X1,X2,X3 are M.
      Prob(at least two M) = 3*pM^2*p0 + 3*pM^2*p1 + pM^3 = 3*pM^2*(p0+p1) + pM^3
      = 3*pM^2*(1-pM) + pM^3 = 3*pM^2 - 3*pM^3 + pM^3 = 3*pM^2 - 2*pM^3
      So P(sum < 2M | X4=M) = 1 - 3*pM^2 + 2*pM^3
      
    Total P = p1 * p0^2*(p0+3*p1) + pM * (1 - 3*pM^2 + 2*pM^3)
    
    With p0 + p1 + pM = 1, let's set pM small, say 0.1, and optimize p0, p1.
    p0 + p1 = 0.9
    P = p1 * p0^2*(p0+3*p1) + 0.1*(1 - 0.03 + 0.002) = p1*p0^2*(p0+3*p1) + 0.1*0.972
    = p1*p0^2*(p0+3*p1) + 0.0972
    
    With p1 = 0.9 - p0:
    P = (0.9-p0)*p0^2*(p0+3*(0.9-p0)) + 0.0972
    = (0.9-p0)*p0^2*(2.7-2*p0) + 0.0972
    
    Let me try p0=0.5, p1=0.4, pM=0.1:
    P = 0.4*0.25*(0.5+1.2) + 0.0972 = 0.4*0.25*1.7 + 0.0972 = 0.17 + 0.0972 = 0.2672
    
    Try p0=0.6, p1=0.3, pM=0.1:
    P = 0.3*0.36*(0.6+0.9) + 0.0972 = 0.3*0.36*1.5 + 0.0972 = 0.162 + 0.0972 = 0.2592
    
    Try p0=0.4, p1=0.5, pM=0.1:
    P = 0.5*0.16*(0.4+1.5) + 0.0972 = 0.5*0.16*1.9 + 0.0972 = 0.152 + 0.0972 = 0.2492
    
    So p0=0.5 gives ~0.267. Still not great.
    
    Let me try a larger M and smaller pM. pM=0.05:
    P = p1*p0^2*(p0+3*p1) + 0.05*(1-3*0.0025+2*0.000125) = p1*p0^2*(p0+3*p1) + 0.05*0.9925
    = p1*p0^2*(p0+3*p1) + 0.049625
    
    p0+p1=0.95. Try p0=0.5, p1=0.45:
    P = 0.45*0.25*(0.5+1.35) + 0.049625 = 0.45*0.25*1.85 + 0.049625 = 0.208125 + 0.049625 = 0.25775
    
    Hmm, it seems like the two/three point distributions don't beat the uniform baseline easily.
    
    Let me just go with a reasonable heuristic: use a distribution that has a spike at 0, 
    decays geometrically, and has a small tail. Let me use n=16 points with geometric weights.
    """
    n = 16
    support = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    
    # Geometric distribution: support[i] = i, weights[i] = r^i for some r < 1
    # But we need to tune r. Let's try r=0.7
    r = 0.7
    for i in range(n):
        support[i] = float(i)
        weights[i] = r ** i
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return support, weights


# EVOLVE-BLOCK-END
