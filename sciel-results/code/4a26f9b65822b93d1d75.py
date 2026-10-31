import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    def autoconv_max_abs(f):
        """Compute max |C(k)| for k != 0 using FFT."""
        F = np.fft.fft(f, 2 * N)
        C = np.fft.ifft(F * np.conj(F)).real[:N]
        return np.max(np.abs(C[1:]))
    
    def score_of(f):
        S = np.sum(f)
        if S == 0:
            return float('inf')
        return autoconv_max_abs(f) / (S * S)
    
    # Baseline: ones(N)
    f_ones = np.ones(N)
    best_f = f_ones.copy()
    best_score = score_of(f_ones)  # = (N-1)/N^2
    
    # Strategy 1: Single-transition square wave
    # f[i] = +1 for i < a, -1 for i >= a
    # Parameter: a in 1..N-1
    # Also try the "centered" version: shift so the transition is at various positions
    # Actually for a single contiguous block of +1s followed by -1s, 
    # the position of the block matters due to boundary effects.
    
    # More general: two contiguous blocks
    # f[i] = +1 for i in [p, p+a), -1 otherwise
    # Parameters: p (start position), a (width of +1 block)
    # But this is O(N^2) which is 160000 evaluations - might be slow with FFT each time
    
    # Better: precompute or use analytical formula
    
    # For a single contiguous +1 block of width a starting at position p:
    # f[i] = +1 if p <= i < p+a, else -1
    # sum = a - (N - a) = 2a - N
    # C(k) = sum_i f[i]f[i+k]
    
    # Let's think: f[i] = 2*indicator(p<=i<p+a) - 1
    # f[i]f[i+k] = 4*I(p<=i<p+a)*I(p<=i+k<p+a) - 2*I(p<=i<p+a) - 2*I(p<=i+k<p+a) + 1
    # Sum over valid i:
    # C(k) = 4 * overlap(k) - 2 * a - 2 * a + (N - k)  [for the +1 term, sum of 1s over N-k terms]
    # Wait, that's not right. Let me redo.
    
    # C(k) = sum_{i=0}^{N-1-k} f[i]*f[i+k]
    # f[i] = 2*I(p<=i<p+a) - 1
    # f[i+k] = 2*I(p<=i+k<p+a) - 1
    # product = 4*I(p<=i<p+a)*I(p<=i+k<p+a) - 2*I(p<=i<p+a) - 2*I(p<=i+k<p+a) + 1
    
    # Sum over i=0 to N-1-k:
    # Term 1: 4 * (number of i in [0,N-1-k] with p<=i<p+a AND p<=i+k<p+a)
    #         = 4 * (number of i with max(p, p-k) <= i < min(p+a, p+a-k) AND 0<=i<=N-1-k)
    #         For k >= 0: i in [p, p+a-k) intersect [0, N-1-k]
    #         = 4 * max(0, min(p+a-k, N-k) - max(p, 0))  [if the interval is valid]
    
    # This is getting complex. Let me just do a smart numerical search.
    
    # Strategy: Search over structured families with efficient scoring
    # Family 1: Single block of +1s, rest -1s
    #   Parameters: start p, width a
    #   But many (p,a) give same C(k) up to shift... actually no, boundaries matter.
    
    # Key insight: For a "block" signal, the autoconvolution depends on the block position
    # relative to boundaries. The worst case is when the block is at an edge.
    # The best case for minimizing peak might be when the block is centered.
    
    # Let me try: centered blocks
    # f[i] = +1 for i in [floor((N-a)/2), floor((N-a)/2)+a), -1 otherwise
    
    # Actually, let me try a different structural family:
    # Alternating blocks: +1 block of width a, then -1 block of width b, repeating
    # This is a periodic square wave.
    
    # For periodic signals, the autoconvolution is also periodic and can be computed analytically.
    
    # Let me try: f[i] = sign(sin(2*pi*m*i/N)) for various m
    # This gives m cycles of a square wave.
    
    # For m=1: one full cycle: +1 for N/2 samples, -1 for N/2 samples
    # sum = 0, so score = inf. Bad.
    
    # For m where N/m is odd: 
    #   period = N/m, half-period = N/(2m)
    #   +1 for floor(N/(2m)) samples, -1 for ceil(N/(2m)) samples per period
    #   sum = m * (floor(N/(2m)) - ceil(N/(2m))) = -m * (N/m % 2)
    
    # Hmm, this is getting complicated. Let me just do a focused numerical search.
    
    # Strategy: Try several structured families and pick the best.
    
    # Family A: ones(N) - baseline
    # Family B: Single contiguous +1 block, rest -1, centered
    # Family C: Two contiguous +1 blocks (like a "dipole" pattern)
    # Family D: Random sign flips from ones, optimized with local search
    # Family E: Analytical optimization for 2-block structure
    
    # Let me focus on Family B: centered single block
    # f[i] = +1 if |i - N/2| < a/2, else -1  (approximately)
    
    # Actually, the simplest structured family to search:
    # f = +1 on a contiguous interval of length a, -1 elsewhere
    # The autoconvolution peak will be maximized when the block is near a boundary
    # or when the block is centered (symmetric).
    
    # For a centered block: the autoconvolution is symmetric and the peak is at k=0 (excluded)
    # or at the edges of the block.
    
    # Let me just numerically evaluate a grid of (p, a) for single-block signals.
    # To make it fast, I'll precompute using the analytical formula.
    
    # For f[i] = +1 if p <= i < p+a, else -1:
    # C(k) = sum_{i=0}^{N-1-k} f[i]f[i+k]
    
    # Let me define g[i] = f[i] + 1 = 2*I(p<=i<p+a)
    # Then f[i] = g[i] - 1
    # C_f(k) = sum (g[i]-1)(g[i+k]-1) = sum g[i]g[i+k] - sum g[i] - sum g[i+k] + (N-k)
    #        = C_g(k) - 2a + (N-k)  [since sum g[i] = 2a for both shifts, for valid range]
    # Wait, sum_{i=0}^{N-1-k} g[i] = 2 * (number of i in [0,N-1-k] with p<=i<p+a)
    # This depends on k! So it's not simply 2a.
    
    # OK let me just use FFT but be smart about it.
    # N=400, so FFT of size 800 is very fast.
    # Grid search over a in [1, N-1] and p in [0, N-a]:
    # That's sum_{a=1}^{N-1} (N-a+1) ≈ N^2/2 = 80000 evaluations
    # Each FFT of size 800: ~800*log(800) ≈ 8000 ops, so 80000 * 8000 = 640M ops
    # That might be too slow in Python. Let me reduce the search.
    
    # Better: Only search centered blocks (p = floor((N-a)/2))
    # That's N-1 evaluations, very fast.
    
    # Even better: Analytical formula for centered block autoconvolution.
    
    # For a centered block of width a (a < N):
    # p = (N-a)//2, block is [p, p+a)
    # By symmetry, C(k) = C(-k)
    
    # Let me just compute it numerically for centered blocks.
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        p = (N - a) // 2
        f = -np.ones(N)
        f[p:p+a] = 1.0
        S = 2*a - N
        if S == 0:
            continue
        sc = score_of(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
    
    # Family C: Two +1 blocks (dipole-like)
    # f = +1 on [p1, p1+a1) and [p2, p2+a2), -1 elsewhere
    # This is more complex. Let me try a simpler variant:
    # f = +1 on [0, a) and [N-b, N), -1 in the middle
    # i.e., +1 blocks at both edges
    
    for a in range(1, N//2 + 1):
        if time.time() > deadline:
            break
        for b in range(1, N//2 + 1):
            if time.time() > deadline:
                break
            if a + b >= N:
                continue
            f = -np.ones(N)
            f[:a] = 1.0
            f[N-b:] = 1.0
            S = 2*(a+b) - N
            if S == 0:
                continue
            sc = score_of(f)
            if sc < best_score:
                best_score = sc
                best_f = f.copy()
    
    # Family D: Local search from best found so far
    # Flip individual signs to improve
    
    f = best_f.copy()
    current_score = best_score
    start_local = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try flipping a random position
        idx = np.random.randint(N)
        f[idx] *= -1
        new_score = score_of(f)
        if new_score < current_score:
            current_score = new_score
            best_f = f.copy()
            best_score = current_score
            no_improve_count = 0
        else:
            f[idx] *= -1  # revert
            no_improve_count += 1
    
    # Family E: Try specific known-good patterns
    # Pattern: +1, -1, +1, -1, ... (alternating)
    # sum = 0 if N even, so skip
    # Pattern: blocks of size 2: ++--++--...
    # This has period 4, sum over period = 0, so total sum = 0. Skip.
    
    # Pattern: +1 for first 2/3, -1 for last 1/3
    # This is already covered by the single-block search.
    
    # Let me try: f[i] = +1 if i mod 3 < 2, else -1 (period 3)
    # sum per period = 2-1 = 1, total sum = N/3 ≈ 133
    f_period3 = np.zeros(N)
    for i in range(N):
        f_period3[i] = 1.0 if (i % 3) < 2 else -1.0
    sc = score_of(f_period3)
    if sc < best_score:
        best_score = sc
        best_f = f_period3.copy()
    
    # Period 5: +1 for 3, -1 for 2
    f_period5 = np.zeros(N)
    for i in range(N):
        f_period5[i] = 1.0 if (i % 5) < 3 else -1.0
    sc = score_of(f_period5)
    if sc < best_score:
        best_score = sc
        best_f = f_period5.copy()
    
    # More local search from the best
    f = best_f.copy()
    current_score = best_score
    start_local2 = time.time()
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        idx = np.random.randint(N)
        f[idx] *= -1
        new_score = score_of(f)
        if new_score < current_score:
            current_score = new_score
            best_f = f.copy()
            best_score = current_score
            no_improve_count = 0
        else:
            f[idx] *= -1
            no_improve_count += 1
    
    return best_f
# EVOLVE-BLOCK-END
