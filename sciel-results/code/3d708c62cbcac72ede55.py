import numpy as np
import time

N = 100

def max_overlap(h):
    # h: array of N values in [0,1], sum = N/2
    # overlap at shift k = sum_i h_i * (1 - h_{(i+k)%N})
    # return max over k in 1..N-1 (or 0..N-1?)
    N = len(h)
    best = 0.0
    for k in range(1, N):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > best:
            best = ov
    return best

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin
    
    N = 100
    # baseline
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = max_overlap(h0)
    
    # Try random 0/1 and fractional sequences with local improvement
    rng = np.random.default_rng(0)
    
    def repair(h):
        # adjust to make sum exactly N/2 while keeping in [0,1]
        s = h.sum()
        target = N/2.0
        diff = target - s
        if abs(diff) < 1e-12:
            return h
        # distribute diff
        h = h + diff/N
        h = np.clip(h, 0.0, 1.0)
        # fix sum again
        s = h.sum()
        diff = target - s
        # adjust elements not at bounds
        idx = np.where((h > 1e-9) & (h < 1-1e-9))[0]
        if len(idx) > 0:
            h[idx] += diff/len(idx)
            h = np.clip(h, 0.0, 1.0)
        return h
    
    def local_search(h, budget):
        h = h.copy()
        best = max_overlap(h)
        step = 0.1
        t = time.time()
        it = 0
        while time.time() - t < budget and it < 2000:
            it += 1
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j: continue
            d = (rng.random()-0.5)*step
            h2 = h.copy()
            h2[i] += d
            h2[j] -= d
            h2 = np.clip(h2, 0, 1)
            s = h2.sum()
            if abs(s - N/2) > 1e-6:
                continue
            v = max_overlap(h2)
            if v < best - 1e-9:
                best = v
                h = h2
            else:
                step = max(0.01, step*0.95)
        return h, best
    
    # Try several random starts
    for restart in range(30):
        if time.time() > deadline:
            break
        # random fractional start
        h = rng.random(N)
        h = repair(h)
        h, v = local_search(h, 3.0)
        if v < best_val - 1e-9:
            best_val = v
            best_h = h
    
    return best_h
