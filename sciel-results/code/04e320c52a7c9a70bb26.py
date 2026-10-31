import time
import numpy as np
from scipy.fft import fft, ifft

N = 600

def compute_objective(h):
    n_fft = 2 * N
    H = fft(h, n=n_fft)
    acf = np.real(ifft(H * np.conj(H)))[:N]
    r0 = acf[0]
    if r0 <= 0:
        return 0.0
    return np.max(acf[1:]) / r0

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_h = None
    best_val = np.inf
    
    def try_pattern(marks):
        nonlocal best_h, best_val
        h = np.zeros(N)
        h[marks] = 1.0
        val = compute_objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Pattern 1: Squares
    marks = [i*i for i in range(25)]
    try_pattern(marks)
    
    # Pattern 2: Greedy Golomb ruler, m=30
    m = 30
    marks = [0]
    for _ in range(m-1):
        best_pos = None
        best_dup = np.inf
        for pos in range(marks[-1]+1, N):
            dists = [pos - mark for mark in marks]
            existing = set()
            for i in range(len(marks)):
                for j in range(i+1, len(marks)):
                    existing.add(marks[j]-marks[i])
            dup = sum(1 for d in dists if d in existing)
            if dup < best_dup:
                best_dup = dup
                best_pos = pos
                if dup == 0:
                    break
        marks.append(best_pos)
    try_pattern(marks)
    
    # Pattern 3: Greedy Golomb ruler, m=25
    m = 25
    marks = [0]
    for _ in range(m-1):
        best_pos = None
        best_dup = np.inf
        for pos in range(marks[-1]+1, N):
            dists = [pos - mark for mark in marks]
            existing = set()
            for i in range(len(marks)):
                for j in range(i+1, len(marks)):
                    existing.add(marks[j]-marks[i])
            dup = sum(1 for d in dists if d in existing)
            if dup < best_dup:
                best_dup = dup
                best_pos = pos
                if dup == 0:
                    break
        marks.append(best_pos)
    try_pattern(marks)
    
    # Pattern 4: Uniform (fallback)
    try_pattern(list(range(N)))
    
    if best_h is None:
        return np.ones(N).tolist()
    
    return best_h.tolist()
