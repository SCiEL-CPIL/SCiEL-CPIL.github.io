import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)
    
    def compute_counts_vec(f):
        c = np.convolve(f, f, mode='full')
        counts = np.zeros(num_targets, dtype=np.int64)
        for i, s in enumerate(target_sums):
            if s < len(c):
                ordered = c[s]
                if s % 2 == 0:
                    mid = s // 2
                    if 0 <= mid <= n:
                        counts[i] = (ordered + f[mid]) // 2
                    else:
                        counts[i] = ordered // 2
                else:
                    counts[i] = ordered // 2
        return counts
    
    def get_peak(counts):
        return int(np.max(counts))
    
    def is_valid(counts):
        return bool(np.all(counts >= 1))
    
    # Try multiple initial structures
    best_peak = float('inf')
    best_A = set()
    best_f = None
    
    # Structure 1: Seed
    A_set = set(range(64, 129))
    f = np.zeros(n + 1, dtype=np.int64)
    for a in A_set:
        f[a] = 1
    counts = compute_counts_vec(f)
    if is_valid(counts):
        peak = get_peak(counts)
        if peak < best_peak:
            best_peak = peak
            best_A = set(A_set)
            best_f = f.copy()
    
    # Structure 2: Evenly spaced
    for step_size in [3, 4, 5, 6]:
        A_set = set(range(0, n + 1, step_size))
        f = np.zeros(n + 1, dtype=np.int64)
        for a in A_set:
            f[a] = 1
        counts = compute_counts_vec(f)
        if is_valid(counts):
            peak = get_peak(counts)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A_set)
                best_f = f.copy()
    
    # Structure 3: Two clusters
    for lo, hi in [(0, 40), (20, 60), (40, 80), (60, 100), (80, 128)]:
        A_set = set(range(lo, hi))
        f = np.zeros(n + 1, dtype=np.int64)
        for a in A_set:
            f[a] = 1
        counts = compute_counts_vec(f)
        if is_valid(counts):
            peak = get_peak(counts)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A_set)
                best_f = f.copy()
    
    # Structure 4: Random sets of various sizes
    random.seed(42)
    for _ in range(50):
        size = random.randint(30, 55)
        A_set = set(random.sample(range(n + 1), size))
        f = np.zeros(n + 1, dtype=np.int64)
        for a in A_set:
            f[a] = 1
        counts = compute_counts_vec(f)
        if is_valid(counts):
            peak = get_peak(counts)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A_set)
                best_f = f.copy()
    
    print(f"Best initial peak: {best_peak}")
    
    A_set = set(best_A)
    f = best_f.copy()
    
    # Local search
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        step += 1
        move_type = random.choice(['add', 'remove', 'swap'])
        improved = False
        
        if move_type == 'add' and len(A_set) < 65:
            candidates = [x for x in range(n + 1) if x not in A_set]
            if candidates:
                x = random.choice(candidates)
                f_new = f.copy()
                f_new[x] = 1
                counts_new = compute_counts_vec(f_new)
                if is_valid(counts_new):
                    peak_new = get_peak(counts_new)
                    if peak_new < best_peak:
                        A_set.add(x)
                        f = f_new
                        best_peak = peak_new
                        improved = True
        
        elif move_type == 'remove' and len(A_set) > 25:
            x = random.choice(list(A_set))
            f_new = f.copy()
            f_new[x] = 0
            counts_new = compute_counts_vec(f_new)
            if is_valid(counts_new):
                peak_new = get_peak(counts_new)
                if peak_new < best_peak:
                    A_set.discard(x)
                    f = f_new
                    best_peak = peak_new
                    improved = True
        
        elif move_type == 'swap' and len(A_set) > 0:
            x = random.choice(list(A_set))
            candidates = [y for y in range(n + 1) if y not in A_set]
            if candidates:
                y = random.choice(candidates)
                f_new = f.copy()
                f_new[x] = 0
                f_new[y] = 1
                counts_new = compute_counts_vec(f_new)
                if is_valid(counts_new):
                    peak_new = get_peak(counts_new)
                    if peak_new < best_peak:
                        A_set.discard(x)
                        A_set.add(y)
                        f = f_new
                        best_peak = peak_new
                        improved = True
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 300 and step % 50 == 0 and time.time() < deadline - 5:
            # Restart with random set
            size = random.randint(30, 55)
            A_set = set(random.sample(range(n + 1), size))
            f = np.zeros(n + 1, dtype=np.int64)
            for a in A_set:
                f[a] = 1
            counts = compute_counts_vec(f)
            if is_valid(counts):
                peak = get_peak(counts)
                if peak < best_peak:
                    best_peak = peak
                    no_improve = 0
    
    return sorted(A_set)

