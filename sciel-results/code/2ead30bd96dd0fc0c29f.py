"""
Strategy: Deterministic Algebraic Construction + Local Collision Repair.

Instead of random greedy search, we use a structured family of sets:
Quadratic progressions $u_i = A \cdot i + B \cdot i^2$.
These sets have the property that the sumset $U+U$ is "tight" (few collisions)
while the difference set $U-U$ is "spread out" (many unique values) because
the range is large relative to the count of elements.

The score is $1 + \frac{\log(|U-U|/|U+U|)}{\log(2 \max U + 1)}$.
To maximize this, we want:
1. $|U-U|$ to be as large as possible (ideally $N^2$).
2. $|U+U|$ to be as small as possible (ideally $\approx N^2/2$).
3. $\max U$ to be as small as possible.

Quadratic sequences $u_i = i^2$ have $|U+U| \approx N^2/2$ (since $i^2+j^2=k$ has few solutions)
and $|U-U| \approx N^2$ (since $i^2-j^2=(i-j)(i+j)$ is mostly unique for distinct pairs).
However, $u_i=i^2$ grows fast. We can scale down by using a smaller base or different polynomial.
Actually, for non-negative integers, $U=\{0, 1, 4, 9, ...\}$ works well.
Let's try $U = \{ k \cdot i^2 \}$ or $U = \{ i^2 + c \}$.

We will generate candidates from a few algebraic families and then perform a local search
to remove elements that cause sum collisions (reducing $|U+U|$) or add elements that
increase difference uniqueness without increasing max value too much.
"""

import time
import math
import random
import numpy as np

def compute_score_fast(U):
    """
    Computes the lower bound exponent using FFT for sumset and set for diffset.
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # 1. Compute Sumset |U+U| using FFT
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    
    # Count non-zero entries in C[0..2*maxU]
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    # 2. Compute Difference Set |U-U|
    # For small N, O(N^2) is fast. For N~100, 10k ops.
    diff_set = set()
    for u1 in U:
        for u2 in U:
            diff_set.add(u1 - u2)
            
    size_diff = len(diff_set)
    size_sum = sum_set_size
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def generate_quadratic_set(N, scale=1):
    """Generate a set based on quadratic progression."""
    U = set()
    for i in range(N):
        U.add(int(scale * i * i))
    U.add(0)
    return sorted(list(U))

def generate_geometric_set(N, base):
    """Generate a set based on geometric progression."""
    U = set()
    val = 1
    U.add(0)
    for i in range(N - 1):
        U.add(val)
        val *= base
    return sorted(list(U))

def local_repair(U, time_limit=5.0):
    """
    Local search to improve the set by removing elements that cause sum collisions
    or adjusting elements to increase difference spread.
    Focus: Removing an element that is part of many sum collisions reduces |U+U| significantly
    while only reducing |U-U| by a smaller factor (since diffs are symmetric and spread out).
    """
    start_time = time.time()
    current_U = sorted(set(U))
    current_score = compute_score_fast(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    while time.time() < time_limit:
        improved = False
        # Try removing each element (except 0)
        for i in range(len(current_U) - 1, 0, -1):
            if time.time() > time_limit:
                break
            candidate = current_U[:i] + current_U[i+1:]
            if not candidate:
                continue
            s = compute_score_fast(candidate)
            if s > current_score + 1e-9:
                current_U = candidate
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(current_U)
                improved = True
                break # Restart loop with new set
        
        if not improved:
            # Try adding a small random element if max is small?
            # Or try swapping a large element for a smaller one?
            # For now, stick to removal as it's safer for reducing sumset collisions.
            break
            
    return best_U, best_score

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_U = [0, 1, 4, 9, 16]
    best_score = compute_score_fast(best_U)
    
    # Strategy 1: Explore different algebraic families
    families = []
    
    # Quadratic: u_i = i^2
    for N in [5, 10, 15, 20, 30, 40, 50]:
        U = generate_quadratic_set(N, scale=1)
        families.append(U)
        
    # Scaled Quadratic: u_i = 2*i^2
    for N in [5, 10, 15, 20, 30]:
        U = generate_quadratic_set(N, scale=2)
        families.append(U)
        
    # Geometric: base 2, 3, 4
    for base in [2, 3, 4, 5]:
        for N in [5, 8, 10, 12, 15]:
            U = generate_geometric_set(N, base)
            families.append(U)
            
    # Mixed: u_i = i^2 + i
    for N in [5, 10, 15, 20, 30]:
        U = set([0])
        for i in range(1, N):
            U.add(i*i + i)
        families.append(sorted(list(U)))
        
    # Evaluate all initial candidates
    for U in families:
        if time.time() > deadline - 10:
            break
        s = compute_score_fast(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
    # Strategy 2: Local Repair on the best candidate
    # The local repair focuses on removing elements that cause sum collisions.
    # This is effective because |U+U| is in the denominator (log).
    # Reducing |U+U| increases the log ratio.
    # Removing an element reduces |U| by 1, which reduces |U-U| by ~2N, 
    # but if that element was involved in many sum collisions, |U+U| drops significantly.
    
    repaired_U, repaired_score = local_repair(best_U, time_limit=10.0)
    if repaired_score > best_score:
        best_score = repaired_score
        best_U = repaired_U
        
    # Strategy 3: If time remains, try a multi-start local search from other good candidates
    if time.time() < deadline - 15:
        # Sort families by score descending, take top 3
        scored_families = []
        for U in families:
            s = compute_score_fast(U)
            scored_families.append((s, U))
        scored_families.sort(key=lambda x: x[0], reverse=True)
        
        for s, U in scored_families[:3]:
            if time.time() > deadline - 15:
                break
            if s <= best_score:
                continue
            rep_U, rep_s = local_repair(U, time_limit=10.0)
            if rep_s > best_score:
                best_score = rep_s
                best_U = rep_U
                
    return best_U
