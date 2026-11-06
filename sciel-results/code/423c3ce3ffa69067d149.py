import numpy as np
import time
import random

N = 128
LO, HI = 128, 256

def compute_stats(A):
    # A: set of ints in [0,128]
    if not A:
        return 1000, False
        
    arr = np.zeros(N+1, dtype=np.int64)
    for a in A:
        if 0 <= a <= N:
            arr[a] = 1
            
    conv = np.convolve(arr, arr, mode='full')
    
    # Sums of interest
    sums = np.arange(LO, HI+1)
    
    # Ordered counts from convolution
    ordered_counts = conv[sums]
    
    # Diagonal correction: s = a + a => a = s/2
    # If s is even, and s/2 is in A, then one pair is (a,a)
    # Unordered count = (Ordered + Diag) / 2
    # Diag is 1 if s is even and s/2 in A
    
    diag = np.zeros(len(sums), dtype=np.int64)
    Aset = set(A)
    for i, s in enumerate(sums):
        if s % 2 == 0:
            if (s // 2) in Aset:
                diag[i] = 1
                
    unord_counts = (ordered_counts + diag) // 2
    
    peak = np.max(unord_counts)
    covered = np.all(unord_counts > 0)
    
    return int(peak), bool(covered)

def construct():
    start = time.time()
    deadline = start + 55
    
    best_A = None
    best_peak = float('inf')
    
    # Multi-start local search
    num_restarts = 1000 # We will break early based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        # Start with a random set of moderate size
        # Theoretical lower bound for size is ~ sqrt(2*Range) = sqrt(256) = 16?
        # Usually denser is needed. Let's try size 40-60.
        size = random.randint(40, 60)
        A = set(random.sample(range(N+1), size))
        
        # Ensure coverage by adding elements if necessary
        # This might be slow if we do it naively, but N is small.
        peak, covered = compute_stats(A)
        attempts = 0
        while not covered and attempts < 100:
            # Add a random element not in A
            candidates = [i for i in range(N+1) if i not in A]
            if not candidates:
                break
            a = random.choice(candidates)
            A.add(a)
            peak, covered = compute_stats(A)
            attempts += 1
            
        if not covered:
            continue
            
        # 2. Local Search (Hill Climbing)
        current_peak = peak
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to remove elements
            # Sort elements by "importance" or just iterate
            # If removing an element keeps coverage and doesn't increase peak, do it.
            # Prefer removals that decrease peak.
            best_removal = None
            best_removal_peak = current_peak
            
            A_list = list(A)
            for a in A_list:
                A_test = set(A)
                A_test.remove(a)
                if not A_test:
                    continue
                p, c = compute_stats(A_test)
                if c:
                    if p < best_removal_peak:
                        best_removal_peak = p
                        best_removal = a
                    elif p == best_removal_peak and best_removal is None:
                        # Tie-breaking: remove if it doesn't hurt peak
                        # We might want to keep the set small, so removal is good
                        best_removal = a
                        
            if best_removal is not None:
                A.remove(best_removal)
                current_peak = best_removal_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A)
                continue # Restart loop after modification
            
            # Phase 2: Try to swap elements
            # Try swapping a in A with b not in A
            # Look for a swap that decreases peak.
            A_list = list(A)
            not_in_A = [i for i in range(N+1) if i not in A]
            
            best_swap = None
            best_swap_peak = current_peak
            
            # To save time, don't try ALL swaps if set is large.
            # But N=128, so max 128*128 = 16k swaps. Each eval is fast.
            # 16k * 16k ops = 256M ops? No, eval is vectorized.
            # Conv is O(N^2 log N) or similar. 128^2 is tiny.
            
            for a in A_list:
                if time.time() > deadline:
                    break
                for b in not_in_A:
                    A_test = set(A)
                    A_test.remove(a)
                    A_test.add(b)
                    p, c = compute_stats(A_test)
                    if c and p < best_swap_peak:
                        best_swap_peak = p
                        best_swap = (a, b)
                        
            if best_swap is not None:
                a, b = best_swap
                A.remove(a)
                A.add(b)
                current_peak = best_swap_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A)
                continue # Restart loop
            
        # End of local search for this restart
        
    if best_A is None:
        # Fallback
        A = set(range(64, N+1))
        return list(A)
        
    return list(best_A)
