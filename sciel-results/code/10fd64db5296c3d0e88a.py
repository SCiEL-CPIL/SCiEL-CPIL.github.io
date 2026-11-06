import numpy as np
import time
import random

def construct():
    n = 512
    # Represent set as boolean numpy array
    # Index 0..511 corresponds to 1..512
    
    def count_aps(S):
        """
        Count 4-term APs in S.
        S is a boolean numpy array of length n.
        """
        count = 0
        # d is the common difference
        # a, a+d, a+2d, a+3d
        # a ranges from 0 to n-1-3d
        for d in range(1, (n - 1) // 3 + 1):
            # Slices
            # a: 0 .. n-1-3d
            # a+d: d .. n-1-2d
            # a+2d: 2d .. n-1-d
            # a+3d: 3d .. n-1
            # We need element-wise AND
            # Length is n - 3d
            L = n - 3 * d
            if L <= 0:
                break
            # S[0:L] & S[d:d+L] & S[2*d:2*d+L] & S[3*d:3*d+L]
            # Note: S is 0-indexed for values 1..n? 
            # Let's say S[i] is True if (i+1) is in the set.
            # AP: x, x+d, x+2d, x+3d.
            # Indices: i, i+d, i+2d, i+3d.
            # i max is n-1-3d.
            count += np.sum(S[0:L] & S[d:d+L] & S[2*d:2*d+L] & S[3*d:3*d+L])
        return count

    def add_element(S, idx):
        S[idx] = True
        return S

    def remove_element(S, idx):
        S[idx] = False
        return S

    # Initial Greedy to get a decent start
    # We can use the logic from the original but vectorized? 
    # Or just random start and repair.
    
    # Let's try a random start of size 150
    # And use SA to remove APs.
    
    start_time = time.time()
    max_time = 55.0
    
    # Initialize S
    # Try to find a good size
    # We will iterate: increase target size, try to find a valid set.
    
    best_set = np.zeros(n, dtype=bool)
    best_size = 0
    
    # Helper to run SA to minimize APs for a given S
    def repair(S, target_aps=0, time_limit=0.5):
        """
        Run Simulated Annealing to reduce AP count to target_aps.
        Returns S and ap_count.
        """
        current_ap = count_aps(S)
        if current_ap == target_aps:
            return S, current_ap
            
        T = 10.0
        T_min = 0.01
        cooling = 0.95
        t_start = time.time()
        
        while T > T_min and time.time() - t_start < time_limit:
            # Pick a random element to remove
            ones = np.where(S)[0]
            if len(ones) == 0:
                break
            i = ones[random.randint(0, len(ones)-1)]
            
            # Pick a random element to add (from zeros)
            zeros = np.where(~S)[0]
            if len(zeros) == 0:
                break
            j = zeros[random.randint(0, len(zeros)-1)]
            
            # Swap i and j
            S[i] = False
            S[j] = True
            
            new_ap = count_aps(S)
            delta = new_ap - current_ap
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_ap = new_ap
            else:
                # Revert
                S[j] = False
                S[i] = True
            
            T *= cooling
            
        return S, current_ap

    # Strategy: 
    # 1. Start with a small random set.
    # 2. Repair it to 0 APs.
    # 3. Try to add elements one by one, repairing each time.
    
    # Initialize with random 50 elements
    S = np.zeros(n, dtype=bool)
    initial_size = 50
    indices = random.sample(range(n), initial_size)
    for i in indices:
        S[i] = True
        
    # Repair initial set
    S, ap_count = repair(S, 0, 2.0)
    
    current_size = np.sum(S)
    best_size = current_size
    best_set = S.copy()
    
    # Try to grow
    while time.time() - start_time < max_time:
        # Try to add a random element
        zeros = np.where(~S)[0]
        if len(zeros) == 0:
            break
            
        j = zeros[random.randint(0, len(zeros)-1)]
        S[j] = True
        
        # Repair
        S, ap_count = repair(S, 0, 0.1) # Short repair time
        
        if ap_count == 0:
            new_size = np.sum(S)
            if new_size > best_size:
                best_size = new_size
                best_set = S.copy()
        else:
            # Revert if we couldn't repair to 0
            S[j] = False
            
    return [i+1 for i in range(n) if best_set[i]]
