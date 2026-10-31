import numpy as np
import random

N = 50

def is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def get_sums(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums.add(S[i] + S[j])
    return sums

def can_add(S, sums, idx):
    """Check if idx can be added to S given its current sums."""
    n = len(S)
    for i in range(n):
        s = S[i] + idx
        if s in sums:
            return False
        # Note: we don't add to sums here, just check
    # Also need to check if idx+idx collides?
    # The loop above checks S[i] + idx.
    # If i is such that S[i] == idx? No, idx is new.
    # The sum idx+idx is not checked in the loop if idx is not in S.
    # Wait, the Sidon property requires all pairwise sums (including i=j) to be unique.
    # The existing sums set contains S[i]+S[j] for i<=j.
    # New sums are S[i]+idx for i in range(n) AND idx+idx.
    # We must check:
    # 1. idx + idx not in sums
    # 2. S[i] + idx not in sums for any i
    # 3. S[i] + idx != S[j] + idx (trivially true if i!=j)
    # 4. S[i] + idx != idx + idx (i.e. S[i] != idx, which is true since idx is new)
    
    if (idx + idx) in sums:
        return False
        
    for i in range(n):
        if (S[i] + idx) in sums:
            return False
            
    return True

def find_max_sidon_greedy(N, num_restarts=200):
    best_set = []
    best_size = 0
    
    # Helper to run one greedy pass
    def greedy_pass(randomized):
        current_set = []
        current_sums = set()
        available = list(range(N))
        
        while available:
            valid_candidates = []
            for idx in available:
                if can_add(current_set, current_sums, idx):
                    valid_candidates.append(idx)
            
            if not valid_candidates:
                break
            
            if randomized:
                choice = random.choice(valid_candidates)
            else:
                # Least constraining value: pick idx that allows most future additions
                best_idx = -1
                max_options = -1
                for idx in valid_candidates:
                    # Count how many other available indices can be added after idx
                    # This is expensive, so we approximate or just pick smallest?
                    # For N=50, let's try picking the one with smallest value to be deterministic
                    # or just random.
                    # Let's stick to random for the restarts, and maybe smallest for the first pass?
                    # Actually, "least constraining" is better.
                    # But calculating it is O(N^3). N=50, N^3=125k. 50 steps * 125k = 6M ops. Acceptable.
                    
                    # Let's implement a lighter heuristic: pick the idx that minimizes the number of "blocked" pairs?
                    # Simpler: just pick the first one? Or smallest?
                    # Let's use random for both, but fixed seed.
                    pass
                
                choice = random.choice(valid_candidates)
            
            current_set.append(choice)
            # Update sums
            n = len(current_set) - 1
            for i in range(n):
                current_sums.add(current_set[i] + choice)
            current_sums.add(choice + choice)
            
            available.remove(choice)
            
        return current_set

    # First pass: Deterministic-ish (using random with fixed seed but maybe different logic?)
    # Let's just do random restarts with a fixed seed.
    random.seed(42)
    
    for r in range(num_restarts):
        s = greedy_pass(randomized=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
        if best_size >= 8:
            break
            
    return best_set

def construct():
    S = find_max_sidon_greedy(N, num_restarts=500)
    if not S:
        S = [0]
        
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0 / len(S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
