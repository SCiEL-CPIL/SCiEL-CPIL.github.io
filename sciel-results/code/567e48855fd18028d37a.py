import numpy as np
import random

N = 50

def construct():
    # Use greedy with random restarts to find a large Sidon set (Golomb ruler)
    best_S = []
    best_score = 0
    
    for _ in range(100):
        random.shuffle(range(N))
        S = []
        for i in range(N):
            # Check if i can be added to S while maintaining Sidon property
            valid = True
            for j in S:
                # Check sums i+j and j+i (same)
                # Sidon: i+j = k+l implies {i,j}={k,l}
                # We check if i+j collides with any k+l for k,l in S
                # Since we build S incrementally, we check against existing pairs
                for k in S:
                    if k == j: continue
                    # Sums to check: i+j, i+k
                    # Existing sums in S: x+y for x,y in S
                    # This O(N^3) check is too slow for 100 restarts * 50 steps * 50^2 checks.
                    # Optimize: Keep a set of sums.
                    pass
            
            # Simpler: Just check if i creates a collision with existing S
            # Sums in S: {s1+s2 for s1 in S for s2 in S if s1<=s2}
            # New sums with i: {i+s for s in S} U {2*i}
            # Collision if any new sum is in existing sums (and not from i itself if i not in S)
            # Actually, if i is new, i+i is new. i+s is new unless i+s = x+y for x,y in S.
            
            # Efficient check:
            sums = set()
            for x in S:
                for y in S:
                    if x <= y:
                        sums.add(x+y)
            
            can_add = True
            for y in S:
                if i + y in sums:
                    can_add = False
                    break
                if 2*i in sums:
                    can_add = False
                    break
            if can_add:
                S.append(i)
        
        if len(S) > len(best_S):
            best_S = S
            
    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
        
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
