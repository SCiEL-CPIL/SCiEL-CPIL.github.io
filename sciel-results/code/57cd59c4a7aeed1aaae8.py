import numpy as np

N = 50

def construct():
    # Generate a Sidon set in range(N)
    # A Sidon set has all pairwise sums distinct.
    # Simple greedy Sidon set generation.
    def generate_sidon(n, max_val):
        s = []
        for i in range(max_val):
            if len(s) >= n:
                break
            # Check if i can be added
            ok = True
            sums = set()
            for a in s:
                for b in s:
                    if a + b in sums:
                        ok = False
                        break
                if not ok: break
                sums.add(a + i)
            if ok:
                # Check self sum 2*i
                if 2*i in sums:
                    ok = False
            if ok:
                s.append(i)
        return s

    # We want K peaks.
    # Try K from 10 down to 5?
    # Max Sidon size for 50 is around 7-8.
    # Let's try to find a good set.
    
    best_h = None
    best_score = 0.0
    
    # Try different K
    for K in range(2, 12):
        s = generate_sidon(K, N)
        if len(s) < K:
            continue
        
        h = np.zeros(N)
        # Assign equal height
        h[s] = 1.0 / K # Normalize L1 to 1
        
        # Calculate Score
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        conv = np.convolve(h, h, mode='full')
        sup = np.max(conv)
        
        if l1 == 0 or sup == 0:
            continue
            
        score = l2_sq / (l1 * sup)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        # Fallback to seed
        return [1.0] * N
        
    return list(best_h)
