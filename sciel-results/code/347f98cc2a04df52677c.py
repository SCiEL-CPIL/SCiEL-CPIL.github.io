import time

def construct():
    start_time = time.time()
    limit = 90  # seconds
    
    N = 12
    period = 27720
    
    best_m = 0
    best_f = 0
    
    # Precompute factors? No, just use mod.
    
    def can_match(m, L):
        # Generate candidate offsets for each i in 1..12
        # offset d means value x = m + d
        # We need d in 1..L such that (m+d) % i == 0
        cands = []
        for i in range(1, N + 1):
            lst = []
            # First multiple of i > m
            # m + d >= m + 1
            # (m + d) % i == 0
            # d % i == (-m) % i
            rem = (-m) % i
            if rem == 0:
                first_d = i
            else:
                first_d = rem
            # Wait, if m=0, i=2. (-0)%2=0. first_d=2?
            # m=0, L=2. Values 1, 2.
            # i=1: d=1, 2.
            # i=2: d=2.
            # Let's check:
            # d=1: (0+1)%2 = 1 != 0
            # d=2: (0+2)%2 = 0.
            # So first_d should be the smallest d >= 1 such that (m+d)%i==0.
            
            # Let's re-calculate first_d carefully.
            # We want smallest d >= 1 such that (m+d) % i == 0.
            # (m+d) % i == 0  =>  d % i == (-m) % i.
            # Let r = (-m) % i.
            # If r == 0, d must be a multiple of i. Smallest d >= 1 is i.
            # If r != 0, d must be congruent to r mod i. Smallest d >= 1 is r.
            
            r = (-m) % i
            if r == 0:
                first_d = i
            else:
                first_d = r
                
            # Add all d = first_d, first_d + i, ... <= L
            for d in range(first_d, L + 1, i):
                lst.append(d)
            cands.append(lst)
            
        # Sort by length of candidate list
        # We want to assign i's to distinct d's.
        # It's easier to iterate over i's and pick a d.
        # Order i's by len(cands[i]) ascending.
        order = list(range(N))
        order.sort(key=lambda idx: len(cands[idx]))
        
        used = [False] * (L + 1)
        
        def dfs(k):
            if k == N:
                return True
            i = order[k]
            for d in cands[i]:
                if not used[d]:
                    used[d] = True
                    if dfs(k + 1):
                        return True
                    used[d] = False
            return False
        
        return dfs(0)

    def compute_f(m):
        # Max L is small, try up to 200
        for L in range(1, 200):
            if can_match(m, L):
                return L
        return 200

    # Search m in one period
    # 27720 iterations.
    # Each compute_f is fast.
    
    for m in range(period):
        if m % 1000 == 0:
            if time.time() - start_time > limit - 5:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return best_m
