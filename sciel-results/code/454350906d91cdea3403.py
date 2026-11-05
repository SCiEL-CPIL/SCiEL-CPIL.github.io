import time

def construct():
    import time
    deadline = time.time() + 85  # safety margin
    
    N = 256
    # numbers 2..256, index i -> number i+2
    nums = list(range(2, 257))
    n = len(nums)
    
    # conflict bitsets
    from math import gcd
    conflict = [0]*n
    for i in range(n):
        a = nums[i]
        bits = 0
        for j in range(n):
            if i==j: continue
            b = nums[j]
            l = a*b//gcd(a,b)
            if l <= 256:
                bits |= (1<<j)
        conflict[i] = bits
    
    weights = [1.0/x for x in nums]
    
    def sol_weight(sol):
        s = 0.0
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length()-1
            s += weights[i]
            x ^= lsb
        return s
    
    def valid(sol):
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length()-1
            if sol & conflict[i]:
                # conflict with another in sol
                # conflict[i] doesn't include i itself
                return False
            x ^= lsb
        return True
    
    # Initial solution: greedy by ascending a (seed) 
    # Also try: all of (129,256]
    def make_initial():
        sol = 0
        for i in range(n):
            if (sol & conflict[i]) == 0:
                sol |= (1<<i)
        return sol
    
    best_sol = make_initial()
    best_w = sol_weight(best_sol)
    
    # Also try the (129,256] solution
    sol2 = 0
    for i in range(n):
        if nums[i] > 128:
            sol2 |= (1<<i)
    if valid(sol2):
        w2 = sol_weight(sol2)
        if w2 > best_w:
            best_w = w2
            best_sol = sol2
    
    # Local search
    def local_search(sol, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_move = None  # (new_sol, gain)
            # 1-add
            for j in range(n):
                if sol & (1<<j): continue
                if (sol & conflict[j]) == 0:
                    gain = weights[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (sol | (1<<j), gain)
            # 2-swap
            for i in range(n):
                if not (sol & (1<<i)): continue
                base = sol ^ (1<<i)
                for j in range(n):
                    if sol & (1<<j): continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1<<j), gain)
            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True
        return sol, w
    
    best_sol, best_w = local_search(best_sol, best_w, deadline)
    
    # A few restarts with different inits if time
    import random
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 8:
        attempts += 1
        # random greedy init
        order = list(range(n))
        random.shuffle(order)
        sol = 0
        for i in order:
            if (sol & conflict[i]) == 0:
                sol |= (1<<i)
        w = sol_weight(sol)
        sol, w = local_search(sol, w, deadline)
        if w > best_w:
            best_w = w
            best_sol = sol
    
    return [nums[i] for i in range(n) if best_sol & (1<<i)]
