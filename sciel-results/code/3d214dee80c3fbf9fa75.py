import random
import time

def construct():
    start_time = time.time()
    deadline = 135.0  # Keep 15s margin before 150s limit

    NEED = set(range(1, 31))
    MAXN = 32

    def valid_sidon(s):
        s = sorted(set(s))
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                d = s[j] - si
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def has_all_need(s):
        s = sorted(set(s))
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - si)
        return NEED.issubset(diffs)

    def is_valid(s):
        return valid_sidon(s) and has_all_need(s)

    def get_cost(s):
        s = sorted(set(s))
        if 0 not in s or len(s) > MAXN:
            return 10**9
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                d = s[j] - si
                if d in diffs:
                    return 10**8 + max(s)
                diffs.add(d)
        missing = NEED - diffs
        if missing:
            return 10**6 + len(missing) * 1000 + max(s)
        return max(s)

    def make_greedy(order):
        """Build a Sidon set by adding elements one at a time, 
        prioritizing differences in 'order'."""
        a = [0]
        current_diffs = set()
        
        # We need to cover all d in 1..30
        # We'll try to build a set that is as small in max as possible
        # Strategy: 
        # 1. Start with 0.
        # 2. For each d in 1..30 (or in some order), if d is not yet a difference, 
        #    find the smallest x > 0 such that adding x creates difference d 
        #    and maintains Sidon property.
        
        # Actually, a better greedy:
        # Maintain set A.
        # While there is a d in 1..30 not in diffs(A):
        #    Find smallest x not in A such that A U {x} is Sidon and x - a for some a in A equals d.
        #    Actually, we just need x such that for some a in A, x-a = d (so x = a+d) OR a-x = d (so x = a-d, but x>0 and a>=0, so x=a-d implies a>d).
        #    We want to minimize max(A).
        
        # Let's try a different greedy:
        # Sort candidates x = 1, 2, 3, ...
        # Add x to A if A U {x} is Sidon AND (it adds some new needed difference OR we have fewer elements than needed to be "dense" enough?)
        # No, we just need to ensure all 1..30 are covered.
        
        # A simple effective greedy for Sidon:
        # 1. Start A = [0]
        # 2. For x in 1, 2, 3, ...:
        #    If A U {x} is Sidon:
        #       Check if adding x helps cover a missing difference.
        #       If yes, add it.
        #       Else, maybe add it if we haven't reached size limit and it keeps max small? 
        #       Actually, we can just add it if it's Sidon. This tends to produce small max sets.
        #       Then check if all 1..30 are covered. If yes, stop.
        
        A = [0]
        diffs = set()
        for x in range(1, 10000):
            # Check Sidon
            new_diffs = diffs.copy()
            ok = True
            for a in A:
                d = x - a
                if d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            # Check if this adds a needed difference
            # Actually, just add it if it's Sidon. 
            # We stop when all 1..30 are covered.
            A.append(x)
            diffs = new_diffs
            if NEED.issubset(diffs):
                break
        return A

    # Initial best
    best_a = make_greedy(None)
    best_cost = get_cost(best_a)
    
    # Try different random restarts of greedy with slight variations?
    # Or just use local search.

    # Local Search (Simulated Annealing)
    cur = list(best_a)
    cur_cost = get_cost(cur)
    
    temp = 200.0
    cool = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        a = list(cur)
        # Random move
        mt = random.randint(0, 3)
        
        if mt == 0: # Modify
            if len(a) > 1:
                idx = random.randint(1, len(a) - 1)
                delta = random.randint(-20, 20)
                a[idx] += delta
                if a[idx] < 1:
                    a[idx] = random.randint(1, max(a) + 10)
        
        elif mt == 1: # Swap
            if len(a) > 2:
                i, j = random.sample(range(1, len(a)), 2)
                a[i], a[j] = a[j], a[i]
        
        elif mt == 2: # Insert
            if len(a) < MAXN:
                val = random.randint(1, max(a) + 20)
                a.append(val)
        
        else: # Remove
            if len(a) > 1:
                idx = random.randint(1, len(a) - 1)
                a.pop(idx)
        
        # Clean up
        a = sorted(set(a))
        if 0 not in a:
            a.append(0)
        
        new_cost = get_cost(a)
        
        if new_cost < cur_cost:
            cur = a
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_a = list(cur)
                best_cost = cur_cost
        else:
            # Metropolis
            delta_cost = new_cost - cur_cost
            if delta_cost == 0:
                accept = True
            else:
                accept = random.random() < (1.0 / (1.0 + delta_cost / (temp + 1e-9)))
            
            if accept:
                cur = a
                cur_cost = new_cost
        
        temp *= cool

    # Final validation
    if is_valid(best_a):
        return best_a
    
    # If best_a is somehow invalid, try to repair or return cur
    if is_valid(cur):
        return cur
        
    # Fallback: return a known valid set (large diameter)
    # Greedy from 0
    A = [0]
    diffs = set()
    for x in range(1, 20000):
        new_diffs = diffs.copy()
        ok = True
        for a in A:
            d = x - a
            if d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        A.append(x)
        diffs = new_diffs
        if NEED.issubset(diffs):
            return A
            
    return A
