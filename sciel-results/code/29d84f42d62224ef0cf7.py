"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _check_valid(S):
    """Check if set S has no three distinct a,b,c with all pair sums in S."""
    Sset = set(S)
    elems = list(S)
    n = len(elems)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b) in Sset and (a+c) in Sset and (b+c) in Sset:
                    return False
    return True

def _greedy_add(S, Sset, candidates, order):
    """Try to add elements in the given order, maintaining validity."""
    added = 0
    for x in order:
        if x in Sset:
            continue
        # Check if adding x creates a forbidden triple
        # A forbidden triple with x as one of a,b,c:
        # Case 1: x is the largest of {a,b,c}, say c=x. Then a+b, a+x, b+x all in S.
        #   a,b in S, a+x in S, b+x in S, a+b in S.
        # Case 2: x is middle, say b=x. Then a+x, a+c, x+c all in S.
        #   a,c in S, a+x in S, a+c in S, x+c in S.
        # Case 3: x is smallest, say a=x. Then x+b, x+c, b+c all in S.
        #   b,c in S, x+b in S, x+c in S, b+c in S.
        
        ok = True
        # Case 1: x is c (largest). Need a,b in S with a+b in S, a+x in S, b+x in S.
        # a+x in S means a in S and a+x in S. Similarly b+x in S.
        # And a+b in S.
        # So: exists a,b in S, a<b, such that a+b in S, a+x in S, b+x in S.
        # We can iterate over a in S where a+x in S, then check b.
        for a in S:
            if a + x > N:
                continue
            if (a + x) not in Sset:
                continue
            # Now need b in S, b != a, such that b+x in S and a+b in S.
            for b in S:
                if b == a:
                    continue
                if b + x > N:
                    continue
                if (b + x) not in Sset:
                    continue
                if (a + b) in Sset:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        # Case 2: x is b (middle). Need a,c in S with a < x < c.
        # a+x in S, a+c in S, x+c in S.
        # a+x in S: a in S, a+x in S.
        # x+c in S: c in S, x+c in S.
        # a+c in S: a+c in S.
        # So exists a,c in S, a < x < c, a+x in S, x+c in S, a+c in S.
        for a in S:
            if a >= x:
                continue
            if a + x > N or (a + x) not in Sset:
                continue
            for c in S:
                if c <= x:
                    continue
                if x + c > N or (x + c) not in Sset:
                    continue
                if (a + c) in Sset:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        # Case 3: x is a (smallest). Need b,c in S, x < b < c.
        # x+b in S, x+c in S, b+c in S.
        for b in S:
            if b <= x:
                continue
            if x + b > N or (x + b) not in Sset:
                continue
            for c in S:
                if c <= b:
                    continue
                if x + c > N or (x + c) not in Sset:
                    continue
                if (b + c) in Sset:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        S.append(x)
        Sset.add(x)
        added += 1
    
    return added

def construct():
    start_time = time.time()
    best_size = 0
    best_set = None
    
    # Start with the upper half as a baseline
    base = list(range(N//2 + 1, N + 1))
    best_size = len(base)
    best_set = list(base)
    
    # Try greedy with different orderings
    all_nums = list(range(1, N + 1))
    
    num_restarts = 0
    while time.time() - start_time < DEADLINE - 5:
        # Create a random ordering with bias toward larger numbers
        # Weight: larger numbers get higher priority
        weights = [i + 1 for i in range(N)]  # weight[i] = i+1, so larger numbers have higher weight
        # Actually, let's use a mix: sometimes prefer large, sometimes random, sometimes prefer small
        strategy = random.random()
        if strategy < 0.4:
            # Prefer large numbers
            order = sorted(all_nums, key=lambda x: random.random() * 0.3 + x / N, reverse=True)
        elif strategy < 0.7:
            # Random order
            order = all_nums[:]
            random.shuffle(order)
        else:
            # Prefer small numbers (to see if that helps)
            order = sorted(all_nums, key=lambda x: random.random() * 0.3 + (N - x) / N, reverse=True)
        
        # Start with a random subset of the upper half plus some random lower numbers
        S = []
        Sset = set()
        
        # Include some large numbers
        large_nums = list(range(N//2 + 1, N + 1))
        random.shuffle(large_nums)
        # Include a random fraction of large numbers
        frac = random.uniform(0.5, 1.0)
        num_large = int(len(large_nums) * frac)
        for x in large_nums[:num_large]:
            S.append(x)
            Sset.add(x)
        
        # Maybe add some smaller numbers
        small_nums = list(range(1, N//2 + 1))
        random.shuffle(small_nums)
        num_small = random.randint(0, len(small_nums) // 3)
        for x in small_nums[:num_small]:
            S.append(x)
            Sset.add(x)
        
        # Now try to add remaining numbers in the chosen order
        remaining = [x for x in order if x not in Sset]
        _greedy_add(S, Sset, remaining, remaining)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        num_restarts += 1
    
    # Verify best set
    if best_set is not None and _check_valid(best_set):
        return [int(x) for x in best_set]
    
    # Fallback
    return [int(x) for x in base]
# EVOLVE-BLOCK-END
