"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import gcd

NMAX = 256
N_ELEMENTS = NMAX - 1  # 255 elements (values 2..256)
# Index: value v -> idx = v - 2

def _precompute():
    """
    Precompute:
    - mask_for_d[d]: bitmask of c in [2,256] such that d | c
    - mask_req[x][b]: bitmask of c in [2,256] such that (x // gcd(x,b)) | c
      i.e., x | b*c  iff  c in mask_req[x][b]
    """
    mask_for_d = [0] * (NMAX + 1)
    for d in range(1, NMAX + 1):
        mask = 0
        for c in range(2, NMAX + 1):
            if c % d == 0:
                mask |= (1 << (c - 2))
        mask_for_d[d] = mask

    mask_req = {}
    for x in range(2, NMAX + 1):
        mask_req[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                mask_req[x][b] = 0
            else:
                d = x // gcd(x, b)
                mask_req[x][b] = mask_for_d[d]
    return mask_req

def _conflict_count(x, S_mask, mask_req):
    """
    Count number of ordered pairs (b, c) in S, b != c, b,c != x,
    such that x | b*c.
    For each b in S (b != x): count c in S such that c != b, c != x, and (x//gcd(x,b)) | c.
    """
    x_idx = x - 2
    cnt = 0
    for b in range(2, NMAX + 1):
        if b == x:
            continue
        if not ((S_mask >> (b - 2)) & 1):
            continue
        b_idx = b - 2
        valid = S_mask & mask_req[x][b]
        valid &= ~(1 << b_idx)
        valid &= ~(1 << x_idx)
        cnt += valid.bit_count()
    return cnt

def _can_add(x, S_mask, mask_req):
    """
    Check if x can be added to S (represented by S_mask) without violating constraint.
    For every pair (b,c) in S with b != c:
      - x | b*c ?  -> check via mask_req[x][b]
      - b | x*c ?  -> check via mask_req[b][x]
      - c | x*b ?  -> check via mask_req[c][x]
    """
    x_idx = x - 2
    # For each b in S:
    for b in range(2, NMAX + 1):
        if not ((S_mask >> (b - 2)) & 1):
            continue
        b_idx = b - 2
        # Check x | b*c for some c in S, c != b, c != x
        valid_c = S_mask & mask_req[x][b]
        valid_c &= ~(1 << b_idx)
        valid_c &= ~(1 << x_idx)
        if valid_c:
            return False
        # Check b | x*c for some c in S, c != b, c != x
        valid_c2 = S_mask & mask_req[b][x]
        valid_c2 &= ~(1 << b_idx)
        valid_c2 &= ~(1 << x_idx)
        if valid_c2:
            return False
        # Check c | x*b for some c in S, c != b, c != x
        # c | x*b  iff  (x*b) % c == 0
        # For each c in S: check if c divides x*b
        # But we can use: c | x*b iff (x*b//gcd(x*b,c)) == 1... not directly maskable.
        # Alternative: iterate c in S
        xb = x * b
        for c in range(2, NMAX + 1):
            if c == b or c == x:
                continue
            if not ((S_mask >> (c - 2)) & 1):
                continue
            if xb % c == 0:
                return False
    return True

def _can_add_fast(x, S_list, mask_req):
    """Faster check using list iteration."""
    x_idx = x - 2
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        b_idx = b - 2
        # x | b*c?
        valid_c = 0
        for j in range(n):
            c = S_list[j]
            if c == b or c == x:
                continue
            d = x // gcd(x, b)
            if c % d == 0:
                return False
            # b | x*c?
            d2 = b // gcd(b, x)
            if c % d2 == 0:
                return False
            # c | x*b?
            if (x * b) % c == 0:
                return False
    return True

def _greedy_inclusion(mask_req, order, deadline):
    """
    Greedy inclusion: try to add elements in given order.
    Returns the set as a list.
    """
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        # Check if x can be added
        ok = True
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(n):
                if i == j:
                    continue
                c = S[j]
                # x | b*c?
                d = x // gcd(x, b)
                if c % d == 0:
                    ok = False
                    break
                # b | x*c?
                d2 = b // gcd(b, x)
                if c % d2 == 0:
                    ok = False
                    break
                # c | x*b?
                if (x * b) % c == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(x)
            S_set.add(x)
    return S

def _simulated_annealing(S, mask_req, deadline, seed=42):
    """
    Simulated annealing on the set S.
    Moves: add a random element, remove a random element, swap.
    Objective: maximize |S| subject to validity.
    We only accept valid states.
    """
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    start_temp = 1.0
    end_temp = 0.01
    steps = max(1, int((deadline - time.time()) * 50))  # ~50 steps/sec
    steps = min(steps, 50000)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (step / max(1, steps - 1))
        
        move = rng.random()
        
        if move < 0.35:
            # Try add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            # Check validity
            ok = True
            n = len(S)
            for i in range(n):
                b = S[i]
                for j in range(n):
                    if i == j:
                        continue
                    c = S[j]
                    d = x // gcd(x, b)
                    if c % d == 0:
                        ok = False
                        break
                    d2 = b // gcd(b, x)
                    if c % d2 == 0:
                        ok = False
                        break
                    if (x * b) % c == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S.append(x)
                S_set.add(x)
                if len(S) > best_size:
                    best = S[:]
                    best_size = len(S)
            # else: reject (invalid)
            
        elif move < 0.65:
            # Try remove
            if len(S) <= 2:
                continue
            i = rng.randrange(len(S))
            x = S.pop(i)
            S_set.discard(x)
            # Always accept (removal preserves validity)
            # Try to add something back
            candidates = [y for y in range(2, NMAX + 1) if y not in S_set]
            if candidates:
                # Try a few
                rng.shuffle(candidates)
                for y in candidates[:20]:
                    ok = True
                    n = len(S)
                    for ii in range(n):
                        b = S[ii]
                        for jj in range(n):
                            if ii == jj:
                                continue
                            c = S[jj]
                            d = y // gcd(y, b)
                            if c % d == 0:
                                ok = False
                                break
                            d2 = b // gcd(b, y)
                            if c % d2 == 0:
                                ok = False
                                break
                            if (y * b) % c == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        S.append(y)
                        S_set.add(y)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break
        else:
            # Swap: remove one, add another
            if len(S) <= 2:
                continue
            i = rng.randrange(len(S))
            old = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if not candidates:
                S[i] = old
                continue
            rng.shuffle(candidates)
            swapped = False
            for x in candidates[:30]:
                ok = True
                n = len(S_temp)
                for ii in range(n):
                    b = S_temp[ii]
                    for jj in range(n):
                        if ii == jj:
                            continue
                        c = S_temp[jj]
                        d = x // gcd(x, b)
                        if c % d == 0:
                            ok = False
                            break
                        d2 = b // gcd(b, x)
                        if c % d2 == 0:
                            ok = False
                            break
                        if (x * b) % c == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    break
            if not swapped:
                # Revert
                S = S_temp + [old]
                S_set = set(S)
    
    return best

def _greedy_by_burden(mask_req, deadline):
    """
    Greedy inclusion ordered by 'burden': elements that are more likely to divide
    products of others go first (they're more restrictive, so placing them early
    helps filter). Actually, we want LESS restrictive first.
    
    Burden of x = number of pairs (b,c) in [2,256] such that x | b*c.
    Lower burden = less restrictive = should be added first? No...
    
    Actually for greedy inclusion, we want to add elements that are easy to include.
    An element x is 'easy' if few pairs (b,c) in the current set violate x|bc etc.
    
    Let's try: sort by number of multiples in [2,256] (fewer multiples = less likely
    to divide products). Primes have few multiples.
    """
    # Compute for each x: count of c in [2,256] such that x | c (i.e., multiples of x)
    burden = {}
    for x in range(2, NMAX + 1):
        cnt = 0
        for c in range(2, NMAX + 1):
            if c % x == 0:
                cnt += 1
        burden[x] = cnt
    
    # Sort by burden ascending (less restrictive first)
    order = sorted(range(2, NMAX + 1), key=lambda x: burden[x])
    return _greedy_inclusion(mask_req, order, deadline)

def _greedy_by_burden_desc(mask_req, deadline):
    """Sort by burden descending (most restrictive first)."""
    burden = {}
    for x in range(2, NMAX + 1):
        cnt = 0
        for c in range(2, NMAX + 1):
            if c % x == 0:
                cnt += 1
        burden[x] = cnt
    
    order = sorted(range(2, NMAX + 1), key=lambda x: -burden[x])
    return _greedy_inclusion(mask_req, order, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    mask_req = _precompute()
    
    best = None
    best_size = 0
    
    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Strategy 1: Greedy inclusion - largest first
    if time.time() < deadline - 20:
        S1 = _greedy_inclusion(mask_req, list(range(NMAX, 1, -1)), deadline - 20)
        consider(S1)
    
    # Strategy 2: Greedy inclusion - smallest first
    if time.time() < deadline - 20:
        S2 = _greedy_inclusion(mask_req, list(range(2, NMAX + 1)), deadline - 20)
        consider(S2)
    
    # Strategy 3: Greedy by burden (ascending)
    if time.time() < deadline - 15:
        S3 = _greedy_by_burden(mask_req, deadline - 15)
        consider(S3)
    
    # Strategy 4: Greedy by burden (descending)
    if time.time() < deadline - 15:
        S4 = _greedy_by_burden_desc(mask_req, deadline - 15)
        consider(S4)
    
    # Strategy 5: Randomized greedy (multiple seeds)
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(2, NMAX + 1))
        rng.shuffle(order)
        S_rand = _greedy_inclusion(mask_req, order, deadline - 10)
        consider(S_rand)
    
    # Polish with simulated annealing
    if best and time.time() < deadline - 5:
        for seed in [42, 123, 999, 4242]:
            if time.time() > deadline - 3:
                break
            S_sa = _simulated_annealing(best, mask_req, deadline - 3, seed=seed)
            consider(S_sa)
    
    # Final: try to add any remaining elements
    if best and time.time() < deadline - 1:
        S = list(best)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                # Quick check
                ok = True
                n = len(S)
                for i in range(n):
                    b = S[i]
                    for j in range(n):
                        if i == j:
                            continue
                        c = S[j]
                        d = x // gcd(x, b)
                        if c % d == 0:
                            ok = False
                            break
                        d2 = b // gcd(b, x)
                        if c % d2 == 0:
                            ok = False
                            break
                        if (x * b) % c == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    S.append(x)
                    S_set.add(x)
        consider(S)
    
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
    
    return [int(x) for x in best]
