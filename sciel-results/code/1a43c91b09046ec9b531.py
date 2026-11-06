import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    
    # Precompute divisibility: div[a] = set of s in [2a, 2n] such that a divides s
    # Actually, we need: for a in S, b,c in S with b<=c, a < b, check a | (b+c)
    # b+c ranges from 2*(a+1) to 2*n
    
    # For each a, compute which sums (b+c) are divisible by a
    # Then for a given set S, a is "bad" if there exist b,c in S with a<b<=c and a|(b+c)
    
    # Key insight: if we have a set S, for each a in S, we need no pair (b,c) in S
    # with a < b <= c such that a | (b+c).
    
    # Let's think about structure. If S is an interval [L, n], then for a in [L, n],
    # b,c in [L, n] with b>=a+1, c>=b, so b+c >= 2a+2.
    # a | (b+c) means b+c = ka for some integer k.
    # Since b,c >= a+1, b+c >= 2a+2, so k >= 2 (at least k=2 possible if b+c=2a, but b,c>a so b+c>2a).
    # Actually b>=a+1, c>=a+1, so b+c >= 2a+2. And b,c <= n, so b+c <= 2n.
    # So k ranges from ceil((2a+2)/a) = 2 (since 2a+2 > 2a) up to floor(2n/a).
    # Wait, b+c >= 2(a+1) = 2a+2, and a | (b+c), so b+c >= 2a (next multiple after a is 2a, but 2a < 2a+2).
    # So the smallest multiple of a that is >= 2a+2 is: if a divides 2a+2? No, 2a+2 = 2(a+1).
    # The multiples of a are a, 2a, 3a, ...
    # We need b+c >= 2a+2 and a | (b+c).
    # 2a < 2a+2, so 2a is too small. Next is 3a. Is 3a >= 2a+2? 3a >= 2a+2 iff a >= 2. Yes for a>=2.
    # For a=1: b,c >= 2, b+c >= 4. 1 divides everything, so a=1 is always bad if |S|>=2 with elements >1.
    # Actually if a=1 is in S and there are b,c in S with 1<b<=c, then 1 | (b+c) always. So a=1 can only be in S if S has at most 2 elements total and no two elements both >1... actually if S={1,x} with x>1, then a=1, need b,c with 1<b<=c in S. Only x is >1, can't have two. So S={1,x} is valid. S={1} is valid.
    
    # So the constraint is: for each a in S, there should be NO two elements b,c in S with a<b<=c and a|(b+c).
    
    # Efficient check: for a given sorted S, for each a=S[i], look at S[j] for j>i.
    # For each b=S[j] (j>i), we need c=S[k] (k>=j) such that a|(b+c).
    # Equivalently, c ≡ -b (mod a), c in S, c>=b, c>a.
    
    # Let me write a faster validity check using this.
    
    def is_valid_set(S):
        S_list = sorted(S)
        m = len(S_list)
        # For each a, check if there exist b,c in S with a<b<=c and a|(b+c)
        for i in range(m):
            a = S_list[i]
            if a == 1:
                # 1 divides everything. Need two elements >1 in S.
                if m - i - 1 >= 2:
                    return False
                continue
            # For each b > a in S, check if there's c >= b in S with c ≡ -b (mod a)
            # Use a set for O(1) lookup
            S_set = S  # already a set or we convert
            for j in range(i + 1, m):
                b = S_list[j]
                # c must satisfy c ≡ -b (mod a), c >= b, c in S, c <= n
                # -b mod a = (a - b%a) % a
                target = (-b) % a
                # c = target + t*a for t such that c >= b and c <= n
                # c >= b: target + t*a >= b => t >= (b - target)/a
                # If target == 0, c = t*a, need t*a >= b, t >= ceil(b/a)
                # If target > 0, c = target + t*a >= b => t >= ceil((b - target)/a)
                if target == 0:
                    t_min = (b + a - 1) // a  # ceil(b/a)
                else:
                    t_min = (b - target + a - 1) // a  # ceil((b-target)/a)
                    if t_min < 0:
                        t_min = 0
                # Check c = target + t*a for t >= t_min, c <= n, c in S
                c = target + t_min * a
                if c < b:
                    c += a
                while c <= n:
                    if c in S_set:
                        return False
                    c += a
        return True
    
    # Strategy: The optimal set is likely a large upper interval plus some carefully chosen lower elements.
    # Let's first find the best upper interval [L, n] that is valid.
    
    # For [L, n] to be valid: for each a in [L, n], no b,c in [L, n] with a<b<=c and a|(b+c).
    # Since b,c >= a+1, b+c >= 2a+2.
    # a|(b+c) means b+c = ka for k >= 2 (actually k >= ceil((2a+2)/a) = 2 for a>=2, but 2a < 2a+2 so k>=3 for the smallest... wait)
    # b+c >= 2a+2. Multiples of a: 2a, 3a, 4a, ...
    # 2a < 2a+2, so 2a is not achievable (b+c > 2a).
    # 3a >= 2a+2 iff a >= 2. So for a >= 2, the smallest possible multiple is 3a (if 3a >= 2a+2, i.e., a>=2).
    # But we also need b,c <= n, so b+c <= 2n. So k <= floor(2n/a).
    
    # For the interval [L, n], a is in [L, n]. b,c in [a+1, n].
    # We need: no b,c in [a+1, n] with b<=c and b+c = ka for integer k.
    # This is equivalent to: for each k in [3, floor(2n/a)] (or [2, ...] if 2a >= 2a+2 which is false),
    #   there are no b,c in [a+1, n] with b<=c and b+c=ka.
    # b+c=ka, b<=c, b>=a+1, c<=n => b in [a+1, floor(ka/2)] and c=ka-b in [ceil(ka/2), n].
    # So b in [max(a+1, ka-n), min(floor(ka/2), n-a)] (since c=ka-b>=b means b<=ka/2, and c<=n means b>=ka-n).
    
    # For interval [L,n], a >= L. The worst case is small a.
    # Let's just check empirically for each L.
    
    best_L = n
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S = set(range(L, n + 1))
        if is_valid_set(S):
            best_L = L
            break
    
    # Start with [best_L, n]
    current_set = set(range(best_L, n + 1))
    best_size = len(current_set)
    best_result = set(current_set)
    
    # Try to add elements from 1 to best_L-1
    # Process in random order, and also try multiple orders
    
    def try_add_elements(S, candidates, time_limit):
        S = set(S)
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x not in S:
                test = S | {x}
                if is_valid_set(test):
                    S.add(x)
        return S
    
    # Multiple attempts with different random seeds
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        S = set(range(best_L, n + 1))
        candidates = list(range(1, best_L))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x not in S:
                test = S | {x}
                if is_valid_set(test):
                    S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_result = set(S)
    
    # Also try starting from different L values
    for L in range(best_L, min(best_L + 20, n + 1)):
        if time.time() > deadline - 3:
            break
        S = set(range(L, n + 1))
        if not is_valid_set(S):
            continue
        candidates = list(range(1, L))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3:
                break
            if x not in S:
                test = S | {x}
                if is_valid_set(test):
                    S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_result = set(S)
    
    # Local search: try swapping elements
    if time.time() < deadline - 2:
        S = set(best_result)
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            # Try to add a random element not in S
            x = random.randint(1, n)
            if x in S:
                continue
            test = S | {x}
            if is_valid_set(test):
                S = test
                best_size = len(S)
                best_result = set(S)
    
    return list(best_result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
