import random
import time

def construct():
    deadline = time.time() + 80
    
    def make_bitset(S):
        # bitset for 1..128
        bs = 0
        for x in S:
            bs |= (1 << x)
        return bs
    
    def is_valid_fast(S, bs):
        """Fast validity check using bitset."""
        n = len(S)
        lst = sorted(S)
        for i in range(n):
            a = lst[i]
            for j in range(i+1, n):
                b = lst[j]
                s_ab = a + b
                if s_ab > 128:
                    continue
                if not (bs >> s_ab) & 1:
                    continue
                # Check if there exists c in S, c != a, c != b, such that a+c in S and b+c in S
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = lst[k]
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac <= 128 and s_bc <= 128:
                        if (bs >> s_ac) & 1 and (bs >> s_bc) & 1:
                            return False
                    elif s_ac > 128 or s_bc > 128:
                        continue
        return True
    
    def greedy_construct(rng, bias='uniform'):
        candidates = list(range(1, 129))
        if bias == 'uniform':
            rng.shuffle(candidates)
        elif bias == 'middle':
            # Weight toward middle values
            weights = [1.0 / abs(x - 64) + 0.1 for x in candidates]
            total = sum(weights)
            # Sample without replacement using weights
            chosen = []
            remaining = list(candidates)
            rem_weights = list(weights)
            for _ in range(len(candidates)):
                r = rng.random() * sum(rem_weights)
                cum = 0
                for idx, w in enumerate(rem_weights):
                    cum += w
                    if r <= cum:
                        chosen.append(remaining[idx])
                        remaining.pop(idx)
                        rem_weights.pop(idx)
                        break
            candidates = chosen
        S = set()
        bs = 0
        for x in candidates:
            S.add(x)
            bs |= (1 << x)
            if not is_valid_fast(S, bs):
                S.discard(x)
                bs &= ~(1 << x)
        return S, bs
    
    def local_search(S, bs, rng, time_limit):
        S = set(S)
        bs = make_bitset(S)
        best = set(S)
        best_bs = bs
        best_size = len(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            if rng.random() < 0.6:
                # Try add
                missing = [x for x in range(1, 129) if x not in S]
                if not missing:
                    continue
                x = rng.choice(missing)
                S.add(x)
                bs |= (1 << x)
                if is_valid_fast(S, bs):
                    if len(S) > best_size:
                        best = set(S)
                        best_bs = bs
                        best_size = len(S)
                else:
                    S.discard(x)
                    bs &= ~(1 << x)
            else:
                # Try remove
                if len(S) <= 1:
                    continue
                x = rng.choice(list(S))
                S.discard(x)
                bs &= ~(1 << x)
        
        return best, best_bs
    
    rng = random.Random(12345)
    
    # Start with trivial solution
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    restarts = 0
    while time.time() < deadline:
        # Choose bias
        if restarts % 3 == 0:
            bias = 'uniform'
        elif restarts % 3 == 1:
            bias = 'middle'
        else:
            bias = 'uniform'
        
        S, bs = greedy_construct(rng, bias)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
        
        # Local search
        remaining = deadline - time.time()
        if remaining > 3:
            ls_time = min(remaining * 0.25, 4.0)
            S2, bs2 = local_search(S, bs, rng, ls_time)
            if len(S2) > best_size:
                best_set = S2
                best_size = len(S2)
        
        # Refine best
        remaining = deadline - time.time()
        if remaining > 4:
            ls_time = min(remaining * 0.15, 2.0)
            S3, bs3 = local_search(set(best_set), make_bitset(best_set), rng, ls_time)
            if len(S3) > best_size:
                best_set = S3
                best_size = len(S3)
        
        restarts += 1
    
    return sorted(best_set)
