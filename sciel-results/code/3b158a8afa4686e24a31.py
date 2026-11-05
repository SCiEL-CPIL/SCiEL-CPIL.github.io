import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    
    def is_valid_incremental(S, x):
        """
        Check if adding x to set S keeps it valid.
        S is a set of integers. x is an integer not in S.
        """
        S_list = list(S)
        n = len(S_list)
        
        # 1. Check for a = x
        # Need 3 distinct b, c, d in S such that (b+c+d) % x == 0
        if n >= 3:
            for i in range(n):
                b = S_list[i]
                for j in range(i+1, n):
                    c = S_list[j]
                    for k in range(j+1, n):
                        d = S_list[k]
                        if (b + c + d) % x == 0:
                            return False
        
        # 2. Check for a in S
        # Need 2 distinct b, c in S \ {a} such that (b+c+x) % a == 0
        for a in S_list:
            # We need to pick b, c from S \ {a}
            # Iterate through all pairs in S, skipping if one is a
            # Or just iterate through all pairs and check
            for i in range(n):
                b = S_list[i]
                if b == a:
                    continue
                for j in range(i+1, n):
                    c = S_list[j]
                    if c == a:
                        continue
                    if (b + c + x) % a == 0:
                        return False
        
        return True

    def is_valid_full(S):
        """
        Check if set S is valid.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 4:
            return True
        
        for idx_a, a in enumerate(S_list):
            others = S_list[:idx_a] + S_list[idx_a+1:]
            m = len(others)
            if m < 3:
                continue
            for i in range(m):
                b = others[i]
                for j in range(i+1, m):
                    c = others[j]
                    for k in range(j+1, m):
                        d = others[k]
                        if (b + c + d) % a == 0:
                            return False
        return True

    # Greedy construction
    def greedy(order):
        S = set()
        for x in order:
            if is_valid_incremental(S, x):
                S.add(x)
        return S

    all_numbers = list(range(1, 193))
    
    # Try descending order first
    best_set = greedy(sorted(all_numbers, reverse=True))
    best_size = len(best_set)
    
    # Try randomized orders
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        order = all_numbers[:]
        rng.shuffle(order)
        # Bias towards larger numbers
        order.sort(key=lambda x: -x + rng.uniform(-20, 20))
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search to improve
    def local_search(S, rng, max_iters=100):
        S = set(S)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding
            candidates = [x for x in all_numbers if x not in S]
            rng.shuffle(candidates)
            for x in candidates[:10]:
                if is_valid_incremental(S, x):
                    S.add(x)
                    improved = True
                    break
            if improved:
                continue
            
            # Try swap
            if len(S) > 0:
                elems = list(S)
                rng.shuffle(elems)
                for rem in elems[:5]:
                    candidates = [x for x in all_numbers if x not in S]
                    rng.shuffle(candidates)
                    for add in candidates[:10]:
                        new_S = (S - {rem}) | {add}
                        if is_valid_full(new_S):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return S

    best_set = local_search(best_set, rng)
    
    return list(best_set)
