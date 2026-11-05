import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin before the 60s kill

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            # Verify validity
            if is_valid(S):
                best_set = set(S)
                best_size = len(best_set)

    def is_valid(S):
        # Bitset DP
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
            if (reachable >> 200) & 1:
                return False
        return True

    def get_bitset(S):
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
        return reachable

    def improve_additions(S):
        # Greedily add elements if possible, trying multiple orderings
        S_curr = set(S)
        base_reach = get_bitset(S_curr)
        
        # Try adding in a few different orders to escape local optima
        for order_type in ['random', 'small_to_large', 'large_to_small']:
            changed = True
            while changed:
                changed = False
                missing = [x for x in range(1, 101) if x not in S_curr]
                if not missing:
                    break
                if order_type == 'random':
                    random.shuffle(missing)
                elif order_type == 'small_to_large':
                    missing.sort()
                else:
                    missing.sort(reverse=True)
                
                for x in missing:
                    new_reach = base_reach | (base_reach << x)
                    if not ((new_reach >> 200) & 1):
                        S_curr.add(x)
                        base_reach = new_reach
                        changed = True
        return S_curr

    def find_witness(S):
        # Returns a subset T of S summing to 200, or None
        elements = list(S)
        random.shuffle(elements)
        
        dp = [False] * 201
        dp[0] = True
        path = [None] * 201  # path[s] = (x, s-x)
        
        for x in elements:
            for s in range(200, x - 1, -1):
                if dp[s - x] and not dp[s]:
                    dp[s] = True
                    path[s] = (x, s - x)
                    
        if dp[200]:
            T = []
            s = 200
            while s > 0:
                x, prev_s = path[s]
                T.append(x)
                s = prev_s
            return T
        return None

    # 1. Deterministic Greedies
    # Large to Small
    s = set()
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.add(x)
            reachable = new_reach
    update_best(s)
    
    # Small to Large
    s = set()
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.add(x)
            reachable = new_reach
    update_best(s)
    
    # 2. Randomized Greedy
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        s = set()
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not ((new_reach >> 200) & 1):
                s.add(x)
                reachable = new_reach
        update_best(s)
        
        # 3. Subtractive + Improve
        S = set(range(1, 101))
        
        # Subtractive Phase
        while True:
            T = find_witness(S)
            if T is None:
                break
            x_to_remove = random.choice(T)
            S.discard(x_to_remove)
            if not S:
                break
        
        # Improvement Phase
        S_improved = improve_additions(S)
        update_best(S_improved)
        
        # 4. Local Search: Swap elements
        if time.time() < deadline:
            S_curr = set(best_set)
            base_reach = get_bitset(S_curr)
            missing = [x for x in range(1, 101) if x not in S_curr]
            if missing:
                random.shuffle(missing)
                for x_add in missing[:10]:  # Try a few additions
                    for x_rem in list(S_curr)[:10]:  # Try a few removals
                        S_test = S_curr - {x_rem} | {x_add}
                        if len(S_test) > best_size:
                            if is_valid(S_test):
                                S_curr = S_test
                                base_reach = get_bitset(S_curr)
                                update_best(S_curr)
    
    return list(best_set)
