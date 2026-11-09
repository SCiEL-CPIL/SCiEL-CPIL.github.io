import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110
    
    S = []
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    # pairs[s] is a set of tuples (u, v) with u <= v, u+v=s
    # Using sets for fast removal? Or lists?
    # Sets are better for removal.
    pairs = [set() for _ in range(2 * max_val + 1)]
    
    def add(x):
        S.append(x)
        in_S[x] = True
        pairs[2*x].add((x, x))
        counts[2*x] += 1
        for y in S:
            if y != x:
                u, v = (x, y) if x < y else (y, x)
                pairs[x+y].add((u, v))
                counts[x+y] += 1
                
    def remove(x):
        S.remove(x)
        in_S[x] = False
        pairs[2*x].remove((x, x))
        counts[2*x] -= 1
        for y in S:
            if y != x:
                u, v = (x, y) if x < y else (y, x)
                pairs[x+y].remove((u, v))
                counts[x+y] -= 1

    def is_valid_add(x):
        if counts[2*x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            if counts[x+y] >= 2:
                return False
        return True

    # 1. Greedy Initialization
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    for x in candidates:
        if is_valid_add(x):
            add(x)
            
    best_size = len(S)
    
    while time.time() < deadline:
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        if is_valid_add(c):
            add(c)
            best_size = max(best_size, len(S))
            continue
            
        # Find blockers
        # For each y in S, if counts[c+y] >= 2, then any element in a pair summing to c+y is a blocker.
        blockers = set()
        for y in S:
            if y == c:
                continue
            s = c + y
            if counts[s] >= 2:
                # Pick one pair from pairs[s]
                # It's a set, so next(iter(...)) is O(1)
                if pairs[s]:
                    u, v = next(iter(pairs[s]))
                    blockers.add(u)
                    blockers.add(v)
                    
        # Try to swap with a blocker
        swapped = False
        # Shuffle blockers to randomize
        blocker_list = list(blockers)
        random.shuffle(blocker_list)
        for a in blocker_list:
            if a == c:
                continue
            # Check if swap a -> c is valid
            # We can use the optimized check or just try it?
            # Since a is a blocker, removing it reduces counts for sums involving a.
            # Adding c increases counts for sums involving c.
            # The conflict was at sum s = c+y.
            # Removing a (which is in a pair summing to s) reduces counts[s] by 1.
            # Adding c increases counts[s] by 1 (from pair c,y).
            # Net change 0. So s is safe.
            # But other sums might become invalid.
            # We must check all sums involving c.
            
            # Let's use the is_valid_swap logic but simplified?
            # Actually, we can just try the swap and rollback if invalid?
            # No, rollback is expensive.
            # Let's use the O(|S|) check.
            
            # Optimization: Only check sums that increase.
            # Sums increasing: 2c and c+z for z in S\{a}.
            
            valid = True
            # Check 2c
            s_2c = 2*c
            red = 0
            b = s_2c - a
            if in_S[b] and b != a:
                red = 1
            if counts[s_2c] + 1 - red > 2:
                valid = False
            
            if valid:
                for z in S:
                    if z == a:
                        continue
                    s = c + z
                    red = 0
                    b = s - a
                    if in_S[b] and b != a:
                        red = 1
                    if counts[s] + 1 - red > 2:
                        valid = False
                        break
                        
            if valid:
                remove(a)
                add(c)
                swapped = True
                break
                
        # If no swap found, try a few random swaps as a fallback?
        if not swapped:
            for _ in range(5):
                a = random.choice(S)
                if a == c:
                    continue
                # Quick check
                valid = True
                s_2c = 2*c
                red = 0
                b = s_2c - a
                if in_S[b] and b != a:
                    red = 1
                if counts[s_2c] + 1 - red > 2:
                    valid = False
                if valid:
                    for z in S:
                        if z == a:
                            continue
                        s = c + z
                        red = 0
                        b = s - a
                        if in_S[b] and b != a:
                            red = 1
                        if counts[s] + 1 - red > 2:
                            valid = False
                            break
                if valid:
                    remove(a)
                    add(c)
                    break

    return S
