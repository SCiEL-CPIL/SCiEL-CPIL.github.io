import random
import time

def _check_violation_with_x(S_set, x):
    """Check if adding x to S_set creates a violation. Returns True if violation exists."""
    # x is not in S_set yet
    # Violation involving x as a member:
    # Need distinct b,c in S_set (b != c, b != x, c != x) such that
    # x+b in S_set, x+c in S_set, b+c in S_set
    # Also x+b != x+c means b != c (already ensured)
    # Also need x+b != b+c => x != c (ensured), x+c != b+c => x != b (ensured)
    # Also x+b != x+c => b != c (ensured)
    
    # For each pair b,c in S_set with b < c:
    # Check if x+b in S_set, x+c in S_set, b+c in S_set
    # But also need x+b, x+c, b+c to be potentially equal to x or each other
    
    # Simpler: iterate over b in S_set, c in S_set, b != c
    # This is O(n^2) per check
    
    elems = list(S_set)
    n = len(elems)
    for i in range(n):
        b = elems[i]
        if b == x:
            continue
        sb = x + b
        if sb not in S_set:
            continue
        for j in range(i+1, n):
            c = elems[j]
            if c == x:
                continue
            sc = x + c
            if sc not in S_set:
                continue
            bc = b + c
            if bc in S_set:
                # Check all three sums are distinct from each other and from a,b,c
                # Actually the problem says "the six displayed values need not be mutually distinct"
                # So a,b,c distinct, and a+b, a+c, b+c just need to be in S
                # But we also need a+b, a+c, b+c to be in S (which they are by construction)
                # The only constraint is a,b,c distinct
                return True
    
    # Violation where x is a sum:
    # x = a+b for some a,b in S_set, a != b
    # Then need c in S_set, c != a, c != b, such that a+c in S_set and b+c in S_set
    for i in range(n):
        a = elems[i]
        if a == x:
            continue
        b = x - a
        if b == a or b == x or b not in S_set:
            continue
        # a+b = x, both in S_set
        # Now need c in S_set, c != a, c != b, with a+c in S_set and b+c in S_set
        for j in range(n):
            c = elems[j]
            if c == a or c == b or c == x:
                continue
            if (a + c) in S_set and (b + c) in S_set:
                return True
    
    return False

def _is_valid_full(S_set):
    """Full validity check."""
    elems = sorted(S_set)
    n = len(elems)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = set(range(65, 129))
    best_size = 64
    
    def try_add_greedy(S, candidates, rng):
        """Try to add candidates to S greedily."""
        S = set(S)
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if not _check_violation_with_x(S, x):
                S.add(x)
        return S
    
    # Multiple greedy attempts
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        S = set(range(65, 129))
        candidates = list(range(1, 65))
        rng.shuffle(candidates)
        S = try_add_greedy(S, candidates, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search
    def local_search(S, seed, max_iter=5000):
        rng = random.Random(seed)
        S = set(S)
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Try adding a random element
            available = [x for x in range(1, 129) if x not in S]
            if available:
                x = rng.choice(available)
                if not _check_violation_with_x(S, x):
                    S.add(x)
                    continue
            # Try removing one and adding two
            if len(S) > 5:
                x = rng.choice(list(S))
                S2 = S - {x}
                available = [y for y in range(1, 129) if y not in S2]
                rng.shuffle(available)
                added = 0
                for y in available:
                    if added >= 2:
                        break
                    if not _check_violation_with_x(S2, y):
                        S2.add(y)
                        added += 1
                if len(S2) > len(S):
                    S = S2
                elif len(S2) == len(S) and rng.random() < 0.2:
                    S = S2
        return S
    
    for seed in range(50):
        if time.time() > deadline:
            break
        if seed < 5:
            S = set(best_set)
        else:
            rng = random.Random(10000 + seed)
            S = set(range(65, 129))
            candidates = list(range(1, 65))
            rng.shuffle(candidates)
            S = try_add_greedy(S, candidates, rng)
        
        S = local_search(S, 20000 + seed, max_iter=3000)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final polish
    if time.time() < deadline - 5:
        for seed in range(30):
            if time.time() > deadline:
                break
            S = local_search(set(best_set), 50000 + seed, max_iter=5000)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
    
    return sorted(best_set)

