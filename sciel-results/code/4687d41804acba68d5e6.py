import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute conflict bitsets
    # For each x, conflicts[x] is a bitset of numbers y such that x and y
    # would form a 3-AP with some third number z (i.e., there exists z in [1,n]
    # such that {x,y,z} is a 3-AP with x != y != z != x)
    # Actually, we need: if we have a set S and want to add x, we need to check
    # if any pair (a,b) in S forms a 3-AP with x.
    # 
    # Alternative: For each pair (a,b) with a < b, if (a+b) is even, mid = (a+b)//2,
    # then a, mid, b is a 3-AP. So a and b conflict if mid exists.
    # Also, for any a, b: if 2a = b + c, then b, a, c is a 3-AP.
    # 
    # Let's define: two numbers x and y "conflict" if there exists a third number z
    # in [1, n] such that {x, y, z} forms a 3-AP (with all three distinct).
    # 
    # A 3-AP is (p, q, r) with p + r = 2q.
    # So for a pair (x, y):
    #   Case 1: x is middle: y + z = 2x => z = 2x - y. Need 1 <= z <= n, z != x, z != y.
    #   Case 2: y is middle: x + z = 2y => z = 2y - x. Need 1 <= z <= n, z != x, z != y.
    #   Case 3: z is middle: x + y = 2z => z = (x+y)/2. Need x+y even, 1 <= z <= n, z != x, z != y.
    #
    # So x and y conflict if any of these z values is valid.

    # Build conflict bitsets
    # conflicts[x] is a bitmask where bit y is set if x and y conflict
    conflicts = [0] * (n + 1)
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            conflict = False
            # Case 1: x is middle, z = 2x - y
            z = 2 * x - y
            if 1 <= z <= n and z != x and z != y:
                conflict = True
            # Case 2: y is middle, z = 2y - x
            if not conflict:
                z = 2 * y - x
                if 1 <= z <= n and z != x and z != y:
                    conflict = True
            # Case 3: z is middle, z = (x+y)/2
            if not conflict:
                if (x + y) % 2 == 0:
                    z = (x + y) // 2
                    if 1 <= z <= n and z != x and z != y:
                        conflict = True
            if conflict:
                conflicts[x] |= (1 << y)
                conflicts[y] |= (1 << x)

    # Helper: check if adding x to a set represented by bitmask is valid
    # A set S is represented as a bitmask where bit i is set if i in S.
    # x can be added if no element in S conflicts with x.
    # i.e., (S & conflicts[x]) == 0
    def can_add(x, S_mask):
        return (S_mask & conflicts[x]) == 0

    def get_sum_from_mask(S_mask):
        total = 0.0
        for i in range(1, n + 1):
            if (S_mask >> i) & 1:
                total += recip[i]
        return total

    def mask_to_list(S_mask):
        return [i for i in range(1, n + 1) if (S_mask >> i) & 1]

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask

    def greedy_fill(order, S_mask):
        """Fill greedily in the given order, adding elements that don't conflict."""
        for x in order:
            if not (S_mask >> x) & 1:
                if can_add(x, S_mask):
                    S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, time_limit):
        """Local search: try removals and swaps to improve the sum."""
        current_sum = get_sum_from_mask(S_mask)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removals: remove one element, then greedily refill
            elements = mask_to_list(S_mask)
            for rem in elements:
                if time.time() > time_limit:
                    break
                trial = S_mask & ~(1 << rem)
                # Greedy refill: try to add any element that was removed or could now be added
                # Actually, just do a full greedy refill from the trial set
                for x in range(1, n + 1):
                    if not (trial >> x) & 1:
                        if can_add(x, trial):
                            trial |= (1 << x)
                new_sum = get_sum_from_mask(trial)
                if new_sum > current_sum + 1e-12:
                    S_mask = trial
                    current_sum = new_sum
                    improved = True
                    break
        return S_mask, current_sum

    # Generate different orderings
    ascending = list(range(1, n + 1))
    descending = list(range(n, 0, -1))
    # Weighted: sort by reciprocal descending (same as ascending for 1/x)
    # Random orderings
    random.seed(42)

    best_mask = 0
    best_sum = 0.0

    # Try multiple initializations
    orders = [ascending, descending]
    for _ in range(20):
        order = list(range(1, n + 1))
        random.shuffle(order)
        orders.append(order)

    for order in orders:
        if time.time() > deadline - 10.0:
            break
        S_mask = greedy_fill(order, 0)
        S_mask, s = local_search(S_mask, deadline - 10.0)
        if s > best_sum:
            best_sum = s
            best_mask = S_mask

    # More local search iterations on the best
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        best_mask, best_sum = local_search(best_mask, deadline - 5.0)

    # Final polish: try all single removals and refills
    if time.time() < deadline - 2.0:
        elements = mask_to_list(best_mask)
        for rem in elements:
            if time.time() > deadline - 2.0:
                break
            trial = best_mask & ~(1 << rem)
            for x in range(1, n + 1):
                if not (trial >> x) & 1:
                    if can_add(x, trial):
                        trial |= (1 << x)
            new_sum = get_sum_from_mask(trial)
            if new_sum > best_sum + 1e-12:
                best_mask = trial
                best_sum = new_sum

    result = mask_to_list(best_mask)

    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")

    return result
