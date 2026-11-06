import time
import random
from bisect import bisect_left, bisect_right

def construct():
    n = 256
    start_time = time.time()
    # 90s limit, keep 5s buffer for overhead/safety
    deadline = start_time + 85.0

    # Precompute for each possible a in [1, n], the set of forbidden sums (b+c)
    # that are divisible by a. Since b+c ranges from 2 to 2n, we can precompute
    # a boolean table: forbid[a][s] = True if s % a == 0.
    # Actually, since n=256, max sum is 512.
    # We'll precompute for each a, the set of residues or just use modulo.
    # Better: precompute for each a, the list of multiples of a up to 2n.
    # But checking (b+c) % a == 0 is fast enough.
    # Let's precompute a 2D list: div[a][s] = (s % a == 0) for a in 1..n, s in 2..2n.
    # This is 256 * 512 = 131072 entries. Very small.
    
    max_sum = 2 * n
    div = [[False] * (max_sum + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for multiple in range(a, max_sum + 1, a):
            div[a][multiple] = True
    
    def is_valid_addition(x, S_list):
        """
        Check if adding x to the sorted list S_list maintains the property:
        For every a < b <= c in the resulting set, a does NOT divide b+c.
        S_list contains elements of S, sorted ascending.
        x is not in S_list.
        """
        s_len = len(S_list)
        
        # Case 1: x is 'a'
        # Triple: (x, b, c) with x < b <= c.
        # b, c are from S_list.
        # Condition: x does NOT divide (b+c).
        start_b = bisect_right(S_list, x)
        
        if start_b < s_len:
            # b and c are in S_list[start_b : s_len]
            for i in range(start_b, s_len):
                b = S_list[i]
                for j in range(i, s_len):
                    c = S_list[j]
                    if div[x][b + c]:
                        return False
        
        # Case 2: x is 'b'
        # Triple: (a, x, c) with a < x <= c.
        # a, c from S_list.
        # Condition: a does NOT divide (x+c).
        # a < x, c >= x.
        # Since x is not in S_list, c > x.
        
        end_a = bisect_left(S_list, x)
        # a in S_list[0 : end_a]
        # c in S_list[start_b : s_len] (where start_b is first > x, same as first >= x since x not in S)
        
        # Check (a, x, x) -> sum = 2x
        for i in range(end_a):
            a = S_list[i]
            if div[a][2 * x]:
                return False
        
        # Check (a, x, c) with c > x
        for i in range(end_a):
            a = S_list[i]
            for j in range(start_b, s_len):
                c = S_list[j]
                if div[a][x + c]:
                    return False
        
        # Case 3: x is 'c'
        # Triple: (a, b, x) with a < b <= x.
        # a, b from S_list.
        # Condition: a does NOT divide (b+x).
        # b < x (since x not in S, b <= x implies b < x).
        # a < b.
        
        # b in S_list[0 : start_b] (elements < x)
        for j in range(start_b):
            b = S_list[j]
            # a in S_list[0 : j]
            for i in range(j):
                a = S_list[i]
                if div[a][b + x]:
                    return False
        
        return True

    def greedy(order, start_S=None):
        S = set()
        S_list = []
        if start_S:
            S = set(start_S)
            S_list = sorted(S)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if is_valid_addition(x, S_list):
                S.add(x)
                # Insert into sorted list
                pos = bisect_left(S_list, x)
                S_list.insert(pos, x)
        return S, S_list

    best_S = set()
    best_S_list = []
    
    # Strategy 1: Descending order
    # Large numbers are less likely to be 'a' in a violating triple because
    # there are fewer numbers larger than them.
    S_desc, S_list_desc = greedy(range(n, 0, -1))
    if len(S_desc) > len(best_S):
        best_S = S_desc
        best_S_list = S_list_desc

    # Strategy 2: Ascending order
    if time.time() < deadline:
        S_asc, S_list_asc = greedy(range(1, n + 1))
        if len(S_asc) > len(best_S):
            best_S = S_asc
            best_S_list = S_list_asc

    # Strategy 3: Randomized Greedy Restarts
    rng = random.Random(42)
    
    # We will run random restarts until deadline
    # To make this faster, we can also try "biased" random orders
    # e.g., shuffle but keep large numbers more likely to be picked early?
    # Or just simple shuffle.
    
    # Let's also try a few specific heuristic orders:
    # 1. Primes
    # 2. Numbers with large smallest prime factor
    # 3. Random
    
    # Precompute some heuristic orders
    # Large numbers first (already done)
    # Small numbers first (already done)
    
    # Try a few fixed heuristic orders
    # Order by number of divisors (ascending)? Numbers with fewer divisors are safer as 'a'
    # Order by number of divisors (descending)?
    
    # For now, stick to random restarts as they explore the space well.
    
    restart_count = 0
    last_improvement_time = start_time
    
    while time.time() < deadline - 5.0:
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        # Optional: bias towards keeping large numbers?
        # rng.shuffle is uniform.
        
        S_rand, S_list_rand = greedy(order)
        
        if len(S_rand) > len(best_S):
            best_S = S_rand
            best_S_list = S_list_rand
            last_improvement_time = time.time()
        
        restart_count += 1
        
        # Stop random restarts if we've had no improvement for a long time
        # and we have a good set?
        # For now, just keep going until deadline.
        
        # If best_S is very large (e.g., > 100), maybe switch to local search?
        # But local search is expensive if done naively.
        
    # Strategy 4: Local Search / Polish
    # Try to add any element from [1, n] not in best_S to best_S.
    # This is a simple hill-climbing step.
    # We can also try swapping?
    # Swap: remove x, add y?
    # This is complex. Let's stick to addition first.
    
    if time.time() < deadline - 1.0:
        current_S = set(best_S)
        current_S_list = sorted(current_S)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for x in range(1, n + 1):
                if x in current_S:
                    continue
                if is_valid_addition(x, current_S_list):
                    current_S.add(x)
                    pos = bisect_left(current_S_list, x)
                    current_S_list.insert(pos, x)
                    improved = True
        
        if len(current_S) > len(best_S):
            best_S = current_S
            best_S_list = current_S_list

    # Strategy 5: Swap-based Local Search
    # Try to replace an element x in S with an element y not in S
    # such that the new set is valid and size is maintained (or improved? no, swap keeps size).
    # This might allow us to "unblock" other additions later.
    # Given time constraints, this might be too slow.
    # Let's see if we have time.
    
    if time.time() < deadline - 0.5 and len(best_S) < n:
        # This is a heuristic. We'll try a few swaps.
        # Pick a random x in S, a random y not in S.
        # Check if S - {x} + {y} is valid.
        # If valid, and if it allows adding more elements later, it's good.
        # But checking "allows adding more" is the whole search.
        # So just check if S - {x} + {y} is valid.
        # If it is, keep it? No, size is same.
        # Maybe it allows adding z which was previously blocked by x?
        
        # This is getting complicated. Let's skip swaps for now
        # unless we have a very specific insight.
        
        # Instead, let's just do a final pass of addition with a different order?
        # Or just return what we have.
        
        pass

    if not best_S:
        return [256]
    
    return list(best_S)

# Quick sanity check / self-test if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    sorted_S = sorted(S)
    for i, a in enumerate(sorted_S):
        for j, b in enumerate(sorted_S):
            for k, c in enumerate(sorted_S):
                if a < b <= c:
                    if (b + c) % a == 0:
                        valid = False
                        print(f"Invalid: {a}, {b}, {c}")
    print(f"Valid: {valid}")
