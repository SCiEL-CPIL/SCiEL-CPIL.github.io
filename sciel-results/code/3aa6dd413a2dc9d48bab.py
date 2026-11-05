import random
from itertools import combinations
from math import gcd

def construct():
    N = 128
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute number of divisors for sorting heuristic
    div_count = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if j % i == 0:
                div_count[j] += 1

    def try_greedy(order):
        S = []
        # To speed up, we can maintain a list of "blocked" pairs? 
        # But K is small, O(K^2) check is fine.
        for x in order:
            valid = True
            # Check against all pairs in S
            # If len(S) < 2, always valid
            if len(S) >= 2:
                # Optimization: if we have many elements, this loop is the bottleneck
                # But 128 is small.
                for a, b in combinations(S, 2):
                    if gcd_table[x][a] == gcd_table[x][b] == gcd_table[a][b]:
                        valid = False
                        break
            if valid:
                S.append(x)
        return S

    best_set = []
    
    # Deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -div_count[x]),
        sorted(range(1, N + 1), key=lambda x: div_count[x]),
        sorted(range(1, N + 1), key=lambda x: (-div_count[x], x)),
        sorted(range(1, N + 1), key=lambda x: (div_count[x], x)),
    ]
    
    for order in orders:
        S = try_greedy(order)
        if len(S) > len(best_set):
            best_set = S

    # Random restarts
    # 60s limit is generous. 
    # Each try_greedy is ~0.01s? 
    # 128 elements, avg K=50. 50^2 = 2500 checks. 128*2500 = 320k ops.
    # Python does ~10M ops/s. So 0.03s per run.
    # We can run 500 random restarts.
    
    for _ in range(500):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = try_greedy(order)
        if len(S) > len(best_set):
            best_set = S

    return best_set
