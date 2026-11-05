import time
import random
from math import gcd
from itertools import combinations

# Precompute GCD table
N = 128
gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_table[i][j] = g
        gcd_table[j][i] = g

# Precompute conflict structure: for each x, which pairs (a,b) in [1,128] form bad triples with x
# bad triple: gcd(a,b) == gcd(a,x) == gcd(b,x)
conflict_pairs = [set() for _ in range(N + 1)]
for x in range(1, N + 1):
    for a in range(1, x):
        for b in range(a + 1, x):
            g_ab = gcd_table[a][b]
            if gcd_table[a][x] == g_ab and gcd_table[b][x] == g_ab:
                conflict_pairs[x].add((a, b))

# For faster lookup: given a set S and element x, count how many bad triples are formed
def count_bad_triples(S, x):
    """Count bad triples that include x and two elements from S."""
    count = 0
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            g_ab = gcd_table[a][b]
            if gcd_table[a][x] == g_ab and gcd_table[b][x] == g_ab:
                count += 1
    return count

def total_bad_triples(S):
    """Count all bad triples in S."""
    S_list = list(S)
    n = len(S_list)
    count = 0
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            g_ab = gcd_table[a][b]
            for k in range(j + 1, n):
                c = S_list[k]
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    count += 1
    return count

def simulated_annealing(initial_set, deadline, seed=42):
    """
    Simulated annealing to maximize |S| while keeping bad_triples(S) = 0.
    State: subset S of [1,128]
    Move: add or remove a single element
    Energy: -|S| if valid (no bad triples), else -|S| + penalty * bad_triples
    """
    rng = random.Random(seed)
    S = set(initial_set)
    best_S = set(S)
    best_score = len(S) if total_bad_triples(S) == 0 else -len(S)
    
    # Start temperature
    T = 5.0
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Choose a random move
        x = rng.randint(1, N)
        
        if x in S:
            # Try removing x
            S_new = set(S)
            S_new.remove(x)
            bad_new = total_bad_triples(S_new)
            score_new = len(S_new) if bad_new == 0 else -len(S_new)
            score_old = len(S) if total_bad_triples(S) == 0 else -len(S)
            
            if score_new >= score_old or rng.random() < (10 ** ((score_new - score_old) / T)):
                S = S_new
                if len(S) > len(best_S) and total_bad_triples(S) == 0:
                    best_S = set(S)
                    best_score = len(S)
        else:
            # Try adding x
            S_new = set(S)
            S_new.add(x)
            bad_new = total_bad_triples(S_new)
            score_new = len(S_new) if bad_new == 0 else -len(S_new)
            score_old = len(S) if total_bad_triples(S) == 0 else -len(S)
            
            if score_new >= score_old or rng.random() < (10 ** ((score_new - score_old) / T)):
                S = S_new
                if len(S) > len(best_S) and total_bad_triples(S) == 0:
                    best_S = set(S)
                    best_score = len(S)
        
        T *= cooling
    
    return best_S

def greedy_construct(order):
    S = set()
    for x in order:
        # Check if adding x creates any bad triple
        S_list = list(S)
        n = len(S_list)
        conflict = False
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                g_ab = gcd_table[a][b]
                if gcd_table[a][x] == g_ab and gcd_table[b][x] == g_ab:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_set = set()
    
    # Phase 1: Greedy with multiple orderings
    rng = random.Random(42)
    
    # Try different greedy orderings
    orderings = []
    
    # Degree-based ordering
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
    
    orderings.append(sorted(range(1, N + 1), key=lambda x: (degrees[x], x)))
    orderings.append(list(range(1, N + 1)))
    
    # Random orderings
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    # Noisy degree orderings
    for _ in range(10):
        noisy = [(degrees[i] + rng.random(), i) for i in range(1, N + 1)]
        orderings.append([x[1] for x in sorted(noisy)])
    
    for order in orderings:
        if time.time() > deadline - 20.0:
            break
        S = greedy_construct(order)
        if len(S) > len(best_set):
            best_set = set(S)
    
    # Phase 2: Simulated annealing from multiple starting points
    if time.time() < deadline - 10.0:
        # Use the best greedy result as starting point
        S = simulated_annealing(best_set, deadline - 5.0, seed=123)
        if len(S) > len(best_set):
            best_set = S
        
        # Also try from random valid sets
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            # Build a small random valid set
            S_init = set()
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if len(S_init) >= 30:
                    break
                S_test = S_init | {x}
                if total_bad_triples(S_test) == 0:
                    S_init.add(x)
            
            S = simulated_annealing(S_init, deadline - 3.0, seed=456 + restart)
            if len(S) > len(best_set):
                best_set = S
    
    return list(best_set)
