"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill
    
    n_max = 180
    
    # Precompute forbidden triples {a, b, c} with a < b < c and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    forbidden_triples = []
    triple_count = [0] * (n_max + 1)
    
    for a in range(1, n_max + 1):
        a_sq = a * a
        # Iterate u up to sqrt(a_sq)
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    # b < c is guaranteed by u < v
                    # Distinctness: a < b, a < c. b != c.
                    forbidden_triples.append((a, b, c))
                    triple_count[a] += 1
                    triple_count[b] += 1
                    triple_count[c] += 1

    # Build conflict mapping: num -> list of (x, y) such that {x, y, num} is a forbidden triple
    conflicts = [[] for _ in range(n_max + 1)]
    for a, b, c in forbidden_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # Precompute for each num, the set of pairs that block it
    # For fast validity check, we track which elements are in S via a boolean array
    def is_valid_addition(S_bool, num):
        for x, y in conflicts[num]:
            if S_bool[x] and S_bool[y]:
                return False
        return True

    def run_greedy(order):
        S_bool = [False] * (n_max + 1)
        for num in order:
            if time.time() - start_time > deadline:
                break
            if is_valid_addition(S_bool, num):
                S_bool[num] = True
        return S_bool

    def S_bool_to_set(S_bool):
        return {i for i in range(1, n_max + 1) if S_bool[i]}

    def set_to_S_bool(S):
        S_bool = [False] * (n_max + 1)
        for x in S:
            S_bool[x] = True
        return S_bool

    def local_search(S, max_iter=200):
        """Fast local search: 1-add, 1-swap, 2-remove+2-add with precomputed candidates."""
        S_bool = set_to_S_bool(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() - start_time > deadline:
                break
            improved = False
            iter_count += 1
            
            # Try 1-addition
            for num in range(1, n_max + 1):
                if not S_bool[num] and is_valid_addition(S_bool, num):
                    S_bool[num] = True
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 1-swap: remove one, add one
            for remove in range(1, n_max + 1):
                if time.time() - start_time > deadline:
                    break
                if not S_bool[remove]:
                    continue
                S_bool[remove] = False
                for add in range(1, n_max + 1):
                    if not S_bool[add] and is_valid_addition(S_bool, add):
                        S_bool[add] = True
                        improved = True
                        break
                if not improved:
                    S_bool[remove] = True
            if improved:
                continue
                
            # Try 2-remove + 2-add (faster: limit candidate pairs)
            S_list = [i for i in range(1, n_max + 1) if S_bool[i]]
            for i in range(len(S_list)):
                if time.time() - start_time > deadline:
                    break
                for j in range(i + 1, len(S_list)):
                    r1, r2 = S_list[i], S_list[j]
                    S_bool[r1] = False
                    S_bool[r2] = False
                    found = False
                    # Try to add two elements
                    for a in range(1, n_max + 1):
                        if found:
                            break
                        if S_bool[a]:
                            continue
                        if not is_valid_addition(S_bool, a):
                            continue
                        S_bool[a] = True
                        for b in range(1, n_max + 1):
                            if S_bool[b]:
                                continue
                            if not is_valid_addition(S_bool, b):
                                continue
                            S_bool[b] = True
                            found = True
                            break
                        if not found:
                            S_bool[a] = False
                    if not found:
                        S_bool[r1] = True
                        S_bool[r2] = True
                    else:
                        improved = True
                        break
                if improved:
                    break
        return S_bool_to_set(S_bool)

    def polish(S):
        """Final pass: greedily add any remaining valid elements."""
        S_bool = set_to_S_bool(S)
        for num in range(1, n_max + 1):
            if not S_bool[num] and is_valid_addition(S_bool, num):
                S_bool[num] = True
        return S_bool_to_set(S_bool)

    # Generate different orders for greedy
    order_deg = list(range(1, n_max + 1))
    order_deg.sort(key=lambda x: triple_count[x])
    
    order_desc = list(range(n_max, 0, -1))
    order_asc = list(range(1, n_max + 1))
    
    # Additional heuristic order: prioritize numbers with fewer conflicts
    order_conflict = list(range(1, n_max + 1))
    order_conflict.sort(key=lambda x: len(conflicts[x]))
    
    best_set = set()
    best_size = 0
    
    # Run greedy with deterministic orders + local search
    for order in [order_deg, order_desc, order_asc, order_conflict]:
        S_bool = run_greedy(order)
        S = S_bool_to_set(S_bool)
        S = local_search(S, max_iter=150)
        S = polish(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Run greedy with random orders + local search (more restarts)
    for _ in range(150):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S_bool = run_greedy(order_rand)
        S = S_bool_to_set(S_bool)
        S = local_search(S, max_iter=100)
        S = polish(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Final local search on best set with more iterations
    best_set = local_search(best_set, max_iter=300)
    best_set = polish(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
