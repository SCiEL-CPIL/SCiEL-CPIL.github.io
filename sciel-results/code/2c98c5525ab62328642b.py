import time
import random
import math
from math import gcd
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Fast validity check: can we add x to set S (represented as boolean array and list)?
    # Returns True if S U {x} is valid.
    def can_add(x, S_list, S_bool):
        # Group elements in S by their gcd with x
        # If for some g, there are two elements y1, y2 in S with gcd(x,y1)=g and gcd(x,y2)=g,
        # we must ensure gcd(y1, y2) != g.
        # Actually, if gcd(x,y1)=g and gcd(x,y2)=g, then gcd(y1, y2) could be g.
        # If gcd(y1, y2) == g, then (x, y1, y2) is a bad triple.
        
        # Optimization: 
        # For each y in S, g = gcd(x, y).
        # We need to check if there exists y' in S (y' != y) such that gcd(x, y') == g AND gcd(y, y') == g.
        
        # Let's group S elements by gcd(x, y).
        # groups[g] = list of y in S such that gcd(x, y) == g.
        # For each g, if len(groups[g]) >= 2, check all pairs (y1, y2) in groups[g].
        # If any pair has gcd(y1, y2) == g, return False.
        
        groups = defaultdict(list)
        for y in S_list:
            g = gcd_mat[x][y]
            groups[g].append(y)
            
        for g, members in groups.items():
            if len(members) >= 2:
                # Check pairs in members
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if gcd_mat[members[i]][members[j]] == g:
                            return False
        return True

    def is_valid(S_list):
        # Check if S_list is a valid set
        # This is O(|S|^3) in worst case, but with N=128, |S| is small (~30-50).
        # We can use the same grouping logic for each x in S, but it's expensive.
        # Instead, we maintain validity incrementally in the search.
        # For initial check or repair, we do a full check.
        for x in S_list:
            S_without_x = [y for y in S_list if y != x]
            if not can_add(x, S_without_x, [False]*(N+1)):
                return False
        return True

    def greedy_construct(ordering='random'):
        S = []
        S_bool = [False] * (N + 1)
        
        if ordering == 'random':
            candidates = numbers[:]
            random.shuffle(candidates)
        elif ordering == 'ascending':
            candidates = numbers[:]
        elif ordering == 'descending':
            candidates = list(reversed(numbers))
        elif ordering == 'prime_power_last':
            # Primes and prime powers are "dangerous" for gcd=1 triples.
            # Put them last to force the set to be built on composites first.
            def is_prime_power(n):
                if n < 2: return False
                p = None
                while n % 2 == 0:
                    if p is not None and p != 2: return False
                    p = 2
                    n //= 2
                if n > 1:
                    if p is not None: return False
                    p = n
                    n = 1
                return n == 1
            # Actually, simpler: just sort by number of divisors descending (composites first)
            candidates = sorted(numbers, key=lambda x: -len([d for d in range(1, int(x**0.5)+1) if x%d==0]))
        else:
            candidates = numbers[:]
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, S_bool):
                S.append(x)
                S_bool[x] = True
        
        return S

    def local_search(S, max_iters=2000):
        S = list(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Try to add any element
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(x, S, S_bool):
                    S.append(x)
                    S_bool[x] = True
                    current_size += 1
                    added = True
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    break
            
            if not added:
                no_improve += 1
                
                # 2. Try swaps to escape local optimum
                if S and not_in_S:
                    improved = False
                    # Try 1-swap: remove u, add v
                    # To be efficient, pick u that is "least useful" or random.
                    # Pick v that is "most promising".
                    
                    # Heuristic: Pick v that has high gcd with many elements in S?
                    # Or just random.
                    
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        u = random.choice(S)
                        v = random.choice(not_in_S)
                        
                        # Check if v can be added if u is removed
                        S_temp = [x for x in S if x != u]
                        if can_add(v, S_temp, S_bool):
                            S.remove(u)
                            S_bool[u] = False
                            S.append(v)
                            S_bool[v] = True
                            improved = True
                            # Size is same, but structure changed. 
                            # We don't track "potential" here, just accept.
                            # To avoid cycles, we could track history, but for now accept.
                            break
                    
                    if not improved:
                        # Try 2-swap: remove u1, u2, add v1, v2
                        if len(S) >= 2 and len(not_in_S) >= 2:
                            for _ in range(20):
                                if time.time() > deadline:
                                    break
                                u1, u2 = random.sample(S, 2)
                                v1, v2 = random.sample(not_in_S, 2)
                                
                                S_temp = [x for x in S if x not in (u1, u2)]
                                # Check if v1 can be added
                                if can_add(v1, S_temp, S_bool):
                                    S_temp.append(v1)
                                    # Check if v2 can be added
                                    if can_add(v2, S_temp, S_bool):
                                        # Perform swap
                                        S.remove(u1)
                                        S.remove(u2)
                                        S_bool[u1] = False
                                        S_bool[u2] = False
                                        S.append(v1)
                                        S.append(v2)
                                        S_bool[v1] = True
                                        S_bool[v2] = True
                                        improved = True
                                        break
                            
                            if not improved and no_improve > 50:
                                break
                        elif no_improve > 50:
                            break
                            
        return best_S, best_size

    best_S = []
    best_size = 0
    
    # Phase 1: Initial Greedy + LS
    orderings = ['random', 'ascending', 'descending', 'prime_power_last']
    for ordering in orderings:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 2: ILS Restart
    while time.time() < deadline - 2:
        # Perturb current best
        if best_S:
            S = list(best_S)
            # Remove 3-5 random elements
            k = random.randint(3, 5)
            if len(S) > k:
                to_remove = random.sample(S, k)
                for x in to_remove:
                    S.remove(x)
            
            S, size = local_search(S)
            
            if size > best_size:
                best_size = size
                best_S = S.copy()
        
        # Occasionally start fresh
        if random.random() < 0.2:
            S = greedy_construct('random')
            S, size = local_search(S)
            if size > best_size:
                best_size = size
                best_S = S.copy()

    return list(best_S)

# EVOLVE-BLOCK-END
