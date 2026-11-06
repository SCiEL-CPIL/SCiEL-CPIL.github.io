import time
import random
import math
from math import gcd

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

    # Precompute conflict pairs: for each x, store set of (a,b) pairs that form bad triple with x
    conflict_pairs = {i: set() for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    # Convert to lists for faster iteration
    conflict_lists = {i: list(conflict_pairs[i]) for i in range(1, N + 1)}
    
    # Fast validity check: can we add x to set S (represented as boolean array)?
    def can_add(x, S_bool):
        for (a, b) in conflict_lists[x]:
            if S_bool[a] and S_bool[b]:
                return False
        return True
    
    # Simulated Annealing
    def simulated_annealing(S_init, t_start=1.0, t_end=0.01, cooling=0.9995, max_iters=50000):
        S = set(S_init)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        
        T = t_start
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if T < t_end:
                break
            
            # Choose random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add a random element not in S
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                x = rng.choice(not_in_S)
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                # If can't add, stay (no penalty in SA for invalid states, we just don't make the move)
            
            elif move_type < 0.6:
                # Try to remove a random element from S
                if not S:
                    continue
                u = rng.choice(list(S))
                # Check if removing u is beneficial (it might allow adding others later)
                # In SA, we always accept removals (they don't make solution invalid)
                S.remove(u)
                S_bool[u] = False
                current_size -= 1
            
            else:
                # Swap: remove u from S, add v not in S
                if not S:
                    continue
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                u = rng.choice(list(S))
                v = rng.choice(not_in_S)
                
                # Check if v can be added after removing u
                # Temporarily remove u
                S_bool[u] = False
                S.remove(u)
                
                if can_add(v, S_bool):
                    S.add(v)
                    S_bool[v] = True
                    # Size stays same
                else:
                    # Revert: put u back
                    S.add(u)
                    S_bool[u] = True
                
                # If swap was valid, size unchanged; if not, we reverted
                # Track best
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            
            # Cooling
            T *= cooling
        
        return best_S, best_size
    
    # Greedy with different orderings
    def greedy_construct(ordering='random'):
        S = set()
        S_bool = [False] * (N + 1)
        
        if ordering == 'random':
            candidates = numbers[:]
            random.shuffle(candidates)
        elif ordering == 'ascending':
            candidates = numbers[:]
        elif ordering == 'descending':
            candidates = list(reversed(numbers))
        elif ordering == 'prime_first':
            # Primes first, then composites
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in numbers if is_prime(x)]
            composites = [x for x in numbers if not is_prime(x)]
            random.shuffle(primes)
            random.shuffle(composites)
            candidates = primes + composites
        elif ordering == 'large_first':
            # Larger numbers first (they have more divisors, might be more constrained)
            candidates = list(reversed(numbers))
            random.shuffle(candidates)
        else:
            candidates = numbers[:]
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True
        
        return S
    
    # Local search with swap-then-verify
    def local_search(S, max_iters=20000):
        S = set(S)
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
            
            # Try to add any element
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(x, S_bool):
                    S.add(x)
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
                
                # Try swaps
                if S and not_in_S:
                    improved = False
                    for _ in range(50):  # Try up to 50 random swaps
                        if time.time() > deadline:
                            break
                        u = random.choice(list(S))
                        v = random.choice(not_in_S)
                        
                        # Remove u, try to add v
                        S_bool[u] = False
                        S.remove(u)
                        
                        if can_add(v, S_bool):
                            S.add(v)
                            S_bool[v] = True
                            # Size unchanged, but we made a different swap
                            improved = True
                        else:
                            # Revert
                            S.add(u)
                            S_bool[u] = True
                    
                    if not improved:
                        # Try double swap: remove 2, add 2
                        if len(S) >= 2 and len(not_in_S) >= 2:
                            for _ in range(20):
                                if time.time() > deadline:
                                    break
                                u1, u2 = random.sample(list(S), 2)
                                v1, v2 = random.sample(not_in_S, 2)
                                
                                # Remove u1, u2
                                S_bool[u1] = False
                                S_bool[u2] = False
                                S.remove(u1)
                                S.remove(u2)
                                
                                # Try to add v1, v2
                                if can_add(v1, S_bool):
                                    S.add(v1)
                                    S_bool[v1] = True
                                    if can_add(v2, S_bool):
                                        S.add(v2)
                                        S_bool[v2] = True
                                        improved = True
                                    else:
                                        # Revert v1
                                        S.remove(v1)
                                        S_bool[v1] = False
                                # Revert u1, u2
                                S.add(u1)
                                S.add(u2)
                                S_bool[u1] = True
                                S_bool[u2] = True
                                
                                if improved:
                                    break
                        else:
                            # Revert if we removed elements
                            pass
                        
                        if not improved:
                            # Stuck, break out
                            if no_improve > 100:
                                break
        
        return best_S, best_size
    
    best_S = set()
    best_size = 0
    
    # Phase 1: Multiple greedy starts with different orderings
    orderings = ['random', 'ascending', 'descending', 'prime_first', 'large_first']
    for ordering in orderings:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S, max_iters=10000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Phase 2: Simulated annealing from best solution and random starts
    if best_S:
        S, size = simulated_annealing(best_S, max_iters=30000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Phase 3: More SA restarts
    while time.time() < deadline - 5:
        S = greedy_construct('random')
        S, size = simulated_annealing(S, t_start=0.5, t_end=0.01, cooling=0.999, max_iters=20000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Final local search polish
    if best_S and time.time() < deadline - 2:
        S, size = local_search(best_S, max_iters=20000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    return list(best_S)

# EVOLVE-BLOCK-END
