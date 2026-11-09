"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random
from collections import defaultdict

DEADLINE = time.time() + 85  # safety margin before 90s kill

def _precompute_gcds():
    """Precompute gcd for all pairs in [1,128]."""
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    return gcd_table

def _count_bad_triples(S, gcd_table):
    """Count number of bad triples in set S."""
    S = list(S)
    n = len(S)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                if g_ab == g_ac:
                    g_bc = gcd_table[b][c]
                    if g_bc == g_ab:
                        count += 1
    return count

def _conflict_count_with(S, x, gcd_table):
    """Count how many pairs (a,b) in S form a bad triple with x."""
    count = 0
    n = len(S)
    for i in range(n):
        a = S[i]
        g_ax = gcd_table[a][x]
        for j in range(i + 1, n):
            b = S[j]
            g_ab = gcd_table[a][b]
            g_bx = gcd_table[b][x]
            if g_ab == g_ax == g_bx:
                count += 1
    return count

def _simulate_annealing(gcd_table, rng, initial_set, max_iterations=50000):
    """Simulated annealing to maximize |S| subject to no bad triples."""
    S = set(initial_set)
    S_list = list(S)
    S_list.sort()
    
    # Precompute for fast conflict checking
    def is_valid_add(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        for a in S_set:
            g_ax = gcd_table[a][x]
            for b in S_set:
                if b <= a:
                    continue
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    return False
        return True
    
    # Fast conflict count using precomputed pair gcds
    def conflict_count(S_set, x):
        """Count bad triples that would be created by adding x."""
        cnt = 0
        S_list_local = list(S_set)
        n = len(S_list_local)
        for i in range(n):
            a = S_list_local[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, n):
                b = S_list_local[j]
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    cnt += 1
        return cnt
    
    current_size = len(S)
    best_S = list(S)
    best_size = current_size
    
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.9997
    
    all_numbers = set(range(1, 129))
    
    for it in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random element not in S
            candidates = all_numbers - S
            if not candidates:
                continue
            x = rng.choice(list(candidates))
            conf = conflict_count(S, x)
            if conf == 0:
                S.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = list(S)
            else:
                # Accept with probability based on temperature
                delta = -conf  # negative = worse
                if rng.random() < min(1.0, 2.71828 ** (delta / T)):
                    S.add(x)
                    current_size += 1
        
        elif move_type < 0.6:
            # Remove a random element from S
            if S:
                x = rng.choice(list(S))
                S.discard(x)
                current_size -= 1
        
        elif move_type < 0.8:
            # Swap: remove x from S, add y not in S
            if S:
                x = rng.choice(list(S))
                candidates = all_numbers - S
                if candidates:
                    y = rng.choice(list(candidates))
                    # Remove x, then try to add y
                    S.discard(x)
                    current_size -= 1
                    conf_y = conflict_count(S, y)
                    if conf_y == 0:
                        S.add(y)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_S = list(S)
                    else:
                        # Maybe accept the removal without adding
                        if rng.random() < min(1.0, 2.71828 ** (-conf_y / T)):
                            S.add(y)
                            current_size += 1
        
        else:
            # 2-swap: remove two from S, add two not in S
            if len(S) >= 2:
                S_list_tmp = list(S)
                x1, x2 = rng.sample(S_list_tmp, 2)
                candidates = all_numbers - S
                if len(candidates) >= 2:
                    y1, y2 = rng.sample(list(candidates), 2)
                    # Compute conflicts for both additions after removing x1, x2
                    S.discard(x1)
                    S.discard(x2)
                    current_size -= 2
                    conf1 = conflict_count(S, y1)
                    # Temporarily add y1 to check y2
                    S.add(y1)
                    conf2 = conflict_count(S, y2)
                    S.discard(y1)
                    
                    total_conf = conf1 + conf2
                    if total_conf == 0:
                        S.add(y1)
                        S.add(y2)
                        current_size += 2
                        if current_size > best_size:
                            best_size = current_size
                            best_S = list(S)
                    else:
                        if rng.random() < min(1.0, 2.71828 ** (-total_conf / T)):
                            S.add(y1)
                            S.add(y2)
                            current_size += 2
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally reheat if stuck
        if it % 5000 == 0 and it > 0 and T < 1.0:
            T = 3.0
    
    return best_S, best_size

def _greedy_seed(gcd_table, order):
    """Build initial set using greedy with given order."""
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        # Check if adding x creates any bad triple
        valid = True
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(x)
            S_set.add(x)
    return S

def construct():
    gcd_table = _precompute_gcds()
    rng = Random(12345)
    
    best_S = None
    best_size = 0
    
    # Generate diverse initial seeds
    seeds = []
    
    # Seed 1: ascending order
    seeds.append(_greedy_seed(gcd_table, list(range(1, 129))))
    
    # Seed 2: descending order
    seeds.append(_greedy_seed(gcd_table, list(range(128, 0, -1))))
    
    # Seed 3: random orders
    for _ in range(10):
        order = list(range(1, 129))
        rng.shuffle(order)
        seeds.append(_greedy_seed(gcd_table, order))
    
    # Seed 4: Priority by number of distinct prime factors (ascending - fewer first)
    def num_distinct_primes(x):
        cnt = 0
        n = x
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt
    
    order4 = sorted(range(1, 129), key=num_distinct_primes)
    seeds.append(_greedy_seed(gcd_table, order4))
    
    # Seed 5: Priority by number of distinct prime factors (descending)
    order5 = sorted(range(1, 129), key=lambda x: -num_distinct_primes(x))
    seeds.append(_greedy_seed(gcd_table, order5))
    
    # Seed 6: Odd numbers first
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(1, 129) if x % 2 == 0]
    seeds.append(_greedy_seed(gcd_table, odds + evens))
    
    # Seed 7: Numbers with large smallest prime factor first
    def spf(x):
        if x < 2:
            return 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                return d
            d += 1
        return x
    
    order7 = sorted(range(1, 129), key=lambda x: -spf(x))
    seeds.append(_greedy_seed(gcd_table, order7))
    
    # Find best seed
    for s in seeds:
        if len(s) > best_size:
            best_size = len(s)
            best_S = s
    
    # Run simulated annealing from multiple seeds
    for seed_idx, seed_set in enumerate(seeds[:8]):  # Use first 8 seeds
        if time.time() > DEADLINE - 5:
            break
        
        # Run SA with different parameters
        for sa_trial in range(3):
            if time.time() > DEADLINE - 3:
                break
            
            rng_sa = Random(seed_idx * 100 + sa_trial)
            sa_result, sa_size = _simulate_annealing(
                gcd_table, rng_sa, seed_set, 
                max_iterations=30000
            )
            if sa_size > best_size:
                best_size = sa_size
                best_S = sa_result
    
    # Final local search: try to add any remaining elements
    if best_S is not None:
        S_set = set(best_S)
        all_numbers = set(range(1, 129))
        remaining = all_numbers - S_set
        for x in sorted(remaining):
            if time.time() > DEADLINE - 2:
                break
            if _conflict_count_with(list(S_set), x, gcd_table) == 0:
                S_set.add(x)
        best_S = list(S_set)
        best_size = len(best_S)
    
    if best_S is None or len(best_S) == 0:
        return [1]
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
