import time
import random
import numpy as np
from itertools import combinations

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def find_clique_for_primes(target_primes, clique_size, max_nodes_to_consider=300):
    """
    Attempts to find a clique of size 'clique_size' among numbers <= 10000
    such that all pairwise sums are smooth w.r.t 'target_primes'.
    """
    # 1. Precompute smoothness of sums
    is_smooth_sum = [False] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        if factors_cache[s].issubset(target_primes):
            is_smooth_sum[s] = True

    # 2. Calculate degree of each node (1..10000)
    # Using numpy for speed
    smooth_sums_np = np.array(is_smooth_sum, dtype=bool)
    
    degrees = np.zeros(10001, dtype=int)
    for i in range(1, 10001):
        start = i + 1
        end = i + 10001
        if end > MAX_SUM + 1:
            end = MAX_SUM + 1
        if start <= end:
            cnt = np.sum(smooth_sums_np[start:end])
            if 2*i < end and 2*i >= start:
                if smooth_sums_np[2*i]:
                    cnt -= 1
            degrees[i] = cnt

    # Get top candidates
    top_indices = np.argsort(degrees)[::-1][:max_nodes_to_consider]
    
    # Filter out nodes with degree < clique_size - 1
    valid_indices = [i for i in top_indices if degrees[i] >= clique_size - 1]
    
    if len(valid_indices) < clique_size:
        return None

    # Build subgraph
    sub_nodes = valid_indices
    n_sub = len(sub_nodes)
    
    # Adjacency matrix
    adj = [[False] * n_sub for _ in range(n_sub)]
    
    for u_idx, u in enumerate(sub_nodes):
        for v_idx in range(u_idx + 1, n_sub):
            v = sub_nodes[v_idx]
            if is_smooth_sum[u + v]:
                adj[u_idx][v_idx] = True
                adj[v_idx][u_idx] = True

    # Backtracking search for clique of size 'clique_size'
    def search_clique(chosen, candidates):
        if len(chosen) == clique_size:
            return chosen[:]
        if not candidates:
            return None
        
        for i in range(len(candidates)):
            node = candidates[i]
            new_chosen = chosen + [node]
            
            new_candidates = []
            for j in range(i + 1, len(candidates)):
                other = candidates[j]
                if adj[node][other]:
                    new_candidates.append(other)
            
            result = search_clique(new_chosen, new_candidates)
            if result:
                return result
        return None

    # Order candidates by degree in subgraph
    sub_degrees = [sum(adj[i]) for i in range(n_sub)]
    order = np.argsort(sub_degrees)[::-1]
    initial_candidates = [int(i) for i in order]
    
    try:
        result = search_clique([], initial_candidates)
        if result:
            return [sub_nodes[i] for i in result]
    except RecursionError:
        pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 999
    best_arr = list(range(1, 13)) # Fallback
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Strategy 1: Try to find a clique for small k (4, 5)
    
    # Try k=4: Primes {2, 3, 5, 7}
    if time.time() < deadline - 10:
        target_primes_4 = {2, 3, 5, 7}
        clique = find_clique_for_primes(target_primes_4, 12, max_nodes_to_consider=300)
        if clique:
            return clique
            
    # Try k=5: Primes {2, 3, 5, 7, 11}
    if time.time() < deadline - 10:
        target_primes_5 = {2, 3, 5, 7, 11}
        clique = find_clique_for_primes(target_primes_5, 12, max_nodes_to_consider=300)
        if clique:
            return clique

    # Strategy 2: Fallback to existing Local Search for k=6, 7, 8, 9
    
    for k in range(6, 10):
        if time.time() > deadline:
            break
            
        prime_indices = list(range(min(15, len(primes))))
        
        target_sets = set()
        
        target_sets.add(frozenset(primes[:k]))
        
        random.seed(42)
        for _ in range(5):
            selected_indices = random.sample(prime_indices, k)
            target_sets.add(frozenset(primes[i] for i in selected_indices))
        
        for target_primes in target_sets:
            if time.time() > deadline:
                break
            
            candidates = []
            for num in range(1, 10001):
                if is_smooth(num, target_primes):
                    candidates.append(num)
            
            if len(candidates) < 12:
                continue
            
            random.shuffle(candidates)
            current_arr = candidates[:12]
            
            def calc_cost(arr):
                bad_primes_set = set()
                n = len(arr)
                for i in range(n):
                    for j in range(i+1, n):
                        s = arr[i] + arr[j]
                        fs = factors_cache[s]
                        for p in fs:
                            if p not in target_primes:
                                bad_primes_set.add(p)
                return len(bad_primes_set)
            
            current_cost = calc_cost(current_arr)
            
            if current_cost == 0:
                best_score = len(target_primes)
                best_arr = current_arr[:]
                return best_arr
            
            temp = 100.0
            cooling = 0.95
            min_temp = 0.1
            
            while temp > min_temp and time.time() < deadline:
                idx = random.randint(0, 11)
                old_val = current_arr[idx]
                new_val = random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                is_dup = False
                for j in range(12):
                    if j != idx and current_arr[j] == new_val:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                current_arr[idx] = new_val
                new_cost = calc_cost(current_arr)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_cost = new_cost
                    if current_cost == 0:
                        best_score = len(target_primes)
                        best_arr = current_arr[:]
                        return best_arr
                else:
                    current_arr[idx] = old_val
                
                temp *= cooling
    
    if best_score == 999:
        best_arr = list(range(1, 13))
        best_score = count_distinct_primes(best_arr)
        
    random.seed(42)
    current_arr = best_arr[:]
    current_score = best_score
    
    for _ in range(500):
        if time.time() > deadline:
            break
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        if new_val in current_arr and new_val != old_val:
            continue
        current_arr[idx] = new_val
        score = count_distinct_primes(current_arr)
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
            
    return best_arr
