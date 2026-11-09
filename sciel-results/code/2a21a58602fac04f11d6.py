import time
import numpy as np

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

def find_clique_bitset(target_primes, clique_size, deadline):
    """
    Fast clique finder using bitset-based neighbor intersection.
    """
    # Build smooth sum indicator
    is_smooth_sum = np.zeros(MAX_SUM + 1, dtype=bool)
    for s in range(2, MAX_SUM + 1):
        if factors_cache[s].issubset(target_primes):
            is_smooth_sum[s] = True
    
    # Compute degree of each node (1..10000)
    degrees = np.zeros(10001, dtype=np.int32)
    smooth_sums = is_smooth_sum
    
    for i in range(1, 10001):
        lo = i + 1
        hi = min(i + 10000, MAX_SUM)
        if lo <= hi:
            cnt = np.sum(smooth_sums[lo:hi+1])
            two_i = 2 * i
            if lo <= two_i <= hi and smooth_sums[two_i]:
                cnt -= 1
            degrees[i] = cnt
    
    # Filter nodes with sufficient degree
    min_deg = clique_size - 1
    valid_mask = degrees >= min_deg
    valid_nodes = np.where(valid_mask)[0]
    valid_nodes = valid_nodes[valid_nodes >= 1]
    
    if len(valid_nodes) < clique_size:
        return None
    
    # Take top candidates by degree
    max_candidates = 800
    if len(valid_nodes) > max_candidates:
        top_idx = np.argsort(degrees[valid_nodes])[::-1][:max_candidates]
        valid_nodes = valid_nodes[top_idx]
    
    n_sub = len(valid_nodes)
    node_list = valid_nodes.tolist()
    
    # Build adjacency using bitsets for fast intersection
    adj_bits = [0] * n_sub
    for u_idx in range(n_sub):
        u = node_list[u_idx]
        mask = 0
        for v_idx in range(u_idx + 1, n_sub):
            v = node_list[v_idx]
            if is_smooth_sum[u + v]:
                mask |= (1 << v_idx)
        adj_bits[u_idx] = mask
    
    # Greedy + backtracking clique search using bitsets
    sub_deg = np.array([bin(adj_bits[i]).count('1') for i in range(n_sub)])
    order = np.argsort(sub_deg)[::-1]
    
    # Greedy initialization
    chosen = []
    available_mask = (1 << n_sub) - 1
    
    while len(chosen) < clique_size and available_mask:
        best = -1
        best_deg = -1
        avail = available_mask
        while avail:
            node = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            deg = bin(adj_bits[node] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best = node
        if best == -1:
            break
        chosen.append(best)
        available_mask &= adj_bits[best]
    
    if len(chosen) == clique_size:
        return [node_list[i] for i in chosen]
    
    # Backtracking with bitsets
    def backtrack(chosen_list, candidates_mask, depth):
        if len(chosen_list) == clique_size:
            return chosen_list[:]
        
        if not candidates_mask:
            return None
        
        if len(chosen_list) + bin(candidates_mask).count('1') < clique_size:
            return None
        
        cands = []
        avail = candidates_mask
        while avail:
            node = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            cands.append(node)
        
        cands.sort(key=lambda x: bin(adj_bits[x] & candidates_mask).count('1'), reverse=True)
        
        for node in cands:
            new_cands = candidates_mask & adj_bits[node]
            result = backtrack(chosen_list + [node], new_cands, depth + 1)
            if result is not None:
                return result
        return None
    
    initial_cands_mask = 0
    for i in order:
        initial_cands_mask |= (1 << i)
    
    try:
        result = backtrack([], initial_cands_mask, 0)
        if result is not None:
            return [node_list[i] for i in result]
    except RecursionError:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try increasing k starting from k=3 (5-smooth)
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        
        target_primes = set(primes[:k])
        
        clique = find_clique_bitset(target_primes, 12, deadline)
        if clique:
            return clique
    
    # Fallback: local search
    best_score = 999
    best_arr = list(range(1, 13))
    best_score = count_distinct_primes(best_arr)
    
    import random
    random.seed(42)
    
    current_arr = best_arr[:]
    current_score = best_score
    
    for _ in range(2000):
        if time.time() > deadline - 2:
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
