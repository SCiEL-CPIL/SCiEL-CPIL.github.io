import time
import random
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

def find_clique_in_smooth_graph(target_primes, k=12, deadline=None):
    """
    Find k distinct numbers <= 10000 such that all pairwise sums are smooth w.r.t target_primes.
    Uses bitset-based clique search for efficiency.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Generate candidates: numbers <= 10000 that are "compatible"
    # A number x is compatible if for many other smooth numbers y, x+y is smooth.
    # Simple heuristic: pick numbers that are themselves smooth or "near smooth".
    # Actually, let's just pick all numbers 1..10000 and build a graph where an edge exists
    # if x+y is smooth.
    
    # To make this feasible, we restrict the candidate pool.
    # If target_primes includes 2, then even sums are more likely smooth?
    # Let's try a pool of numbers that are smooth w.r.t target_primes first.
    # If that's not enough, expand.
    
    candidates = []
    for num in range(1, 10001):
        if is_smooth(num, target_primes):
            candidates.append(num)
    
    # If we don't have enough candidates, we might need to include non-smooth numbers,
    # but then x+y might not be smooth easily.
    # For now, stick to smooth candidates.
    if len(candidates) < k:
        return None
    
    # Build adjacency bitsets
    # n_cand = len(candidates)
    # adj[i] is a bitset (Python int) where bit j is 1 if candidates[i] + candidates[j] is smooth
    
    n_cand = len(candidates)
    adj = [0] * n_cand
    
    # This is O(N^2) which is 10^8 operations for N=10000. Too slow.
    # We need to prune the candidate pool first.
    # Let's limit candidates to the first, say, 200 smooth numbers?
    # Or random sample?
    # If we pick random 200 smooth numbers, the probability of a clique of size 12 depends on edge density.
    # Edge density: P(x+y is smooth).
    
    # Let's try a smaller pool to make clique search feasible.
    # Pick up to 500 candidates.
    if len(candidates) > 500:
        random.shuffle(candidates)
        candidates = candidates[:500]
    
    n_cand = len(candidates)
    adj = [0] * n_cand
    
    # Precompute smoothness of sums for this subset
    # To speed up, we can check is_smooth(candidates[i] + candidates[j])
    
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            if is_smooth(candidates[i] + candidates[j], target_primes):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        if deadline and time.time() > deadline:
            return None

    # Now find a clique of size k in this graph using Bron-Kerbosch with bitsets
    # Or a simple greedy/backtracking search since k is small (12) and graph is sparse/dense?
    
    # Let's use a recursive search with bitset intersection.
    # This is a standard technique for maximum clique.
    
    found_clique = []
    
    def bron_kerbosch(R, P, X):
        if deadline and time.time() > deadline:
            return False
        if len(R) >= k:
            found_clique.extend(R[:k])
            return True
        
        if P == 0 and X == 0:
            return False
            
        # Pivot selection
        # Pick pivot u in P ∪ X with max degree in P
        # For simplicity, just pick any u from P ∪ X if not empty, else return
        if P | X == 0:
            return False
            
        # We want to iterate over P \ N(u)
        # Let's just iterate over P
        # To be efficient, pick a pivot
        all_nodes = P | X
        if all_nodes == 0:
            return False
            
        # Find pivot with max degree in P
        pivot = 0
        max_deg = -1
        temp = all_nodes
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            temp ^= lsb
            
        candidates_to_try = P & ~(adj[pivot])
        
        while candidates_to_try:
            lsb = candidates_to_try & (-candidates_to_try)
            v = lsb.bit_length() - 1
            candidates_to_try ^= lsb
            
            R.append(v)
            if bron_kerbosch(R, P & adj[v], X & adj[v]):
                R.pop()
                return True
            R.pop()
            P ^= lsb
            X |= lsb
            
    # Initial call
    # P = all nodes, X = empty
    initial_P = (1 << n_cand) - 1
    if bron_kerbosch([], initial_P, 0):
        # Map indices back to values
        return [candidates[i] for i in found_clique]
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    best_score = 999
    best_arr = list(range(1, 13))  # fallback
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Strategy 1: Try to find a clique in the smooth graph for small sets of primes.
    # If we find 12 numbers such that all pairwise sums are smooth w.r.t {2,3,5,7},
    # then the answer is 4.
    
    # Try k=4 first: {2,3,5,7}
    # Then k=5: {2,3,5,7,11}
    # etc.
    
    for k in range(4, 9):
        if time.time() > deadline:
            break
            
        # Try a few combinations of primes of size k
        # We prefer smaller primes.
        
        # Option 1: First k primes
        target_primes = frozenset(primes[:k])
        
        clique = find_clique_in_smooth_graph(target_primes, 12, deadline)
        if clique:
            # Verify
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_arr = clique[:]
            # If score == k, we found an optimal solution for this k.
            # Since we are iterating k upwards, the first valid clique with score k is likely the global minimum
            # (assuming k=4 is possible). If k=4 works, we can return early.
            if score == k:
                return best_arr
                
        # Option 2: Random subsets of size k
        random.seed(42 + k)
        for _ in range(3):
            if time.time() > deadline:
                break
            selected = random.sample(primes[:15], k)
            target_primes = frozenset(selected)
            clique = find_clique_in_smooth_graph(target_primes, 12, deadline)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr

    # Strategy 2: Fallback to heuristic search if no perfect smooth clique found
    # Use the previous logic: powers of 2, multiples, etc.
    
    # Powers of 2
    powers_of_2 = [1 << i for i in range(12)]
    if max(powers_of_2) <= 10000:
        score = count_distinct_primes(powers_of_2)
        if score < best_score:
            best_score = score
            best_arr = powers_of_2[:]
            
    # Multiples
    for mult in range(1, 20):
        if time.time() > deadline:
            return best_arr
        for start_k in range(1, 30):
            if time.time() > deadline:
                return best_arr
            arr = [mult * (start_k + i) for i in range(12)]
            if len(set(arr)) < 12 or max(arr) > 10000:
                continue
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
                
    # Local search refinement on best_arr
    if best_score > 4:  # Only if we haven't found the theoretical minimum for small k
        random.seed(42)
        current_arr = best_arr[:]
        current_score = best_score
        
        # Simulated Annealing
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        
        while temp > min_temp and time.time() < deadline:
            idx = random.randint(0, 11)
            old_val = current_arr[idx]
            new_val = random.randint(1, 10000)
            
            if new_val in current_arr and new_val != old_val:
                continue
                
            current_arr[idx] = new_val
            score = count_distinct_primes(current_arr)
            
            delta = score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_score = score
                if score < best_score:
                    best_score = score
                    best_arr = current_arr[:]
            else:
                current_arr[idx] = old_val
                
            temp *= cooling

    return best_arr

if __name__ == "__main__":
    res = construct()
    print(res)
    print(count_distinct_primes(res))
