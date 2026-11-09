import time
import random
import math
from itertools import combinations

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s limit
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all possible sums (2 to 20000)
    # This is useful for the fallback greedy method
    sum_primes_cache = [None] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        sum_primes_cache[s] = get_primes(s)
        
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def find_clique(primes, time_limit=10.0):
        """
        Finds a clique of size target_size in the graph where nodes are 1..max_val
        and edges exist if sum is smooth w.r.t 'primes'.
        Returns the clique as a list or None.
        """
        search_start = time.time()
        
        # 1. Generate smooth sums
        smooth_sums = set()
        for s in range(2, 2 * max_val + 1):
            if is_smooth(s, primes):
                smooth_sums.add(s)
        
        if not smooth_sums:
            return None
            
        # 2. Build neighbor sets
        # neighbors[i] is a set of j such that i+j is in smooth_sums
        # We only care about j > i to maintain ordering in clique search, 
        # but for intersection, having full sets is easier, then we filter by > v.
        neighbors = [set() for _ in range(max_val + 1)]
        
        # Efficient construction:
        # For each s in smooth_sums:
        #   i ranges from max(1, s-max_val) to min(max_val, s//2)
        #   j = s - i
        #   Add j to neighbors[i] and i to neighbors[j]
        
        for s in smooth_sums:
            min_i = max(1, s - max_val)
            max_i = min(max_val, s // 2)
            for i in range(min_i, max_i + 1):
                j = s - i
                if i != j:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # 3. Order nodes by degree
        # We want to start with high degree nodes
        nodes = list(range(1, max_val + 1))
        nodes.sort(key=lambda x: len(neighbors[x]), reverse=True)
        
        # Map original value to index in sorted list? 
        # Or just use the values directly and keep a sorted list of candidates.
        
        def backtrack(clique, candidates):
            """
            clique: list of values
            candidates: set of values > clique[-1] (if clique not empty) 
                        that are connected to all nodes in clique.
            """
            if time.time() - search_start > time_limit:
                return None
                
            if len(clique) == target_size:
                return clique[:]
                
            if not candidates:
                return None
                
            # Heuristic: pick candidates with highest degree within the subgraph induced by candidates
            # To avoid sorting every time, we can just iterate. 
            # But for performance, let's try a few high-degree ones first.
            
            # Convert candidates to list for iteration
            # We can optimize by pre-sorting candidates by degree, but the induced subgraph degrees change.
            # For simplicity and speed in Python, we just iterate.
            
            # To improve, we can pick the candidate with the highest number of neighbors in `candidates`.
            # This is O(C^2). If C is large, this is slow.
            # Let's just iterate in the order of the global degree sort? 
            # We don't have that here. 
            # Let's just iterate.
            
            for v in list(candidates):
                # New candidates must be in candidates, > v, and connected to v
                new_candidates = set()
                # Optimization: only consider c in candidates if c > v and c in neighbors[v]
                for c in candidates:
                    if c > v and c in neighbors[v]:
                        new_candidates.add(c)
                        
                res = backtrack(clique + [v], new_candidates)
                if res:
                    return res
            return None

        # Initial candidates: all nodes. 
        # To enforce ordering, we don't need > clique[-1] for the first step, 
        # but in the recursion we do.
        # Actually, to avoid permutations, we should only pick v such that all previous elements are smaller.
        # My `backtrack` implementation handles this by requiring `c > v` in `new_candidates`.
        # However, the initial `candidates` set contains all nodes 1..10000.
        # If we pick v=10000 first, new_candidates will be empty (no c > 10000).
        # If we pick v=1 first, new_candidates will be {2..10000} intersect neighbors[1].
        # This works.
        
        # To speed up, we should probably not try ALL 10000 nodes as the first pick.
        # We can try the top K nodes by degree.
        top_nodes = nodes[:100]
        
        for start_node in top_nodes:
            if time.time() - search_start > time_limit:
                break
            # Initial candidates for the second element must be > start_node and connected to start_node
            init_candidates = set(n for n in neighbors[start_node] if n > start_node)
            res = backtrack([start_node], init_candidates)
            if res:
                return res
                
        return None

    best_set = None
    best_prime_count = float('inf')
    
    # Fallback: Generate a random set to ensure we always have a valid answer
    fallback_set = random.sample(range(1, max_val + 1), target_size)
    fallback_primes = evaluate(fallback_set)
    best_set = fallback_set
    best_prime_count = len(fallback_primes)
    
    # Try prime sets in order of increasing size (minimum primes first)
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try to find a clique for this prime set
        # Time limit per prime set: 10s
        res = find_clique(primes, time_limit=10.0)
        
        if res:
            # Verify
            if len(set(res)) == target_size and all(1 <= x <= max_val for x in res):
                primes_found = evaluate(res)
                if len(primes_found) < best_prime_count:
                    best_prime_count = len(primes_found)
                    best_set = res
                # If we found a solution with 2 primes, that's optimal
                if best_prime_count == 2:
                    return best_set
    
    # If clique search didn't find a better solution, use Greedy + Local Search
    # This part is inspired by the second solution
    
    def greedy_construct(start_val, seed):
        """Build a set greedily, choosing each new element to minimize new primes."""
        rng = random.Random(seed)
        current_set = [start_val]
        current_primes = set()
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_new_primes_count = float('inf')
            best_resulting_primes = None
            
            # Try a sample of candidates to keep it fast
            # Sample from 1..10000
            candidates = rng.sample(range(1, max_val + 1), 300)
            
            for candidate in candidates:
                if candidate in current_set:
                    continue
                
                # Calculate new primes introduced
                new_primes = set()
                for existing in current_set:
                    s = candidate + existing
                    if s <= 2 * max_val:
                        new_primes.update(sum_primes_cache[s])
                
                total_primes = current_primes | new_primes
                new_count = len(total_primes - current_primes)
                
                if new_count < best_new_primes_count:
                    best_new_primes_count = new_count
                    best_candidate = candidate
                    best_resulting_primes = total_primes
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                current_primes = best_resulting_primes
            else:
                # Fallback if no good candidate found in sample
                current_set.append(rng.randint(1, max_val))
                # Re-evaluate primes for the whole set to be safe
                current_primes = evaluate(current_set)
        
        return current_set, current_primes
    
    # Run greedy construction with multiple seeds
    if time.time() < deadline - 5:
        for seed in range(200):
            if time.time() > deadline - 5:
                break
            
            start_val = random.Random(seed).randint(1, max_val)
            result_set, result_primes = greedy_construct(start_val, seed)
            
            # Verify validity
            if len(set(result_set)) == target_size and all(1 <= x <= max_val for x in result_set):
                actual_primes = evaluate(result_set)
                if len(actual_primes) < best_prime_count:
                    best_prime_count = len(actual_primes)
                    best_set = result_set[:]
                    
                    # If we've found a very good solution, we can stop early
                    if best_prime_count <= 3:
                        break
    
    # Local search improvement on the best set found so far
    if time.time() < deadline - 2 and best_set is not None:
        current_set = best_set[:]
        current_primes = evaluate(current_set)
        current_count = len(current_primes)
        
        # Simple local search: swap one element with a random other
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, target_size - 1)
            old_val = current_set[idx]
            new_val = random.randint(1, max_val)
            
            if new_val == old_val:
                continue
            if new_val in current_set and current_set.index(new_val) != idx:
                continue
            
            new_set = current_set[:]
            new_set[idx] = new_val
            new_primes = evaluate(new_set)
            new_count = len(new_primes)
            
            if new_count < current_count:
                current_set = new_set
                current_primes = new_primes
                current_count = new_count
                
                if current_count < best_prime_count:
                    best_prime_count = current_count
                    best_set = current_set[:]
                    
    return best_set
