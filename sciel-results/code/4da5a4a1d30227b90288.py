import time
import random
import sys
from itertools import combinations

def count_distinct_primes_in_sums(arr):
    """Count distinct primes in the union of factors of all pairwise sums."""
    primes_set = set()
    n = len(arr)
    # Simple factorization for numbers up to 20000
    # Precompute smallest prime factor (SPF) up to 20000
    limit = 20000
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def factorize(x):
        factors = set()
        while x > 1:
            factors.add(spf[x])
            x //= spf[x]
        return factors
        
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes_set.update(factorize(s))
    return len(primes_set)

def generate_smooth_numbers(primes, limit):
    """Generate all numbers <= limit whose prime factors are in primes."""
    smooths = {1}
    # Simple recursive generation
    def gen(idx, current):
        if current > limit:
            return
        if idx == len(primes):
            smooths.add(current)
            return
        p = primes[idx]
        # Multiply by p^k
        val = current
        while val <= limit:
            smooths.add(val)
            gen(idx + 1, val)
            if val > limit // p:
                break
            val *= p
    gen(0, 1)
    return smooths

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Initial best: seed
    seed_set = list(range(1, 13))
    best_score = count_distinct_primes_in_sums(seed_set)
    best_set = seed_set
    
    # Try to find sets with fewer primes
    # Candidate prime sets to test, in order of increasing size
    prime_sets_to_try = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    max_val = 10000
    n_needed = 12
    
    for primes in prime_sets_to_try:
        if time.time() > deadline:
            break
            
        current_prime_count = len(primes)
        if current_prime_count >= best_score:
            # No point in trying if we can't beat the best score found so far
            # (Assuming we found a valid set for a smaller prime set)
            # But we haven't found one yet, so we must try.
            # Actually, if we found a set with best_score=2, we don't need to try 3 primes.
            # But we only update best_score if we find a valid set.
            pass

        # Generate smooth numbers for this prime set
        smooths = generate_smooth_numbers(primes, 2 * max_val)
        
        # Precompute neighbors for each number 1..max_val
        # neighbors[a] = list of b > a such that a+b is in smooths
        neighbors = [[] for _ in range(max_val + 1)]
        sorted_smooths = sorted(smooths)
        
        for a in range(1, max_val + 1):
            # b must be > a
            # a + b in smooths => b = s - a
            # s must be > 2a (since b > a)
            # s <= 2*max_val
            for s in sorted_smooths:
                if s <= 2 * a:
                    continue
                b = s - a
                if b > max_val:
                    break
                if b > a:
                    neighbors[a].append(b)
        
        # Find a clique of size n_needed in this graph
        # Using a greedy randomized search
        found = False
        attempts = 0
        max_attempts = 10000 # Adjust based on time
        
        while not found and attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Pick a random starting node with high degree
            # Prefer nodes with many neighbors
            candidates = [i for i in range(1, max_val + 1) if len(neighbors[i]) > 0]
            if not candidates:
                break
            
            # Weight by degree
            weights = [len(neighbors[i]) for i in candidates]
            # Simple random choice, or weighted
            # To speed up, just pick a random one from the top 10% degrees?
            # For now, random
            start_node = random.choice(candidates)
            
            current_set = [start_node]
            # Valid candidates must be neighbors of all in current_set
            # and greater than the last added (to avoid duplicates and maintain order)
            # Actually, we just need distinct elements.
            # Let's maintain a set of valid candidates.
            
            # Initial candidates: neighbors of start_node
            valid_candidates = set(neighbors[start_node])
            
            # We need to add n_needed - 1 more nodes
            for _ in range(n_needed - 1):
                if not valid_candidates:
                    break
                # Pick a node from valid_candidates
                # Heuristic: pick one that has many neighbors in valid_candidates
                best_candidate = None
                best_score_cand = -1
                # If valid_candidates is large, sampling is better
                if len(valid_candidates) > 100:
                    # Sample 100 candidates
                    sample = random.sample(list(valid_candidates), 100)
                else:
                    sample = list(valid_candidates)
                
                for c in sample:
                    # Count how many neighbors c has in valid_candidates
                    # Note: neighbors[c] contains b > c.
                    # We need intersection with valid_candidates.
                    # This is O(degree).
                    # Since degree is ~100, this is fast.
                    # But we need to be careful: valid_candidates might contain numbers < c.
                    # The graph is undirected. If x is in neighbors[y], then y is in neighbors[x]?
                    # No, neighbors[a] only stores b > a.
                    # So if c is in valid_candidates, and we pick c,
                    # the new valid_candidates must be neighbors of c (which are > c)
                    # INTERSECTED with current valid_candidates (which are > previous nodes).
                    # Wait, if we pick c, all future nodes must be > c?
                    # Not necessarily. They just need to be distinct and connected to all.
                    # However, to avoid permutations, we can enforce an increasing order.
                    # If we enforce increasing order, then when we pick c,
                    # all future nodes must be in neighbors[c] AND > c.
                    # And they must also be in the current valid_candidates set?
                    # Yes, because they must connect to ALL previous nodes.
                    # So new_valid_candidates = valid_candidates & set(neighbors[c])
                    # And we must filter out any node < c?
                    # If we enforce strict increase, yes.
                    
                    # Let's enforce strict increase to simplify.
                    # This means we are looking for a clique in a DAG? No, just ordered clique.
                    # This is valid because any clique can be ordered.
                    
                    # Count neighbors in valid_candidates that are > c
                    count = 0
                    for nb in neighbors[c]:
                        if nb in valid_candidates:
                            count += 1
                    if count > best_score_cand:
                        best_score_cand = count
                        best_candidate = c
                        
                if best_candidate is None:
                    break
                    
                current_set.append(best_candidate)
                # Update valid_candidates
                # New valid candidates must be:
                # 1. In current valid_candidates
                # 2. Neighbors of best_candidate
                # 3. Greater than best_candidate (due to ordering)
                
                new_valid = set()
                for nb in neighbors[best_candidate]:
                    if nb in valid_candidates:
                        new_valid.add(nb)
                valid_candidates = new_valid
                
            if len(current_set) == n_needed:
                # Verify
                if len(set(current_set)) == n_needed:
                    # Check if all sums are smooth
                    ok = True
                    for i in range(n_needed):
                        for j in range(i + 1, n_needed):
                            if current_set[i] + current_set[j] not in smooths:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Found a set with 'current_prime_count' primes
                        if current_prime_count < best_score:
                            best_score = current_prime_count
                            best_set = current_set
                        found = True
                        break

    return best_set

# Run construct once to ensure it works? No, evaluator calls it.
# But we should ensure no global state issues.

if __name__ == "__main__":
    res = construct()
    print(res)
    print(count_distinct_primes_in_sums(res))
