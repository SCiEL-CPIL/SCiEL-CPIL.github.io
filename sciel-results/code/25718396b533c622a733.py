import time
import math

def _sieve_primes(n):
    """Return list of primes <= n using sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, prime in enumerate(sieve) if prime]

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n = 256
    
    primes = _sieve_primes(n)
    primes_set = set(primes)
    
    # Calculate max power of each prime that fits in 1..n
    powers = {}
    for p in primes:
        p_power = p
        while p_power * p <= n:
            p_power *= p
        powers[p] = p_power
    
    # Base set: for each prime p, include p_power
    # This gives a pairwise coprime set (each element is a prime power of distinct primes)
    base_set = set(powers.values())
    base_sum = sum(base_set)
    
    # Consider swaps: replace (P_p, P_q) with (p*q) for small p, q
    # Condition: p*q <= n, and p*q > P_p + P_q (net gain)
    # We need to find max-weight matching in a small graph
    
    # Candidate edges: (p, q, weight) where p < q, p*q <= n, p*q > P_p + P_q
    edges = []
    for i, p in enumerate(primes):
        P_p = powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            P_q = powers[q]
            prod = p * q
            if prod > n:
                break
            gain = prod - P_p - P_q
            if gain > 0:
                edges.append((p, q, gain))
    
    # Max weight matching via brute force over subsets of edges
    # Graph is small (few edges), so 2^E is feasible
    best_gain = 0
    best_edges = []
    
    num_edges = len(edges)
    if num_edges <= 20:
        # Brute force all subsets
        for mask in range(1 << num_edges):
            # Check if edges in this subset are pairwise disjoint
            used_nodes = set()
            total_gain = 0
            valid = True
            for e_idx in range(num_edges):
                if mask & (1 << e_idx):
                    p, q, gain = edges[e_idx]
                    if p in used_nodes or q in used_nodes:
                        valid = False
                        break
                    used_nodes.add(p)
                    used_nodes.add(q)
                    total_gain += gain
            if valid and total_gain > best_gain:
                best_gain = total_gain
                best_edges = [edges[e_idx] for e_idx in range(num_edges) if mask & (1 << e_idx)]
    else:
        # Greedy fallback if too many edges (unlikely for n=256)
        edges_sorted = sorted(edges, key=lambda e: e[2], reverse=True)
        used_nodes = set()
        for p, q, gain in edges_sorted:
            if p not in used_nodes and q not in used_nodes:
                best_edges.append((p, q, gain))
                best_gain += gain
                used_nodes.add(p)
                used_nodes.add(q)
    
    # Construct final set
    final_set = set([1])  # Include 1 (coprime to everything)
    
    # Mark which primes are used in composite edges
    used_in_edge = set()
    for p, q, gain in best_edges:
        used_in_edge.add(p)
        used_in_edge.add(q)
        final_set.add(p * q)
    
    # For primes not used in any edge, add their max power
    for p in primes:
        if p not in used_in_edge:
            final_set.add(powers[p])
    
    # Verify pairwise coprimality
    result = list(final_set)
    result.sort()
    
    # Final validation: all distinct, in range, pairwise coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to just prime powers + 1
                result = list(powers.values()) + [1]
                result.sort()
                break
        else:
            continue
        break
    
    return result
