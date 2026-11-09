"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import random
import sys

sys.setrecursionlimit(10000)

def _precompute_smooth(max_val, primes):
    """Precompute a boolean array where is_smooth[i] is True if i is composed only of given primes."""
    is_smooth = [False] * (max_val + 1)
    is_smooth[1] = True
    for i in range(2, max_val + 1):
        n = i
        valid = True
        for p in primes:
            while n % p == 0:
                n //= p
        if n == 1:
            is_smooth[i] = True
    return is_smooth

def _get_prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _find_clique_size_12(candidates, is_smooth_sum):
    """
    Find a clique of size 12 in the graph defined by candidates and is_smooth_sum.
    Edge (i, j) exists if is_smooth_sum[candidates[i] + candidates[j]] is True.
    Returns list of candidate values or None.
    """
    n_cand = len(candidates)
    if n_cand < 12:
        return None
        
    # Build adjacency as bitmasks for efficiency? 
    # Or just sets. With ~300 nodes, sets are fine.
    adj = [set() for _ in range(n_cand)]
    
    for i in range(n_cand):
        ci = candidates[i]
        for j in range(i + 1, n_cand):
            cj = candidates[j]
            s = ci + cj
            if s <= len(is_smooth_sum) and is_smooth_sum[s]:
                adj[i].add(j)
                adj[j].add(i)
                
    # Sort indices by degree descending
    order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
    
    # Backtracking search
    def backtrack(clique, avail_set):
        if len(clique) == 12:
            return clique
        if not avail_set:
            return None
            
        # Optimization: if not enough candidates left, prune
        if len(avail_set) < 12 - len(clique):
            return None
            
        # Try candidates in order of degree (already sorted in order list, but avail_set is a set)
        # We iterate through the global order and pick those in avail_set
        for v in order:
            if v not in avail_set:
                continue
                
            # Check if v is connected to all in clique
            connected = True
            for u in clique:
                if v not in adj[u]:
                    connected = False
                    break
            if not connected:
                continue
                
            new_clique = clique + [v]
            # New available set: candidates in avail_set (excluding v) that are connected to v
            new_avail = set()
            for c in avail_set:
                if c == v:
                    continue
                if v in adj[c]:
                    new_avail.add(c)
                    
            result = backtrack(new_clique, new_avail)
            if result is not None:
                return result
                
    initial_avail = set(order)
    return backtrack([], initial_avail)

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of
    prime factors of all pairwise sums (distinct indices) is minimized.
    """
    max_val = 10000
    max_sum = 20000
    n_needed = 12
    
    # Strategy 1: 7-smooth sums (primes 2, 3, 5, 7)
    primes_7 = [2, 3, 5, 7]
    is_smooth_7 = _precompute_smooth(max_sum, primes_7)
    
    # Candidates: 7-smooth numbers <= 10000
    candidates_7 = [i for i in range(1, max_val + 1) if is_smooth_7[i]]
    
    if len(candidates_7) >= n_needed:
        clique_indices = _find_clique_size_12(candidates_7, is_smooth_7)
        if clique_indices is not None:
            # Return the actual values
            return [candidates_7[i] for i in clique_indices]
            
    # Strategy 2: 11-smooth sums (primes 2, 3, 5, 7, 11)
    primes_11 = [2, 3, 5, 7, 11]
    is_smooth_11 = _precompute_smooth(max_sum, primes_11)
    
    # Candidates: 11-smooth numbers <= 10000
    candidates_11 = [i for i in range(1, max_val + 1) if is_smooth_11[i]]
    
    if len(candidates_11) >= n_needed:
        clique_indices = _find_clique_size_12(candidates_11, is_smooth_11)
        if clique_indices is not None:
            return [candidates_11[i] for i in clique_indices]
            
    # Fallback: Greedy heuristic
    # Pick numbers that are multiples of a large power of 2 to ensure even sums
    # and try to keep sums smooth.
    # Let's just pick 12 distinct 7-smooth numbers.
    if len(candidates_7) >= n_needed:
        return candidates_7[:n_needed]
        
    if len(candidates_11) >= n_needed:
        return candidates_11[:n_needed]
        
    return list(range(1, 13))

# EVOLVE-BLOCK-END
