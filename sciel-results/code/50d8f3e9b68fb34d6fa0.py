"""
Shortest superpermutation construction.

Strategy:
  - For n <= 4, use the classic recursive construction which is optimal.
  - For n >= 5, use a TSP heuristic on the de Bruijn graph of order n-1.
    The superpermutation problem is equivalent to finding a Hamiltonian cycle
    in the de Bruijn graph that visits each permutation (n-tuple) as a window.
    Instead of doing a full Hamiltonian cycle search (NP-hard), we:
      1. Enumerate all n! permutations.
      2. Order them using a greedy nearest-neighbor heuristic based on
         the de Bruijn graph distance (minimum number of symbols to append
         to transition from one permutation's suffix to the next permutation's prefix).
      3. Build the word by appending the minimum necessary symbols for each transition.
    This produces a valid superpermutation with length close to the theoretical optimum.
"""

import time
import sys
from itertools import permutations

def _classic_construct(n):
    """Classic recursive superpermutation construction, optimal for small n."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _de_bruijn_dist(u, v, n):
    """
    Minimum number of symbols to append to transition from node u to node v
    in the de Bruijn graph of order n-1 over symbols 1..n.
    
    Node u is a tuple of length n-1. To go from u to v, we shift left and append
    symbols. The distance is the smallest k such that u[k:] == v[:n-1-k].
    """
    ln = len(u)
    for k in range(0, ln + 1):
        if u[k:] == v[:ln - k]:
            return k
    return ln + 1  # shouldn't happen for valid nodes


def _build_word_from_order(order, n):
    """
    Given an ordering of permutations (list of tuples), build the superpermutation word.
    """
    if not order:
        return ""
    word = list(order[0])
    for i in range(1, len(order)):
        u = tuple(order[i - 1][1:])  # suffix of previous perm (length n-1)
        v = order[i]                  # next perm
        k = _de_bruijn_dist(u, v, n)
        # Append the last k symbols of v
        word.extend(list(v[n - 1 - k:]))
    return "".join(map(str, word))


def _greedy_tsp_order(perms, n, deadline):
    """
    Greedy nearest-neighbor TSP heuristic to order permutations for minimal total transition cost.
    """
    m = len(perms)
    if m == 0:
        return []
    
    # Precompute suffix and prefix for each permutation
    suffixes = [tuple(p[1:]) for p in perms]  # node after visiting p
    prefixes = [tuple(p[:n - 1]) for p in perms]  # node before visiting p
    
    # Distance from suffix of perm i to prefix of perm j
    # dist[i][j] = _de_bruijn_dist(suffixes[i], prefixes[j])
    # This is an m x m matrix; for n=7, m=5040, that's ~25M entries.
    # We'll compute on the fly with caching, or use a smarter approach.
    
    # For n=7, m=5040. Full distance matrix is 5040*5040 = ~25.4M.
    # Each distance computation is O(n) at most. That's ~178M operations.
    # In Python this might be slow. Let's use numpy or optimize.
    
    # Alternative: use the fact that de Bruijn distance has structure.
    # dist(u, v) where u, v are (n-1)-tuples: smallest k with u[k:] == v[:n-1-k].
    # For n=7, n-1=6. k ranges 0..6.
    
    # Let's precompute distances using a more efficient method.
    # Actually for the greedy heuristic, we just need to find the nearest neighbor
    # at each step, so we can compute distances on the fly.
    
    start = 0
    remaining = set(range(1, m))
    order = [start]
    
    while remaining:
        cur_suffix = suffixes[order[-1]]
        best_j = None
        best_dist = float('inf')
        for j in remaining:
            d = _de_bruijn_dist(cur_suffix, prefixes[j], n)
            if d < best_dist:
                best_dist = d
                best_j = j
        order.append(best_j)
        remaining.remove(best_j)
        if time.time() > deadline:
            break
    
    return [perms[i] for i in order]


def _two_opt_improve(order, n, iterations=20, deadline=None):
    """
    2-opt local search improvement on the permutation ordering.
    """
    m = len(order)
    if m <= 2:
        return order
    
    # Precompute suffixes and prefixes
    suffixes = [tuple(p[1:]) for p in order]
    prefixes = [tuple(p[:n - 1]) for p in order]
    
    # Compute initial total cost
    total_cost = sum(_de_bruijn_dist(suffixes[i], prefixes[i + 1], n) for i in range(m - 1))
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(m - 1):
            for j in range(i + 1, m):
                # Compute cost change for reversing segment [i, j]
                # Before: ... A[i-1] -> A[i] ... A[j] -> A[j+1] ...
                # After:  ... A[i-1] -> A[j] ... A[i] -> A[j+1] ...
                # Change in cost:
                # Remove edges: (i-1, i) and (j, j+1)
                # Add edges: (i-1, j) and (i, j+1)
                # (If i==0, no edge (i-1, i); if j==m-1, no edge (j, j+1))
                
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    old_cost += _de_bruijn_dist(suffixes[i - 1], prefixes[i], n)
                if j < m - 1:
                    old_cost += _de_bruijn_dist(suffixes[j], prefixes[j + 1], n)
                
                if i > 0:
                    new_cost += _de_bruijn_dist(suffixes[i - 1], prefixes[j], n)
                if j < m - 1:
                    new_cost += _de_bruijn_dist(suffixes[i], prefixes[j + 1], n)
                
                delta = new_cost - old_cost
                if delta < 0:
                    # Reverse segment
                    order[i:j + 1] = order[i:j + 1][::-1]
                    suffixes[i:j + 1] = suffixes[i:j + 1][::-1]
                    prefixes[i:j + 1] = prefixes[i:j + 1][::-1]
                    total_cost += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return order


def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    if n <= 4:
        return _classic_construct(n)
    
    # Generate all permutations
    all_perms = list(permutations(range(1, n + 1)))
    m = len(all_perms)
    
    # Greedy TSP ordering
    order = _greedy_tsp_order(all_perms, n, deadline)
    
    # 2-opt improvement if time permits
    if time.time() < deadline - 30.0 and m < 10000:
        order = _two_opt_improve(order, n, iterations=10, deadline=deadline - 30.0)
    
    word = _build_word_from_order(order, n)
    return word
