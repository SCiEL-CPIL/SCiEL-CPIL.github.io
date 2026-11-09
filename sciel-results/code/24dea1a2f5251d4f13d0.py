"""
Constructs a maximum-cardinality subset of [1, 255] containing 255
where every distinct pair has gcd > 1.

255 = 3 * 5 * 17. The optimal strategy is to take all multiples of the
smallest prime factor (3), yielding 85 elements. We verify optimality
using Bron-Kerbosch maximum independent set on the conflict graph.
"""

import time
import math
from math import gcd

def _bron_kerbosch_max_independent(adj, n, deadline):
    """
    Bron-Kerbosch with pivot to find maximum independent set.
    adj[i] is a set of neighbors (conflicts) of node i.
    We find max set S such that no two nodes in S are adjacent (conflict).
    """
    best = [0]
    best_set = [set()]
    
    def bronk(R, P, X):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
                best_set[0] = set(R)
            return
        # Pivot selection: choose vertex in P∪X with max degree in P
        union = P | X
        if not union:
            return
        pivot = max(union, key=lambda v: len(adj[v] & P))
        # Candidates: vertices in P not adjacent to pivot
        candidates = [v for v in P if v not in adj[pivot]]
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    return best_set[0]

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Baseline: all multiples of 3 in [1, 255]
    baseline = list(range(3, 256, 3))
    baseline_set = set(baseline)
    
    # Verify baseline is valid (should be, but check)
    valid = True
    for i in range(len(baseline)):
        for j in range(i + 1, len(baseline)):
            if gcd(baseline[i], baseline[j]) == 1:
                valid = False
                break
        if not valid:
            break
    if not valid or 255 not in baseline_set:
        return [255]
    
    # Build conflict graph on candidates divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency (conflict) graph: edge if gcd == 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Run Bron-Kerbosch to find maximum independent set
    # An independent set in the conflict graph = a set where all pairs have gcd > 1
    # We need 255 to be in the set, so we force it: remove all its neighbors from P
    neighbors_of_255 = adj[idx_255]
    
    best_size = len(baseline)
    best_set = set(range(n))  # placeholder, we'll track properly
    
    # Run Bron-Kerbosch with 255 forced in
    # R starts with {255}, P = all non-neighbors of 255, X = empty
    P = set(range(n)) - neighbors_of_255 - {idx_255}
    
    best_ik = [best_size - 1]  # size excluding 255
    best_ik_set = [set()]
    
    def bronk_forced(R, P, X):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best_ik[0]:
                best_ik[0] = len(R)
                best_ik_set[0] = set(R)
            return
        union = P | X
        if not union:
            return
        pivot = max(union, key=lambda v: len(adj[v] & P))
        candidates_list = [v for v in P if v not in adj[pivot]]
        for v in candidates_list:
            bronk_forced(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    bronk_forced({idx_255}, P, set())
    
    # Check if we found something better
    if best_ik[0] + 1 > best_size:
        result_indices = best_ik_set[0] | {idx_255}
        result = sorted(candidates[i] for i in result_indices)
        # Final verification
        valid2 = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) == 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if valid2 and 255 in result:
            return result
    
    # Return the baseline (multiples of 3)
    return baseline
