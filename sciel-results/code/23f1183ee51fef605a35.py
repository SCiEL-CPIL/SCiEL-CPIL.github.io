"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses structured polarity graph construction from PG(2, q) and local search with
efficient bitset-based branch-and-bound for independence number checks.
"""

import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_independent_set_k(adj_bits, n, k):
    """
    Branch and bound to check if there is an independent set of size k.
    Returns True if such a set exists.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    memo = {}
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        if size + _popcount(rem) < k:
            return False
        
        key = rem
        if key in memo:
            return memo[key]
        
        # Choose vertex with maximum degree in remaining subgraph
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            memo[key] = False
            return False
        
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        if search(new_rem, size + 1):
            memo[key] = True
            return True
        
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        if search(new_rem2, size):
            memo[key] = True
            return True
        
        memo[key] = False
        return False
    
    return search((1 << n) - 1, 0)

def _max_independent_set_size(adj_bits, n):
    """
    Find the maximum independent set size using branch and bound.
    """
    best = [0]
    
    def search(rem, size):
        if size >= best[0]:
            best[0] = size
        if rem == 0:
            return
        if size + _popcount(rem) <= best[0]:
            return
        
        # Choose vertex with maximum degree in remaining subgraph
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        search(new_rem, size + 1)
        
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        search(new_rem2, size)
    
    search((1 << n) - 1, 0)
    return best[0]

def _check_c4_free(adj_bits, n):
    """
    Check if graph is C4-free: no two vertices have more than 1 common neighbor.
    """
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if _popcount(common) > 1:
                return False
    return True

def _can_add_edge(adj_bits, u, v, n):
    """
    Check if adding edge (u, v) preserves C4-freeness.
    """
    # For each neighbor x of v (x != u), u and x must have no common neighbors
    for x in range(n):
        if x == u or x == v:
            continue
        if (adj_bits[v] >> x) & 1:
            # x is neighbor of v
            if adj_bits[u] & adj_bits[x]:
                return False
    # For each neighbor y of u (y != v), v and y must have no common neighbors
    for y in range(n):
        if x == u or x == v:
            continue
        if (adj_bits[u] >> y) & 1:
            # y is neighbor of u
            if adj_bits[v] & adj_bits[y]:
                return False
    return True

def _build_polarity_graph_pg2(q):
    """
    Build the polarity graph of PG(2, q).
    Points are 1D subspaces of GF(q^3).
    Two points are adjacent if they are perpendicular under a suitable polarity.
    
    For PG(2, q), there are (q^3 - 1)/(q - 1) = q^2 + q + 1 points.
    """
    # Generate all 1D subspaces (points) of GF(q^3)
    # We use a finite field representation
    
    if q == 2:
        # PG(2, 2) has 7 points, the Fano plane
        # Polarity graph is the Fano plane complement (which is also the Fano plane)
        # Actually, the polarity graph of PG(2,2) is the 7-vertex graph where
        # each vertex has degree 3 (the complement of the Fano plane lines)
        # Let's just use the standard construction
        n = 7
        edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,4), (4,5), (5,6), (6,0)]
        # This is the Paley graph of order 7
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits, n
    
    if q == 3:
        # PG(2, 3) has 13 points
        # The polarity graph has 13 vertices
        n = 13
        # Use a known construction
        # The polarity graph of PG(2,3) is a (13, 4, 1, 1) graph
        # Each vertex has degree 4
        adj_bits = [0] * n
        # Construct using the standard polarity
        # Points: [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1],
        #         [1,2,0], [2,1,0], [1,1,1], [1,2,1], [2,1,1], [1,1,2], [2,2,1]
        # We'll use a computational approach
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero coordinate 1
                    for i in range(3):
                        if (x, y, z)[i] != 0:
                            inv = pow((x, y, z)[i], q - 2, q)
                            pts = tuple((c * inv) % q for c in (x, y, z))
                            break
                    if pts not in points:
                        points.append(pts)
        
        n = len(points)
        adj_bits = [0] * n
        
        # Polarity: <(x1,y1,z1), (x2,y2,z2)> = x1*y2 + y1*x2 + z1*z2 (mod q)
        # Two points are adjacent if they are perpendicular
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dot = (p1[0] * p2[1] + p1[1] * p2[0] + p1[2] * p2[2]) % q
                if dot == 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        return adj_bits, n
    
    if q == 5:
        # PG(2, 5) has 31 points
        # The polarity graph has 31 vertices
        n = 31
        adj_bits = [0] * n
        
        # Generate points
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    for i in range(3):
                        if (x, y, z)[i] != 0:
                            inv = pow((x, y, z)[i], q - 2, q)
                            pts = tuple((c * inv) % q for c in (x, y, z))
                            break
                    if pts not in points:
                        points.append(pts)
        
        n = len(points)
        adj_bits = [0] * n
        
        # Polarity: <(x1,y1,z1), (x2,y2,z2)> = x1*y2 + y1*x2 + z1*z2 (mod q)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dot = (p1[0] * p2[1] + p1[1] * p2[0] + p1[2] * p2[2]) % q
                if dot == 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        return adj_bits, n
    
    # For other q, return None
    return None, 0

def _local_search_reduce_alpha(adj_bits, n, target_k, max_iters=100, time_limit=10.0):
    """
    Try to reduce alpha below target_k by adding edges.
    """
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find a large independent set
        remaining = (1 << n) - 1
        indep = []
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = _popcount(adj_bits[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            indep.append(best_v)
            remaining &= ~(adj_bits[best_v] | (1 << best_v))
        
        if len(indep) < target_k:
            return True
        
        # Try to add edges between vertices in indep
        changed = False
        for i in range(len(indep)):
            for j in range(i + 1, len(indep)):
                u, v = indep[i], indep[j]
                if not ((adj_bits[u] >> v) & 1):
                    if _can_add_edge(adj_bits, u, v, n):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        changed = True
                        break
            if changed:
                break
        
        if not changed:
            break
    
    return not _has_independent_set_k(adj_bits, n, target_k)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try polarity graphs of PG(2, q) for various q
    for q in [5, 3, 2]:
        if time.time() > deadline - 30:
            break
        
        adj_bits, n = _build_polarity_graph_pg2(q)
        if adj_bits is None:
            continue
        
        # Verify C4-free
        if not _check_c4_free(adj_bits, n):
            continue
        
        # Check independence number
        if not _has_independent_set_k(adj_bits, n, 11):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'edges': edges}
            # Try to extend with local search
            if time.time() < deadline - 20:
                adj_bits_copy = adj_bits[:]
                if _local_search_reduce_alpha(adj_bits_copy, n, 11, max_iters=50, time_limit=15.0):
                    if not _has_independent_set_k(adj_bits_copy, n, 11):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj_bits_copy[u] >> v) & 1:
                                    edges.append([u, v])
                        if n > best_n:
                            best_n = n
                            best_result = {'n': n, 'edges': edges}
    
    # Try randomized greedy construction for larger n
    rng = random.Random(42)
    
    for n in range(43, best_n, -1):
        if time.time() > deadline - 10:
            break
        
        restarts = 10 if n > 35 else 20
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            
            adj_bits = [0] * n
            
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline - 5:
                    break
                if _can_add_edge(adj_bits, u, v, n):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
            
            # Verify C4-free
            if not _check_c4_free(adj_bits, n):
                continue
            
            # Check independence number
            if not _has_independent_set_k(adj_bits, n, 11):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_bits[u] >> v) & 1:
                            edges.append([u, v])
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'edges': edges}
                    break
            
            # Try local search
            if time.time() < deadline - 10:
                adj_bits_copy = adj_bits[:]
                if _local_search_reduce_alpha(adj_bits_copy, n, 11, max_iters=30, time_limit=5.0):
                    if not _has_independent_set_k(adj_bits_copy, n, 11):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj_bits_copy[u] >> v) & 1:
                                    edges.append([u, v])
                        if n > best_n:
                            best_n = n
                            best_result = {'n': n, 'edges': edges}
                        break
    
    # Fallback: 10 disjoint triangles (n=30)
    if best_result is None:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_result = {'n': 30, 'edges': edges}
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
