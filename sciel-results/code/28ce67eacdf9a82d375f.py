"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_bitset(adj_bits, n):
    """Check if adjacency bitsets have a triangle."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                common = adj_bits[i] & adj_bits[j]
                # Remove i and j themselves
                common &= ~(1 << i)
                common &= ~(1 << j)
                if common:
                    return True
    return False


def _max_independent_set_size_bitset(adj_bits, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]

    def bnb(candidates, current_size):
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') <= best[0]:
            return

        # Pick vertex with highest degree in candidates for better pruning
        v = -1
        max_deg = -1
        c = candidates
        while c:
            lb = c & (-c)
            v_idx = lb.bit_length() - 1
            deg = bin(adj_bits[v_idx] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                v = v_idx
            c ^= lb

        if v == -1:
            if current_size > best[0]:
                best[0] = current_size
            return

        rest = candidates & ~(1 << v)

        # Branch: include v -> exclude all neighbors of v
        non_neighbors = rest & ~adj_bits[v]
        bnb(non_neighbors, current_size + 1)

        # Branch: exclude v
        bnb(rest, current_size)

    bnb((1 << n) - 1, 0)
    return best[0]


def _build_circulant(n, connection_set):
    """Build circulant graph: vertex i connected to i+d mod n for d in connection_set."""
    adj_bits = [0] * n
    for i in range(n):
        for d in connection_set:
            j = (i + d) % n
            if j != i:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    return adj_bits


def _check_circulant(n, connection_set):
    """Check if circulant graph with given connection set is triangle-free and alpha < 10."""
    adj_bits = _build_circulant(n, connection_set)

    # Check triangle-free
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                common = adj_bits[i] & adj_bits[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                if common:
                    return False, 0

    alpha = _max_independent_set_size_bitset(adj_bits, n)
    if alpha >= 10:
        return False, alpha

    return True, alpha


def _search_circulants(n):
    """Search over all circulant graphs on n vertices."""
    # For circulant graphs, connection set determines the graph.
    # We need to find a connection set that is triangle-free and has alpha < 10.
    
    # Generate candidate connection sets
    # A circulant graph is determined by which distances are edges.
    # For n vertices, possible distances are 1, 2, ..., n//2.
    
    max_d = n // 2
    best_result = None
    
    # Try all subsets of {1, 2, ..., max_d}
    # This is 2^(n/2) possibilities - too many for large n.
    # Instead, use a smarter approach: start with sparse sets and add edges.
    
    # First, try single-distance circulants
    for d in range(1, max_d + 1):
        if _remaining_time() < 2:
            break
        valid, alpha = _check_circulant(n, [d])
        if valid:
            best_result = ([d], alpha)
            return best_result
    
    # Try two-distance circulants
    for d1 in range(1, max_d + 1):
        for d2 in range(d1 + 1, max_d + 1):
            if _remaining_time() < 1:
                return best_result
            valid, alpha = _check_circulant(n, [d1, d2])
            if valid:
                best_result = ([d1, d2], alpha)
                return best_result
    
    # Try three-distance circulants
    for d1 in range(1, max_d + 1):
        for d2 in range(d1 + 1, max_d + 1):
            for d3 in range(d2 + 1, max_d + 1):
                if _remaining_time() < 1:
                    return best_result
                valid, alpha = _check_circulant(n, [d1, d2, d3])
                if valid:
                    best_result = ([d1, d2, d3], alpha)
                    return best_result
    
    return best_result


def _local_search_circulant(n, connection_set):
    """Try to improve circulant by adding/removing edges while maintaining triangle-free."""
    adj_bits = _build_circulant(n, connection_set)
    alpha = _max_independent_set_size_bitset(adj_bits, n)
    
    if alpha < 10:
        return adj_bits, alpha
    
    # Try adding individual edges that don't create triangles
    improved = False
    for i in range(n):
        for j in range(i + 1, n):
            if _remaining_time() < 2:
                break
            if (adj_bits[i] >> j) & 1:
                continue
            
            # Check if adding (i,j) creates a triangle
            common = adj_bits[i] & adj_bits[j]
            common &= ~(1 << i)
            common &= ~(1 << j)
            if common:
                continue
            
            # Try adding edge
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            
            new_alpha = _max_independent_set_size_bitset(adj_bits, n)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                if alpha < 10:
                    return adj_bits, alpha
            else:
                adj_bits[i] &= ~(1 << j)
                adj_bits[j] &= ~(1 << i)
    
    return adj_bits, alpha


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj_bits = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try circulant graphs from high n to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 5:
            break
        
        result = _search_circulants(n_target)
        
        if result is not None:
            conn_set, alpha = result
            adj_bits = _build_circulant(n_target, conn_set)
            
            # Verify
            if not _has_triangle_bitset(adj_bits, n_target):
                best_adj_bits = adj_bits
                best_n = n_target
                best_alpha = alpha
                break
        
        # Also try local search from a random starting point
        if _remaining_time() > 10:
            rng = random.Random(n_target)
            # Start with a sparse random graph
            adj_bits = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if rng.random() < 0.1:
                        # Check triangle-free
                        common = adj_bits[i] & adj_bits[j]
                        common &= ~(1 << i)
                        common &= ~(1 << j)
                        if not common:
                            adj_bits[i] |= (1 << j)
                            adj_bits[j] |= (1 << i)
            
            if not _has_triangle_bitset(adj_bits, n_target):
                adj_bits, alpha = _local_search_circulant(n_target, [])
                if alpha < 10:
                    best_adj_bits = adj_bits
                    best_n = n_target
                    best_alpha = alpha
                    break
    
    # Fallback: use a known small construction
    if best_adj_bits is None:
        n = 26
        adj_bits = [0] * n
        # Andrásfai graph And(9): 27 vertices, but let's use 26
        # Simpler: use circulant with distance 1
        for i in range(n):
            for d in [1, 2]:
                j = (i + d) % n
                if j != i:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        # Verify and fix if needed
        if _has_triangle_bitset(adj_bits, n):
            # Use even simpler: just distance 1
            adj_bits = [0] * n
            for i in range(n):
                j1 = (i + 1) % n
                j2 = (i - 1) % n
                adj_bits[i] |= (1 << j1)
                adj_bits[i] |= (1 << j2)
        best_adj_bits = adj_bits
        best_n = n
        best_alpha = _max_independent_set_size_bitset(adj_bits, n)
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
