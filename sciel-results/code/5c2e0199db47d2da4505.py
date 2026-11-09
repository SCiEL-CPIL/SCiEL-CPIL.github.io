import time
import random
import itertools
import numpy as np

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin before 120s kill

    # Try to find a valid graph with n=21, then 20, etc.
    for target_n in [21, 20, 19, 18, 17]:
        if time.time() > deadline - 15:
            break

        # Strategy 1: Algebraic/Paley-type construction
        result = _try_algebraic(target_n, deadline)
        if result is not None:
            return result

        # Strategy 2: Random graphs with vectorized verification
        result = _try_random_search(target_n, deadline)
        if result is not None:
            return result

        # Strategy 3: Greedy construction with vectorized check
        result = _try_greedy(target_n, deadline)
        if result is not None:
            return result

    # Last resort: n=5 (trivially valid)
    return {'n': 5, 'edges': []}


def _try_algebraic(n, deadline):
    """
    Use quadratic character over GF(p) to create a Paley-type graph.
    For p = 2n+1 or nearby prime, define edge (i,j) iff (i-j)^2 is a quadratic residue mod p.
    This creates highly symmetric graphs that may avoid regular induced subgraphs.
    """
    if time.time() > deadline - 10:
        return None

    # Find a suitable prime p >= n
    primes = [p for p in range(n, n + 50) if _is_prime(p)]
    
    for p in primes:
        if time.time() > deadline - 10:
            break
        
        # Quadratic residues mod p
        # For vertex i, j: edge if (i-j) mod p is a quadratic residue (and i != j)
        # Paley graph: vertices are Z_p, edge between i,j iff i-j is a quadratic residue
        adj = np.zeros((n, n), dtype=np.uint8)
        
        # Precompute quadratic residues
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = (i - j) % p
                if diff in residues:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Verify with vectorized check
        if _is_valid_vectorized(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    return None


def _try_random_search(n, deadline):
    """
    Try random graphs with NumPy-vectorized validity checking.
    """
    if time.time() > deadline - 10:
        return None

    rng = np.random.RandomState(42)
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Generate random graph
        adj = (rng.random((n, n)) < 0.5).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        # Symmetrize
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        
        if _is_valid_vectorized(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # If simple random didn't work, try local search from a random start
    if time.time() < deadline - 15:
        adj = (rng.random((n, n)) < 0.5).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        
        result = _local_search(adj, n, deadline)
        if result is not None:
            return result
    
    return None


def _try_greedy(n, deadline):
    """
    Greedy vertex addition with bitmask adjacency and vectorized verification.
    """
    if time.time() > deadline - 10:
        return None

    # Build adjacency as bitmask for fast local operations
    adj_bits = [0] * n
    rng = random.Random(777)
    
    for k in range(1, n):
        if time.time() > deadline - 10:
            break
        
        best_pattern = None
        best_violations = float('inf')
        
        # Try several random patterns
        for _ in range(50):
            pattern = 0
            for i in range(k):
                if rng.random() < 0.5:
                    pattern |= (1 << i)
            
            # Quick violation count using bitmask
            violations = _count_violations_bitmask(adj_bits, k, pattern)
            if violations < best_violations:
                best_violations = violations
                best_pattern = pattern
                if violations == 0:
                    break
        
        if best_pattern is None:
            best_pattern = 0
        
        adj_bits[k] = best_pattern
        for i in range(k):
            if (best_pattern >> i) & 1:
                adj_bits[i] |= (1 << k)
    
    # Convert to numpy for verification
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if (adj_bits[i] >> j) & 1:
                adj[i, j] = 1
    
    if _is_valid_vectorized(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Local search
    if time.time() < deadline - 5:
        result = _local_search(adj, n, deadline)
        if result is not None:
            return result
    
    return None


def _is_valid_vectorized(adj, n):
    """
    Vectorized check: no subset of size >= 6 induces a regular graph.
    Uses NumPy for fast degree computation.
    """
    # Check all subsets of size 6 first (most likely violations)
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            idx = np.array(subset, dtype=np.intp)
            sub_adj = adj[np.ix_(idx, idx)]
            degs = sub_adj.sum(axis=1)
            if np.all(degs == degs[0]):
                return False
    return True


def _local_search(adj, n, deadline):
    """
    Local search: flip edges to eliminate regular induced subgraphs.
    """
    rng = np.random.RandomState(123)
    
    for step in range(5000):
        if time.time() > deadline - 5:
            break
        
        if _is_valid_vectorized(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Find a violation (sample random 6-subsets for speed)
        violation_found = False
        for _ in range(20):
            subset = rng.choice(n, size=6, replace=False)
            idx = np.array(subset, dtype=np.intp)
            sub_adj = adj[np.ix_(idx, idx)]
            degs = sub_adj.sum(axis=1)
            if np.all(degs == degs[0]):
                # Flip a random edge within this subset
                u, v = rng.choice(6, size=2, replace=False)
                adj[subset[u], subset[v]] = 1 - adj[subset[u], subset[v]]
                adj[subset[v], subset[u]] = adj[subset[u], subset[v]]
                violation_found = True
                break
        
        if not violation_found:
            # No 6-subset violation found by sampling; check larger or do random flip
            # Do a random edge flip
            u, v = rng.choice(n, size=2, replace=False)
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
    
    return None


def _count_violations_bitmask(adj_bits, k, pattern):
    """
    Count new violations when adding vertex k with connection pattern to vertices 0..k-1.
    Only checks subsets of size 5 from old vertices (forming 6-sets with k).
    """
    count = 0
    k_old = k
    # Check subsets of size 5 from vertices 0..k-1
    for subset in itertools.combinations(range(k_old), 5):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # m = number of neighbors of k in this subset
        m = bin(pattern & mask).count('1')
        
        # Check if all vertices in subset have degree m in the induced subgraph on subset U {k}
        is_reg = True
        for u in subset:
            deg_u = bin(adj_bits[u] & mask).count('1')
            conn = 1 if (pattern >> u) & 1 else 0
            if deg_u + conn != m:
                is_reg = False
                break
        
        if is_reg:
            count += 1
    
    return count


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
