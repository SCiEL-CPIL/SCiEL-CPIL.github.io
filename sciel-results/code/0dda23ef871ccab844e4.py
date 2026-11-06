import time
import random
import numpy as np
from itertools import combinations

def _paley_orientation(n=83):
    """Return orientation bits for Paley tournament on n vertices."""
    p = n
    squares = set((x * x) % p for x in range(1, p))
    bits = []
    for i in range(p):
        for j in range(i + 1, p):
            bits.append(0 if (j - i) % p in squares else 1)
    return bits

def _count_uncovered(bits, n, sample_size=None, rng=None):
    """
    Count the number of 4-sets that do NOT have a common dominator.
    A vertex v dominates a 4-set {a,b,c,d} if v has arcs to all of a,b,c,d.
    
    We use a vectorized approach: for each vertex v, compute the set of vertices
    that v dominates (out-neighbors). Then for each 4-set, check if any vertex
    dominates all 4.
    
    For efficiency, we'll check: for each 4-set, is there a vertex with out-degree 
    covering all 4? We can precompute for each vertex its out-neighborhood as a bitset.
    """
    if rng is None:
        rng = random.Random(42)
    
    # Build adjacency: adj[i] is a set of j such that i -> j
    # Actually, let's use bitmasks for speed with numpy
    n_pairs = n * (n - 1) // 2
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if bits[idx] == 0:
                adj[i, j] = True  # i -> j
            else:
                adj[j, i] = True  # j -> i
            idx += 1
    
    # For each vertex v, out_neighbors[v] is the set of vertices v points to
    # A 4-set {a,b,c,d} is covered if exists v such that v->a, v->b, v->c, v->d
    # i.e., v is in the common in-neighborhood of {a,b,c,d}
    
    # For each 4-set, compute the intersection of in-neighborhoods of a,b,c,d
    # If non-empty, the 4-set is covered.
    
    # in_neighbors[v] = set of u such that u -> v
    in_adj = adj.T  # in_adj[v, u] = adj[u, v] = (u -> v)
    
    total_4sets = 0
    uncovered = 0
    
    if sample_size is not None and sample_size < len(list(combinations(range(n), 4))):
        # Sample 4-sets
        all_4sets = list(combinations(range(n), 4))
        sampled = rng.sample(all_4sets, min(sample_size, len(all_4sets)))
        for (a, b, c, d) in sampled:
            # Check if there exists v such that v->a, v->b, v->c, v->d
            # i.e., v is in in_adj[:, a] & in_adj[:, b] & in_adj[:, c] & in_adj[:, d]
            common = in_adj[:, a] & in_adj[:, b] & in_adj[:, c] & in_adj[:, d]
            if not common.any():
                uncovered += 1
        # Scale up
        total_4sets = len(all_4sets)
        return uncovered * (total_4sets / len(sampled))
    else:
        # Check all 4-sets
        for (a, b, c, d) in combinations(range(n), 4):
            common = in_adj[:, a] & in_adj[:, b] & in_adj[:, c] & in_adj[:, d]
            if not common.any():
                uncovered += 1
        return uncovered

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with Paley tournament on 83 vertices
    best_bits = _paley_orientation(83)
    best_n = 83
    
    # Try to find a valid tournament with fewer vertices
    # Strategy: Try n from 31 up to 82, using genetic algorithm or local search
    
    rng = random.Random(12345)
    np_rng = np.random.RandomState(42)
    
    # First, try a quick local search / hill climbing for each n
    for n in range(31, min(83, 82 + 1)):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple restarts
        max_restarts = 5 if n < 60 else 3
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Initialize: random orientation
            bits = np_rng.randint(0, 2, size=n * (n - 1) // 2)
            
            # Local search: flip bits to reduce uncovered 4-sets
            # Use a greedy approach: find the bit flip that reduces uncovered count the most
            
            # For efficiency, we'll use a simpler approach: random restarts with simulated annealing
            current_bits = bits.copy()
            current_uncovered = _count_uncovered(current_bits.tolist(), n, sample_size=2000, rng=rng)
            
            best_local_bits = current_bits.copy()
            best_local_uncovered = current_uncovered
            
            # Simulated annealing
            T = 100.0
            cooling = 0.995
            max_iter = 5000
            patience = 0
            patience_limit = 1000
            
            for it in range(max_iter):
                if time.time() - start_time > deadline - 5:
                    break
                
                if patience > patience_limit:
                    break
                
                # Pick a random bit to flip
                idx = np_rng.randint(0, len(current_bits))
                old_bit = current_bits[idx]
                current_bits[idx] = 1 - old_bit
                
                new_uncovered = _count_uncovered(current_bits.tolist(), n, sample_size=1000, rng=rng)
                delta = new_uncovered - current_uncovered
                
                if delta < 0 or np_rng.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_uncovered = new_uncovered
                    if new_uncovered < best_local_uncovered:
                        best_local_uncovered = new_uncovered
                        best_local_bits = current_bits.copy()
                        patience = 0
                    else:
                        patience += 1
                else:
                    current_bits[idx] = old_bit
                    patience += 1
                
                T *= cooling
            
            if best_local_uncovered == 0:
                # Found a valid tournament!
                return {'n': n, 'orientation': best_local_bits.tolist()}
            
            if best_local_uncovered < 0.01:  # Very few uncovered
                # Do a more thorough check
                exact_uncovered = _count_uncovered(best_local_bits.tolist(), n, sample_size=None, rng=rng)
                if exact_uncovered == 0:
                    return {'n': n, 'orientation': best_local_bits.tolist()}
    
    # If we couldn't find a smaller one, return the Paley tournament
    return {'n': 83, 'orientation': best_bits}
