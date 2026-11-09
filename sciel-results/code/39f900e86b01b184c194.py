"""K4-free graphs without independent 6-sets (Ramsey number R(4,6))."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

_DEADLINE = 115.0  # seconds; evaluator kills at 120s, keep margin

def _has_k4(adj):
    """Check if adjacency matrix (list of bitmasks) contains a K4.
    adj[i] is a bitmask where bit j is set iff (i,j) is an edge.
    Returns True if K4 exists."""
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            # common neighbors of i and j
            common = adj[i] & adj[j]
            # remove i and j from common
            common &= ~(1 << i)
            common &= ~(1 << j)
            # check if any two vertices in common are adjacent -> K4
            bits = list(_bit_positions(common))
            for a_idx in range(len(bits)):
                a = bits[a_idx]
                for b_idx in range(a_idx + 1, len(bits)):
                    b = bits[b_idx]
                    if (adj[a] >> b) & 1:
                        return True
    return False

def _bit_positions(mask):
    """Yield positions of set bits."""
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def _has_independent_6(adj):
    """Check if graph has an independent set of size 6.
    adj[i] is bitmask of neighbors of i.
    Independent set = clique in complement.
    We check complement for K6."""
    n = len(adj)
    # complement adjacency
    comp = []
    for i in range(n):
        c = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
        comp.append(c)
    # Look for K6 in complement (i.e., independent 6-set in original)
    # Use a simple backtracking
    def _find_clique_k(comp_adj, k, start, current):
        if len(current) == k:
            return True
        remaining = k - len(current)
        # pruning: not enough vertices left
        if len(comp_adj) - start < remaining:
            return False
        for v in range(start, len(comp_adj)):
            if v in current:
                continue
            # v must be adjacent to all in current
            ok = True
            for u in current:
                if not (comp_adj[v] >> u) & 1:
                    ok = False
                    break
            if not ok:
                continue
            current.append(v)
            if _find_clique_k(comp_adj, k, v + 1, current):
                return True
            current.pop()
        return False
    return _find_clique_k(comp, 6, 0, [])

def _is_valid(adj):
    """Full validity check: no K4 and no independent 6-set."""
    if _has_k4(adj):
        return False
    if _has_independent_6(adj):
        return False
    return True

def _build_adj_from_edges(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _random_flip_search(n, max_iter, rng):
    """Random local search: start from random graph, flip edges to fix violations."""
    adj = [0] * n
    # Start with a random graph with moderate edge density
    p = 0.3
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_adj = None
    best_score = -1
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        # Check violations
        has_k4 = _has_k4(adj)
        has_ind6 = _has_independent_6(adj)
        
        if not has_k4 and not has_ind6:
            return adj
        
        score = 0
        if has_k4:
            score -= 100
        if has_ind6:
            score -= 50
        
        if score > best_score:
            best_score = score
            best_adj = [a for a in adj]
        
        # Try random flips
        for _ in range(50):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            # Flip edge (u,v)
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Quick check: only check K4 if we added an edge, only check ind6 if we removed
            # Actually just do full check periodically
            if _it % 10 == 0:
                h4 = _has_k4(adj)
                h6 = _has_independent_6(adj)
                if not h4 and not h6:
                    return adj
                s = 0
                if h4: s -= 100
                if h6: s -= 50
                if s > best_score:
                    best_score = s
                    best_adj = [a for a in adj]
            # revert
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    if best_adj is not None and _is_valid(best_adj):
        return best_adj
    return None

def _greedy_construct(n, rng):
    """Build graph vertex by vertex, ensuring constraints are maintained."""
    # We build the graph incrementally.
    # adj[i] for i in 0..k-1 already set.
    # When adding vertex k, choose its neighbors among 0..k-1 such that:
    #   - No K4 is formed
    #   - No independent 6-set is formed (this is harder to check incrementally)
    # This approach is complex; let's try a different structural approach.
    pass

def _paley_like(n, rng):
    """Try to construct a graph using a structured approach with random perturbations."""
    # Use a random regular-like graph as starting point
    adj = [0] * n
    # Random 3-regular graph
    for i in range(n):
        targets = [j for j in range(n) if j != i]
        rng.shuffle(targets)
        count = 0
        for j in targets:
            if count >= 3:
                break
            if not (adj[i] >> j) & 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                count += 1
    return adj

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_n = 0
    best_edges = []
    
    # Try increasing n from 20 up to 36
    # Known: Paley(17) works. Exoo has 35.
    # We'll try to find valid graphs for n = 20, 22, 24, ..., 35
    
    # First, try some known constructions or random search for small n
    # Then attempt larger n
    
    for target_n in [20, 22, 24, 26, 28, 30, 32, 34, 35, 36]:
        if time.time() > _DEADLINE - 30:
            break
        
        found = False
        attempts = 0
        max_attempts = 200 if target_n <= 26 else 100 if target_n <= 30 else 50
        
        while attempts < max_attempts and time.time() < _DEADLINE - 30:
            attempts += 1
            
            # Generate a random graph
            adj = [0] * target_n
            # Use a moderate density
            p = 0.25 + rng.random() * 0.15
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check validity
            if _is_valid(adj):
                found = True
                if target_n > best_n:
                    best_n = target_n
                    best_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                best_edges.append([i, j])
                break
            
            # If not valid, try local search to fix
            # Simple hill climbing: flip edges to reduce violations
            for flip_iter in range(200):
                if time.time() > _DEADLINE - 30:
                    break
                
                h4 = _has_k4(adj)
                h6 = _has_independent_6(adj)
                
                if not h4 and not h6:
                    found = True
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if (adj[i] >> j) & 1:
                                    best_edges.append([i, j])
                    break
                
                # Pick a random edge to flip
                u = rng.randint(0, target_n - 1)
                v = rng.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Flip
                if (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Check if improved
                new_h4 = _has_k4(adj)
                new_h6 = _has_independent_6(adj)
                
                old_bad = (1 if h4 else 0) + (1 if h6 else 0)
                new_bad = (1 if new_h4 else 0) + (1 if new_h6 else 0)
                
                if new_bad > old_bad:
                    # Revert
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                if not new_h4 and not new_h6:
                    found = True
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if (adj[i] >> j) & 1:
                                    best_edges.append([i, j])
                    break
        
        if found:
            # Found valid graph for target_n, try next larger
            continue
        else:
            # Failed for this n, try next
            continue
    
    # Fallback: use Paley graph on 17 vertices if nothing better found
    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
