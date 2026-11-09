"""
Fast construction of a graph with no regular induced subgraph on 6+ vertices.

Strategy:
- Generate candidate graphs using random or algebraic constructions.
- Use fast bitmask-based validation to check for regular induced subgraphs.
- Perform local search (edge flipping) to repair violations if found.
- Use a tight time guard to ensure we return the best result before timeout.
"""
import time
import random
import itertools
import numpy as np

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin before 120s kill

    best_n = 0
    best_edges = []

    # Try target sizes in descending order
    # n=20 is known to be possible. n=21 is the target for improvement.
    for target_n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break

        # Try multiple seeds for each target n
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(seed * 1000 + target_n)
            # Generate a random graph with p=0.5
            # Or use a specific construction if available
            adj = _generate_random_graph(target_n, rng, 0.5)
            
            # Validate and repair
            adj = _repair_graph(adj, target_n, deadline - 2)
            
            if adj is not None and _is_valid_fast(adj, target_n):
                if target_n > best_n:
                    best_n = target_n
                    best_edges = _adj_to_edges(adj, target_n)
                    # If we found a valid graph for this n, we can try a larger n?
                    # The loop structure tries larger n first, so if we succeed at 21, we break out.
                    # If we succeed at 20, we break out of seed loop, then try 19? No, we want max n.
                    # So if we find valid 21, we return. If we find valid 20, we return.
                    # The outer loop goes 21, 20, 19... so if 21 fails, it tries 20.
                    # If 20 succeeds, it breaks the seed loop. Then it goes to 19?
                    # No, we should break the outer loop if we found a valid graph for the current target, 
                    # unless we are specifically looking for a larger one and failed.
                    # But since we iterate descending, the first valid one found is the max for this run.
                    break
        
        if best_n >= target_n:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}

    # Fallback: return a small valid graph (e.g., empty graph on 5 vertices)
    return {'n': 5, 'edges': []}


def _generate_random_graph(n, rng, p):
    """Generate a random graph on n vertices with edge probability p."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _is_regular_fast(adj, mask, size):
    """
    Check if the induced subgraph on vertices in 'mask' is regular.
    'mask' is an integer bitmask. 'size' is the number of bits in mask.
    """
    if size < 2:
        return True # Trivially regular? Or just ignore. We check size >= 6.
    
    deg = -1
    # Iterate through set bits in mask
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        m ^= lsb
        
        # Calculate degree of v within the subgraph
        # adj[v] contains neighbors. & mask restricts to subgraph.
        # bit_count gives the number of neighbors in the subgraph.
        d = (adj[v] & mask).bit_count()
        
        if deg == -1:
            deg = d
        elif d != deg:
            return False
            
    return True


def _is_valid_fast(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Uses bitmask iteration for speed.
    """
    # We check all subsets of size k from 6 to n
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_fast(adj, mask, k):
                return False
    return True


def _find_violation_fast(adj, n):
    """
    Find one regular induced subgraph of size >= 6.
    Returns (mask, size) or None.
    """
    # Check size 6 first, then 7, etc.
    # Usually size 6 violations are most common or easiest to find?
    # Or maybe larger ones are easier to break?
    # Let's check 6 first.
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_fast(adj, mask, k):
                return mask, k
    return None


def _repair_graph(adj, n, deadline):
    """
    Local search: repeatedly find a violation and flip an edge to break it.
    Returns the repaired adj list or None if timeout.
    """
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Check if valid
        if _is_valid_fast(adj, n):
            return adj
            
        # Find a violation
        viol = _find_violation_fast(adj, n)
        if viol is None:
            # Should be valid if no violation found
            return adj
            
        mask, size = viol
        
        # Pick a random edge within the violation to flip
        # Get list of vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            verts.append(v)
            
        # Pick random pair
        i, j = random.sample(verts, 2)
        
        # Flip edge (i, j)
        if adj[i] & (1 << j):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    return None


def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
