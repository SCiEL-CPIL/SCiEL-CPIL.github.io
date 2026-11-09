"""Construct a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Target n=21 first, then 20
    for target_n in [21, 20]:
        if time.time() > deadline - 20:
            break
        result = _local_search(target_n, deadline)
        if result is not None:
            return result

    # Fallback
    return {'n': 5, 'edges': []}


def _local_search(n, deadline):
    """
    Find a valid graph on n vertices using local search.
    Objective: Eliminate all regular induced subgraphs of size 6.
    """
    max_iterations = 500
    best_edges = None
    
    # Try a few different starting probabilities
    p_starts = [0.5, 0.4, 0.6, 0.3, 0.7]
    
    for p_init in p_starts:
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p_init:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Local search loop
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Find a violation
            violation = _find_violation(adj_mask, n)
            
            if violation is None:
                # Valid graph found
                edges = _masks_to_edges(adj_mask, n)
                return {'n': n, 'edges': edges}
            
            # Flip a random edge within the violation to break it
            # violation is a tuple of 6 vertices
            u, v = random.sample(violation, 2)
            
            # Flip edge (u, v)
            if adj_mask[u] & (1 << v):
                # Remove edge
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
    return None


def _find_violation(adj_mask, n):
    """
    Find any subset of 6 vertices that induces a regular subgraph.
    Returns the subset as a tuple, or None if no violation found.
    """
    # We only need to check size 6. If there is a regular induced subgraph of size > 6,
    # it might not contain a regular induced subgraph of size 6?
    # Actually, the problem states "No set of 6 or more vertices may induce a regular subgraph".
    # If a set of 7 is regular, does it contain a regular set of 6?
    # Not necessarily.
    # However, checking all sizes is expensive.
    # Let's check size 6 first. If we clear all size-6 regular subgraphs, 
    # do we clear larger ones?
    # A regular graph of size k >= 7 might not have a regular induced subgraph of size 6.
    # Example: C7 is 2-regular. Induced subgraphs of size 6 are paths or cycles?
    # C7 induced on 6 vertices: remove one vertex. Result is P6 (path of 6 vertices).
    # Degrees in P6: 1, 2, 2, 2, 2, 1. Not regular.
    # So clearing size 6 does NOT guarantee clearing size 7.
    # We must check all sizes >= 6.
    
    # But checking all sizes is very slow.
    # Strategy: 
    # 1. Check size 6. If found, fix it.
    # 2. After no size 6 violations remain, check size 7.
    # 3. Then size 8, etc.
    
    # For the local search step, finding *any* violation is enough.
    # We can prioritize smaller sizes as they are more frequent?
    # Or just check size 6 first, then 7, etc.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check regularity
            first_deg = (adj_mask[subset[0]] & mask).bit_count()
            is_reg = True
            for v in subset[1:]:
                if (adj_mask[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return subset
                
    return None


def _masks_to_edges(adj_mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                edges.append([i, j])
    return edges
