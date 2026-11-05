import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(80):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add_fast(adj_sets, n, u, v):
    """
    Check if adding edge (u,v) creates a C4 using set intersection.
    adj_sets: list of sets of neighbors.
    Returns True if C4 would be created.
    """
    neighbors_u = adj_sets[u]
    neighbors_v = adj_sets[v]
    
    for x in neighbors_u:
        if x == v:
            continue
        common = adj_sets[x] & neighbors_v
        if u in common:
            common = common - {u}
        if x in common:
            common = common - {x}
        if common:
            return True
    return False


def _polarity_seed(n, seed_offset):
    """
    Create an initial adjacency using a structured construction.
    """
    rng = random.Random(seed_offset * 104729 + 41)
    adj_sets = [set() for _ in range(n)]
    
    # Assign each vertex a random "type"
    types = [rng.randint(0, 3) for _ in range(n)]
    
    # Connect vertices if their types satisfy a condition that avoids C4
    for u in range(n):
        for v in range(u + 1, n):
            if abs(types[u] - types[v]) == 1:
                if not _check_c4_add_fast(adj_sets, n, u, v):
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
    
    return adj_sets


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    if seed_offset % 3 == 0:
        adj_sets = _polarity_seed(n, seed_offset)
    else:
        adj_sets = [set() for _ in range(n)]
    
    # Phase 1: Greedy build
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    if seed_offset % 2 == 0:
        degrees = [len(s) for s in adj_sets]
        all_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]])
        rng.shuffle(all_edges)
    else:
        rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if not _check_c4_add_fast(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
    # Phase 2: Local search to improve minimum degree
    max_iters = 2000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add_fast(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added = True
                break
        
        if not added:
            all_current_edges = []
            for u in range(n):
                for w in adj_sets[u]:
                    if u < w:
                        all_current_edges.append((u, w))
                        
            if not all_current_edges:
                break
                
            # Try multiple swap candidates
            swap_found = False
            for _ in range(5):
                if time.time() > deadline:
                    break
                    
                a, b = rng.choice(all_current_edges)
                
                adj_sets[a].remove(b)
                adj_sets[b].remove(a)
                
                added_edges = []
                candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
                rng.shuffle(candidates)
                
                for u in candidates:
                    if time.time() > deadline:
                        break
                    if not _check_c4_add_fast(adj_sets, n, u, v):
                        adj_sets[u].add(v)
                        adj_sets[v].add(u)
                        added_edges.append(u)
                        if len(added_edges) >= 3:
                            break
                            
                new_degrees = [len(s) for s in adj_sets]
                new_min = min(new_degrees)
                
                if new_min > min_deg:
                    swap_found = True
                    break
                else:
                    adj_sets[a].add(b)
                    adj_sets[b].add(a)
                    for u in added_edges:
                        adj_sets[u].remove(v)
                        adj_sets[v].remove(u)
            
            if not swap_found:
                break

    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END
