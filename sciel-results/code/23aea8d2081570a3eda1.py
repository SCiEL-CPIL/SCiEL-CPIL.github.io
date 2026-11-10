"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_is_regular(adj_bits, subset):
    """Check if induced subgraph on subset is regular. subset is a tuple."""
    k = len(subset)
    if k < 6:
        return False
    mask = 0
    for v in subset:
        mask |= (1 << v)
    first_deg = None
    for u in subset:
        nb = adj_bits[u] & mask
        if nb & (1 << u):
            nb &= ~(1 << u)
        d = nb.bit_count()
        if first_deg is None:
            first_deg = d
        elif d != first_deg:
            return False
    return True

def _count_six_violations(adj_bits, n):
    """Count number of 6-vertex subsets that induce a regular graph."""
    count = 0
    for subset in combinations(range(n), 6):
        if _subset_is_regular(adj_bits, subset):
            count += 1
    return count

def _has_any_regular_induced(adj_bits, n):
    """Full check: any induced subgraph of size >= 6 regular?"""
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            if _subset_is_regular(adj_bits, subset):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    random.seed(42)
    
    for n in range(22, 9, -1):
        if time.time() > deadline - 15.0:
            break
        
        best_overall_count = float('inf')
        best_overall_adj = None
        
        for restart in range(40):
            if time.time() > deadline - 3.0:
                break
            
            # Random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Quick full check
            if not _has_any_regular_induced(adj, n):
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
                return {'n': n, 'edges': edges}
            
            # Local search on 6-vertex violations
            current_count = _count_six_violations(adj, n)
            if current_count == 0:
                if not _has_any_regular_induced(adj, n):
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
                    return {'n': n, 'edges': edges}
            
            best_count = current_count
            best_adj = adj[:]
            
            for step in range(300):
                if time.time() > deadline - 2.0:
                    break
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                old = (adj[i] >> j) & 1
                if old:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_count = _count_six_violations(adj, n)
                
                if new_count <= current_count:
                    current_count = new_count
                    if new_count == 0:
                        if not _has_any_regular_induced(adj, n):
                            edges = [[a, b] for a in range(n) for b in range(a + 1, n) if adj[a] & (1 << b)]
                            return {'n': n, 'edges': edges}
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj[:]
                else:
                    if old:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
            
            if best_count < best_overall_count:
                best_overall_count = best_count
                best_overall_adj = best_adj
        
        # Final attempt with best_adj
        if best_overall_adj is not None and best_overall_count < 10 and time.time() < deadline - 8.0:
            adj = best_overall_adj[:]
            current_count = _count_six_violations(adj, n)
            for step in range(500):
                if time.time() > deadline - 2.0:
                    break
                if current_count == 0:
                    if not _has_any_regular_induced(adj, n):
                        edges = [[a, b] for a in range(n) for b in range(a + 1, n) if adj[a] & (1 << b)]
                        return {'n': n, 'edges': edges}
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                old = (adj[i] >> j) & 1
                if old:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_count = _count_six_violations(adj, n)
                if new_count <= current_count:
                    current_count = new_count
                else:
                    if old:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
    
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
