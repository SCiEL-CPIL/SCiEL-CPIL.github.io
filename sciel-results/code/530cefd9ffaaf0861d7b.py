import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def _is_regular_subset(vertices, adj_bits):
        """Check if the induced subgraph on 'vertices' is regular."""
        k = len(vertices)
        if k < 2:
            return True
        mask = 0
        for v in vertices:
            mask |= (1 << v)
        
        degs = []
        for v in vertices:
            d = (adj_bits[v] & mask).bit_count()
            degs.append(d)
            
        return len(set(degs)) == 1

    def _count_violations(n, adj_bits, min_k=6, max_k=None):
        """Count number of regular induced subgraphs of size k in [min_k, max_k]."""
        if max_k is None:
            max_k = n
        count = 0
        for k in range(min_k, max_k + 1):
            for comb in itertools.combinations(range(n), k):
                if _is_regular_subset(comb, adj_bits):
                    count += 1
        return count

    def _is_valid(n, adj_bits):
        """Check if graph has NO regular induced subgraph of size >= 6."""
        return _count_violations(n, adj_bits, 6, n) == 0

    def _local_search(n, adj_bits, time_limit):
        """
        Local search to minimize number of regular 6-sets.
        Returns adj_bits with 0 violations if possible, else best found.
        """
        current_bad = _count_violations(n, adj_bits, 6, 6)
        best_adj = adj_bits[:]
        best_bad = current_bad
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < time_limit:
            u, v = random.choice(edges)
            
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            
            new_bad = _count_violations(n, adj_bits, 6, 6)
            
            if new_bad <= current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_adj = adj_bits[:]
                if current_bad == 0:
                    if _is_valid(n, adj_bits):
                        return adj_bits
                    break
            else:
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
        return best_adj

    def _try_n(n, time_budget):
        """Try to find a valid graph on n vertices."""
        for seed in range(10):
            if time.time() > start_time + 110:
                break
            
            random.seed(seed)
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            restart_budget = time_budget / 10
            adj_res = _local_search(n, adj_bits, time.time() + restart_budget)
            
            if _is_valid(n, adj_res):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_res[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
        return None

    for n in range(20, 9, -1):
        if time.time() > start_time + 100:
            break
        
        budget = 10 if n > 15 else 15
        result = _try_n(n, budget)
        
        if result is not None:
            return result

    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': 6, 'edges': edges}
