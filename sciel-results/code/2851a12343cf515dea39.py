import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def full_check(n, adj):
        """Exhaustive check: no subset of size >= 6 is regular."""
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                d0 = bin(adj[subset[0]] & mask).count('1')
                for v in subset[1:]:
                    if bin(adj[v] & mask).count('1') != d0:
                        break
                else:
                    return False
        return True
    
    def find_violation(n, adj, num_samples=200):
        """Find a regular subset of size >= 6, or return None."""
        # Check all 6-subsets
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            d0 = bin(adj[subset[0]] & mask).count('1')
            regular = True
            for v in subset[1:]:
                if bin(adj[v] & mask).count('1') != d0:
                    regular = False
                    break
            if regular:
                return subset
        # Check random larger subsets
        for _ in range(num_samples):
            size = random.randint(7, min(n, 16))
            subset = random.sample(range(n), size)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            d0 = bin(adj[subset[0]] & mask).count('1')
            regular = True
            for v in subset[1:]:
                if bin(adj[v] & mask).count('1') != d0:
                    regular = False
                    break
            if regular:
                return subset
        return None
    
    def try_build(n, max_rounds=3000):
        for attempt in range(30):
            if time.time() > deadline:
                return None
            
            edges = set()
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.45:
                        edges.add((i, j))
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            for rnd in range(max_rounds):
                if time.time() > deadline:
                    return None
                
                viol = find_violation(n, adj)
                if viol is None:
                    # Do full check
                    if full_check(n, adj):
                        return {'n': n, 'edges': list(edges)}
                    continue
                
                # Fix: modify an edge within viol
                vlist = list(viol)
                pairs = [(vlist[i], vlist[j]) for i in range(len(vlist)) for j in range(i+1, len(vlist))]
                u, v = random.choice(pairs)
                if (adj[u] >> v) & 1:
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.discard((u, v))
                else:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        
        return None
    
    best = None
    for n in range(22, 4, -1):
        if time.time() > deadline:
            break
        res = try_build(n)
        if res:
            best = res
            break
    
    if best is None:
        best = {'n': 5, 'edges': []}
    
    return best
