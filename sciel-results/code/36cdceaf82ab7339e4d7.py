import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    random.seed(42)
    
    def find_k4(n, adj):
        nbr_mask = [0] * n
        for v in range(n):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            nbr_mask[v] = mask
            
        for v in range(n):
            nbrs = [u for u in range(n) if adj[v][u]]
            for i in range(len(nbrs)):
                u = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    w = nbrs[j]
                    if not adj[u][w]:
                        continue
                    common = nbr_mask[v] & nbr_mask[u] & nbr_mask[w]
                    common &= ~(1 << v)
                    common &= ~(1 << u)
                    common &= ~(1 << w)
                    if common:
                        bit = common & (-common)
                        x = bit.bit_length() - 1
                        return [v, u, w, x]
        return None

    def find_ind6(n, adj):
        non_nbr_mask = [0] * n
        for v in range(n):
            mask = 0
            for u in range(n):
                if u != v and not adj[v][u]:
                    mask |= (1 << u)
            non_nbr_mask[v] = mask
            
        def backtrack(available, depth, current_set):
            if depth == 0:
                return current_set
            if available == 0:
                return None
            if bin(available).count('1') < depth:
                return None
            v_bit = available & (-available)
            v = v_bit.bit_length() - 1
            new_avail = non_nbr_mask[v] & available
            new_avail &= ~v_bit
            res = backtrack(new_avail, depth - 1, current_set + [v])
            if res: return res
            res = backtrack(available & ~v_bit, depth - 1, current_set)
            if res: return res
            return None

        all_mask = (1 << n) - 1
        return backtrack(all_mask, 6, [])

    def repair(n, adj, time_limit):
        while time.time() < time_limit:
            k4 = find_k4(n, adj)
            if k4:
                u, v = random.sample(k4, 2)
                adj[u][v] = False
                adj[v][u] = False
                continue
            
            ind6 = find_ind6(n, adj)
            if ind6:
                u, v = random.sample(ind6, 2)
                adj[u][v] = True
                adj[v][u] = True
                continue
            
            return adj

    for n in range(36, 29, -1):
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            adj = repair(n, adj, deadline)
            
            if not find_k4(n, adj) and not find_ind6(n, adj):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback: Paley 17
    p = 17
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}
