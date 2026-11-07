"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import math
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    Uses periodicity with period LCM(1..12) = 27720.
    """
    n = 12
    max_m = 1000000
    max_L = 144
    
    # LCM of 1..12
    lcm_12 = 1
    for i in range(1, 13):
        lcm_12 = lcm_12 * i // math.gcd(lcm_12, i)
    # lcm_12 is 27720

    class Dinic:
        def __init__(self, num_nodes):
            self.n = num_nodes
            self.graph = [[] for _ in range(num_nodes)]
        
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.graph[u]:
                    v, cap, rev = e
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0
        
        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(self.it[u], len(self.graph[u])):
                self.it[u] = i
                v, cap, rev = self.graph[u][i]
                if cap > 0 and self.level[u] < self.level[v]:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        self.graph[u][i][1] -= d
                        self.graph[v][rev][1] += d
                        return d
            return 0
        
        def max_flow(self, s, t):
            flow = 0
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, 10**9)
                    if f == 0:
                        break
                    flow += f
            return flow

    def feasible(m, L):
        """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
        # Nodes:
        # 0: Source
        # 1..12: Indices 1..12
        # 13..13+L-1: Values (offsets 1..L)
        # 13+L: Sink
        num_nodes = 1 + n + L + 1
        source = 0
        sink = num_nodes - 1
        
        dinic = Dinic(num_nodes)
        
        # Source to indices
        for i in range(1, n + 1):
            dinic.add_edge(source, i, 1)
        
        # Indices to values
        # For index i, valid offsets k in 1..L such that (m+k) % i == 0
        # k % i == (-m) % i
        for i in range(1, n + 1):
            target_residue = (-m) % i
            # Find k in 1..L with k % i == target_residue
            # Smallest k >= 1 with k % i == target_residue
            # If target_residue == 0, k is multiple of i. Smallest is i (if i<=L).
            # If target_residue > 0, k is target_residue, target_residue+i, ...
            
            start_k = target_residue
            if start_k == 0:
                start_k = i
            elif start_k > L:
                # No solution in 1..L
                pass
            else:
                # Iterate k = start_k, start_k+i, ... <= L
                k = start_k
                while k <= L:
                    val_node = 1 + n + (k - 1) # 1-based offset k maps to node 1+n+k-1
                    dinic.add_edge(i, val_node, 1)
                    k += i
        
        # Values to Sink
        for k in range(1, L + 1):
            val_node = 1 + n + (k - 1)
            dinic.add_edge(val_node, sink, 1)
        
        return dinic.max_flow(source, sink) >= n

    best_m = 0
    best_L = 0
    
    # Search only in one period
    for m in range(lcm_12):
        # Binary search for min L
        lo, hi = 1, max_L
        res_L = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(m, mid):
                res_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if res_L > best_L:
            best_L = res_L
            best_m = m
            
    # Ensure best_m is in range [0, 1000000]
    # Since we iterated m in [0, 27719], best_m is already in range.
    return int(best_m)

# EVOLVE-BLOCK-END
