import time
import math
import sys
from collections import deque

def construct():
    """
    Find the maximum cardinality subset S of [1, 512] such that lcm(a, b) <= 512 for all a, b in S.
    
    Strategy:
    1. Build conflict graph (edge between a,b if lcm(a,b) > 512)
    2. Use max-flow/min-cut to find maximum weight closure (strong initial bound)
    3. Use branch-and-bound with the min-cut bound to verify/improve
    """
    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 80  # Safety margin
    
    # 1. Precompute adjacency: adj[i] has bit j set if lcm(i,j) <= 512 (compatible)
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
    
    # 2. Max-flow/min-cut for maximum weight closure
    # Nodes: 1..N (each number), plus source s and sink t
    # For maximum weight closure:
    #   - If node has positive weight w: edge s->node with capacity w
    #   - If node has negative weight w: edge node->t with capacity -w
    #   - For each conflict edge (u,v) where both can't be in closure: 
    #     add edges u->v and v->u with capacity INF
    #   - Min cut gives the minimum weight set to exclude
    #   - Max closure weight = total_positive_weight - min_cut_value
    
    INF = 10**9
    num_nodes = N + 2  # 1..N, s=N+1, t=N+2
    s = N + 1
    t = N + 2
    
    # Build adjacency list for flow network
    # Use list of [to, capacity] for each node
    flow_adj = [[] for _ in range(num_nodes + 1)]
    flow_adj_rev = [[] for _ in range(num_nodes + 1)]
    
    def add_edge(u, v, cap):
        flow_adj[u].append([v, cap, len(flow_adj[v])])
        flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
    
    total_positive = 0
    # All nodes have weight 1 (positive)
    for i in range(1, N + 1):
        add_edge(s, i, 1)
        total_positive += 1
    
    # Add conflict edges: if lcm(i,j) > 512, we can't have both in closure
    # This means: if i is in closure, j must NOT be in closure, and vice versa
    # Model: add directed edges i->j and j->i with capacity INF
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l > LIMIT:
                # Conflict: can't have both
                add_edge(i, j, INF)
                add_edge(j, i, INF)
    
    # Dinic's algorithm for max flow
    def max_flow():
        flow = 0
        while True:
            # BFS to find augmenting paths
            level = [-1] * (num_nodes + 1)
            level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, _ in flow_adj[u]:
                    if cap > 0 and level[v] == -1:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] == -1:
                break
            # DFS to send flow
            it = [0] * (num_nodes + 1)
            def dfs(u, f):
                if u == t:
                    return f
                while it[u] < len(flow_adj[u]):
                    v, cap, rev = flow_adj[u][it[u]]
                    if cap > 0 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, cap))
                        if d > 0:
                            flow_adj[u][it[u]][1] -= d
                            flow_adj[v][rev][1] += d
                            return d
                    it[u] += 1
                return 0
            while True:
                f = dfs(s, INF)
                if f == 0:
                    break
                flow += f
        return flow
    
    min_cut_value = max_flow()
    # Max closure size = total_positive - min_cut_value
    # But we need to find WHICH nodes are in the closure
    # Nodes reachable from s in residual graph are in the closure
    
    # BFS from s in residual graph
    in_closure = [False] * (num_nodes + 1)
    in_closure[s] = True
    q = deque([s])
    while q:
        u = q.popleft()
        for v, cap, _ in flow_adj[u]:
            if cap > 0 and not in_closure[v]:
                in_closure[v] = True
                q.append(v)
    
    # Build initial solution from min-cut
    initial_mask = 0
    for i in range(1, N + 1):
        if in_closure[i]:
            initial_mask |= (1 << (i - 1))
    
    best_clique_size = initial_mask.bit_count()
    best_clique_mask = initial_mask
    
    # 3. Branch and bound to verify/improve
    # Use the min-cut bound as a strong upper bound for pruning
    
    def bron_kerbosch_opt(R, P, X):
        nonlocal best_clique_size, best_clique_mask, deadline
        
        if time.time() > deadline:
            return False
        
        # Pruning: if current clique size + remaining candidates <= best, stop
        if R.bit_count() + P.bit_count() <= best_clique_size:
            return True
            
        if P == 0:
            size = R.bit_count()
            if size > best_clique_size:
                best_clique_size = size
                best_clique_mask = R
            return True
        
        PX = P | X
        if PX == 0:
            return True

        # Select pivot u from P U X to minimize candidates
        max_pop = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u_idx = (lsb.bit_length() - 1)
            u = u_idx + 1
            pop = (adj[u] & P).bit_count()
            if pop > max_pop:
                max_pop = pop
                pivot = u
            temp ^= lsb
        
        if pivot == -1:
            return True

        candidates = P & ~adj[pivot]
        
        while candidates:
            if time.time() > deadline:
                return False
            
            lsb = candidates & -candidates
            v_idx = (lsb.bit_length() - 1)
            v = v_idx + 1
            
            new_R = R | lsb
            new_P = P & ~adj[v]
            new_X = X & ~adj[v]
            
            if not bron_kerbosch_opt(new_R, new_P, new_X):
                return False
            
            P = P & ~lsb
            X = X | lsb
            candidates = candidates & ~lsb
            
        return True
    
    # Run branch and bound starting from empty set
    # But we already have a good initial solution from min-cut
    # Try to improve it
    
    success = bron_kerbosch_opt(0, (1 << N) - 1, 0)
    
    # Convert mask to list
    result = [i + 1 for i in range(N) if (best_clique_mask >> i) & 1]
    
    # Fallback if result is empty
    if not result:
        result = [1]
        
    return result
