import numpy as np

def priority(instance):
        durations = np.asarray(instance["durations"], dtype=float)
        demands = instance["demands"]
        capacities = np.asarray(instance["capacities"], dtype=float)
        predecessors = instance["predecessors"]
        
        n = len(durations)
        
        # 1. Build successors
        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in predecessors[i]:
                successors[p].append(i)
                
        # 2. Topological sort (reverse) to compute LPE
        # Using Kahn's algorithm or simple DFS
        # Since N is small (30), we can even do a simple recursive memoization or just
        # iterate until no changes (Bellman-Ford style on DAG)
        
        # Let's use a standard reverse topological order.
        # Calculate in-degrees for topological sort
        in_degree = [0] * n
        for i in range(n):
            for p in predecessors[i]:
                in_degree[i] += 1 # Wait, predecessors[i] are nodes that point TO i.
                # So edge p -> i.
                # In-degree of i is len(predecessors[i]).
                
        # Actually, in_degree[i] = len(predecessors[i])
        
        from collections import deque
        q = deque()
        for i in range(n):
            if len(predecessors[i]) == 0:
                q.append(i)
                
        topo_order = []
        while q:
            u = q.popleft()
            topo_order.append(u)
            for v in successors[u]:
                # Remove edge u->v
                # We don't have a mutable graph easily, but we can track remaining predecessors
                pass
        
        # Simpler for N=30:
        # Just compute LPE via recursion with memoization or dynamic programming on topological order.
        
        # Let's get topological order properly.
        in_deg = [len(preds) for preds in predecessors]
        q = deque([i for i in range(n) if in_deg[i] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in successors[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
                    
        # Compute LPE (Longest Path to End)
        lpe = [0.0] * n
        # Iterate in reverse topological order
        for u in reversed(topo):
            if not successors[u]:
                lpe[u] = durations[u]
            else:
                max_succ_lpe = max(lpe[v] for v in successors[u])
                lpe[u] = durations[u] + max_succ_lpe
                
        # Compute Demand Ratio
        # demands is likely list of lists or numpy array
        # Let's assume demands[i][r]
        max_ratios = []
        for i in range(n):
            max_r = 0.0
            for r in range(len(capacities)):
                if capacities[r] > 0:
                    ratio = demands[i][r] / capacities[r]
                    if ratio > max_r:
                        max_r = ratio
            max_ratios.append(max_r)
            
        # Combine
        scores = []
        for i in range(n):
            # Score = LPE * (1 + Ratio)
            score = lpe[i] * (1.0 + max_ratios[i])
            scores.append(score)
            
        return np.array(scores)
