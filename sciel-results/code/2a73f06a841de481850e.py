import numpy as np
import random

def construct():
    N = 256
    
    # 1. Build the conflict graph
    # We need to find all pairs (i, j) with i < j such that (i*j) % (i+j) == 0
    edges = []
    adj = [set() for _ in range(N)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                u, v = i - 1, j - 1
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
    
    def greedy_is(ordering):
        """
        Greedy Independent Set based on a given ordering of nodes.
        """
        in_set = [False] * N
        selected = []
        for node in ordering:
            if in_set[node]:
                continue
            # Check if any neighbor is in the set
            conflict = False
            for neighbor in adj[node]:
                if in_set[neighbor]:
                    conflict = True
                    break
            if not conflict:
                in_set[node] = True
                selected.append(node)
        return selected

    def expand_is(selected):
        """
        Try to add any remaining nodes that don't conflict with the current set.
        """
        in_set = [False] * N
        for node in selected:
            in_set[node] = True
        
        changed = True
        while changed:
            changed = False
            for node in range(N):
                if in_set[node]:
                    continue
                conflict = False
                for neighbor in adj[node]:
                    if in_set[neighbor]:
                        conflict = True
                        break
                if not conflict:
                    in_set[node] = True
                    selected.append(node)
                    changed = True
        return selected

    best_set = []
    
    # 1. Min-degree greedy
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(N)]
    min_deg_order = sorted(range(N), key=lambda x: degrees[x])
    s1 = greedy_is(min_deg_order)
    s1 = expand_is(s1)
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Random restarts
    # Since N=256 is small, we can do many restarts
    random.seed(42)
    for _ in range(100):
        random_order = list(range(N))
        random.shuffle(random_order)
        s_rand = greedy_is(random_order)
        s_rand = expand_is(s_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    # 3. Local Search (Simple Hill Climbing)
    # Try to swap nodes to increase size
    # This is optional if greedy is good enough, but let's add a simple version
    # Actually, for sparse graphs, greedy+expand is often near-optimal.
    # Let's try one more strategy: Max-degree greedy?
    max_deg_order = sorted(range(N), key=lambda x: -degrees[x])
    s2 = greedy_is(max_deg_order)
    s2 = expand_is(s2)
    if len(s2) > len(best_set):
        best_set = s2

    # Convert 0-indexed to 1-indexed
    return [node + 1 for node in best_set]
