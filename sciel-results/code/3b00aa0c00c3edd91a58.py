import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try n=21 first
    # If found, return it
    # Else try n=20
    # Else try n=19 ...
    
    best_graph = None
    best_n = 5 # Fallback
    
    # Helper to check if graph is valid
    def is_valid_graph(n, adj):
        # Check all subsets of size 6
        # If any is regular, return False
        # adj is list of lists or list of sets
        # Let's use list of sets for fast membership
        for subset in itertools.combinations(range(n), 6):
            s_set = set(subset)
            degrees = []
            for u in subset:
                deg = 0
                # Count neighbors in subset
                for v in subset:
                    if u != v and v in adj[u]:
                        deg += 1
                degrees.append(deg)
            if len(set(degrees)) == 1:
                return False
        return True

    def count_bad(n, adj):
        count = 0
        for subset in itertools.combinations(range(n), 6):
            s_set = set(subset)
            degrees = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and v in adj[u]:
                        deg += 1
                degrees.append(deg)
            if len(set(degrees)) == 1:
                count += 1
        return count

    def random_graph(n):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def flip_edge(adj, u, v):
        if v in adj[u]:
            adj[u].remove(v)
            adj[v].remove(u)
        else:
            adj[u].add(v)
            adj[v].add(u)

    def search(n, time_limit):
        # Simulated Annealing
        current_adj = random_graph(n)
        current_cost = count_bad(n, current_adj)
        best_adj = [s.copy() for s in current_adj]
        best_cost = current_cost
        T = 1.0
        cooling = 0.999
        
        while time.time() < time_limit:
            if current_cost == 0:
                return current_adj
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            flip_edge(current_adj, u, v)
            new_cost = count_bad(n, current_adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = [s.copy() for s in current_adj]
            else:
                # Accept with probability
                if new_cost > current_cost:
                    prob = 1.0 / (1.0 + new_cost - current_cost)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        flip_edge(current_adj, u, v)
                # If equal, accept
                else:
                    current_cost = new_cost
            
            T *= cooling
            
        if best_cost == 0:
            return best_adj
        return None

    # Try n=21
    t_start = time.time()
    time_for_21 = min(10, deadline - t_start)
    if time_for_21 > 1:
        res_21 = search(21, t_start + time_for_21)
        if res_21:
            edges = []
            for i in range(21):
                for j in res_21[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': 21, 'edges': edges}

    # Try n=20
    t_start = time.time()
    time_for_20 = min(30, deadline - t_start)
    if time_for_20 > 1:
        res_20 = search(20, t_start + time_for_20)
        if res_20:
            edges = []
            for i in range(20):
                for j in res_20[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}
            
    # If not found, try smaller n to ensure valid output
    # n=10 is likely to be found instantly or is trivial
    # Actually, let's just try n=15 with remaining time
    t_start = time.time()
    remaining = deadline - t_start
    if remaining > 5:
        res_15 = search(15, t_start + remaining)
        if res_15:
            edges = []
            for i in range(15):
                for j in res_15[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': 15, 'edges': edges}
            
    # Fallback: Empty graph on 5 vertices (known valid)
    return {'n': 5, 'edges': []}
