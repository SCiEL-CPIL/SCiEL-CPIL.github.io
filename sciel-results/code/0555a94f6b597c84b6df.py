import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def solve_instance(n, p, seed):
        adj = get_graph(n, p, seed)
        
        # Build complement graph
        adj_comp = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and v not in adj[u]:
                    adj_comp[u].add(v)
        
        best_clique = []
        best_size = 0
        
        P = set(range(n))
        X = set()
        R = []
        
        def bk(R, P, X):
            nonlocal best_size, best_clique
            if time.time() > deadline:
                return
            
            if not P and not X:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return
            
            if not P:
                return
                
            if len(R) + len(P) <= best_size:
                return

            # Greedy coloring for bound
            # Sort P by degree in comp graph descending
            p_list = sorted(P, key=lambda x: len(adj_comp[x]), reverse=True)
            colors = {}
            for v in p_list:
                neighbor_colors = set()
                for u in adj_comp[v]:
                    if u in colors:
                        neighbor_colors.add(colors[u])
                c = 0
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
            
            p_ordered = sorted(p_list, key=lambda x: (colors[x], x))
            max_color = max(colors.values()) if colors else 0
            
            if len(R) + max_color <= best_size:
                return

            for v in p_ordered:
                if time.time() > deadline:
                    return
                
                # v must be in P
                if v not in P:
                    continue

                new_R = R + [v]
                new_P = P - {v} - adj_comp[v]
                new_X = X - {v}
                
                bk(new_R, new_P, new_X)
                
                P.remove(v)
                X.add(v)
                
                # Optional: prune if remaining P is too small
                # The bound for remaining P is not strictly max_color anymore, 
                # but we can use len(R) + len(P) <= best_size
                if len(R) + len(P) <= best_size:
                    return

        bk(R, P, X)
        
        # Max Independent Set in G is Max Clique in G_comp
        # Transversal is V \ MaxIS
        transversal = sorted(set(range(n)) - set(best_clique))
        
        return transversal

    results = []
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    for n, p, seed in instances:
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
