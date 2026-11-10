import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                pivot = max(P, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
                
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj):
        n = len(adj)
        cliques = bron_kerbosch(adj)
        if not cliques:
            return []
        
        # Map clique to index
        clique_id = {c: i for i, c in enumerate(cliques)}
        num_cliques = len(cliques)
        
        # Bitmasks for vertices
        # mask[v] has bit i set if v is in clique i
        masks = [0] * n
        for c in cliques:
            idx = clique_id[c]
            bit = 1 << idx
            for v in c:
                masks[v] |= bit
        
        all_mask = (1 << num_cliques) - 1
        
        best_solution = None
        
        # Randomized Greedy + Minimization
        # Number of restarts depends on time
        # Estimate time per restart: O(n * num_cliques * log(n)) roughly
        # Let's try 20 restarts initially, adjust if needed.
        
        for restart in range(100):
            if time.time() > deadline - 5.0:
                break
            
            rng = random.Random(restart + 42)
            uncovered = all_mask
            selected = []
            available = list(range(n))
            rng.shuffle(available) # Randomize tie-breaking order
            
            while uncovered:
                best_v = -1
                max_hits = -1
                # To speed up, we can sample or just iterate all. n is small (40).
                for v in available:
                    if not (masks[v] & uncovered):
                        continue
                    hits = (masks[v] & uncovered).bit_count()
                    # Add random noise to break ties and encourage diversity
                    noise = rng.random() * 0.1
                    score = hits + noise
                    if score > max_hits:
                        max_hits = score
                        best_v = v
                
                if best_v == -1:
                    # Should not happen if graph is valid, but safety
                    break
                
                selected.append(best_v)
                available.remove(best_v)
                uncovered &= ~masks[best_v]
            
            # Minimization Phase
            # Calculate cover counts
            cover_count = [0] * num_cliques
            for v in selected:
                m = masks[v]
                # Iterate set bits
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # Try to remove vertices
            # Random order for removal
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for v in sel_list:
                if v not in selected:
                    continue
                # Check if v is critical
                critical = False
                m = masks[v]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        critical = True
                        break
                    m ^= lsb
                
                if not critical:
                    # Remove v
                    selected.remove(v)
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
            
            if best_solution is None or len(selected) < len(best_solution):
                best_solution = list(selected)
                
            if not best_solution:
                break

        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = random_graph(n, p, seed)
        sol = solve_instance(adj)
        out.append(sol)
        
    return out
