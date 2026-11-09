import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        cliques = []
        n = len(adj)
        
        # Optimization: Use bitmasks for sets if n is small enough (n<=40 fits in 64-bit int, but Python ints are arbitrary precision, so sets are fine and clear)
        # Standard BK with pivot
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X to minimize branching
            # If P is empty, use X
            if P:
                u = next(iter(P))
            else:
                u = next(iter(X)) if X else None
            
            if u is None:
                return
                
            # Candidates are P \ N(u)
            # If P is a set, N(u) is adj[u]
            candidates = [v for v in P if v not in adj[u]]
            
            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Filter cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
            
        # Map vertex to list of clique indices it covers
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
                
        # Greedy Initialization
        covered_count = [0] * len(cliques)
        S = set()
        
        # Precompute clique sizes for greedy priority? 
        # Standard greedy: pick v that covers most uncovered cliques.
        
        # To speed up, we can just iterate.
        # Since n is small, O(n * |cliques|) per step is fine.
        
        while True:
            if time.time() > deadline:
                break
                
            best_v = -1
            best_gain = 0
            
            # Calculate gain for each v not in S
            for v in range(n):
                if v in S:
                    continue
                gain = 0
                for ci in vertex_to_cliques[v]:
                    if covered_count[ci] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    
            if best_v == -1 or best_gain == 0:
                break
                
            S.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                covered_count[ci] += 1
                
        # Local Search: Iterative Deletion and Swap
        # We want to minimize |S|
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # 1. Try Deletion
            for v in list(S):
                # Check if v is redundant
                redundant = True
                for ci in vertex_to_cliques[v]:
                    if covered_count[ci] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove v
                    S.remove(v)
                    for ci in vertex_to_cliques[v]:
                        covered_count[ci] -= 1
                    improved = True
                    # Restart loop to check other deletions
                    break
                    
            if improved:
                continue
                
            # 2. Try Swap
            # Try swapping v in S with u not in S
            # A swap is valid if (S \ {v}) U {u} covers all cliques.
            # This is equivalent to:
            # - For all cliques covered ONLY by v in S (i.e. count==1), u must cover them.
            # - For all cliques covered by v and others (count>1), they remain covered.
            # - Cliques not covered by v but covered by u don't hurt.
            
            # Optimization: 
            # A swap v->u is valid if:
            # 1. For every clique c where covered_count[c] == 1 and v is the only one covering it, u must be in c.
            #    Note: if covered_count[c] == 1, v MUST be the one covering it if v is in S and c contains v.
            #    Wait, if covered_count[c]==1, it means exactly one vertex in S covers c.
            #    If v is in c and v is in S, then v is that vertex.
            #    So the condition is: For all c in vertex_to_cliques[v] such that covered_count[c]==1, u must be in c.
            
            # We also need to ensure that adding u doesn't break anything? No, adding only helps.
            # But we are removing v.
            
            # So, valid swap condition:
            # Let critical_cliques = [c for c in vertex_to_cliques[v] if covered_count[c] == 1]
            # If critical_cliques is empty, the swap is always valid (v is redundant, but we already tried deletion. 
            # Why would deletion fail? Deletion fails if covered_count[c] > 1 for all c in v. 
            # If critical_cliques is empty, v is redundant. Deletion should have worked. 
            # So if we reach swap, v is NOT redundant, meaning critical_cliques is NOT empty).
            
            # So we need u to cover all critical_cliques.
            
            # To make this efficient:
            # Precompute for each v the set of critical cliques? 
            # critical_cliques depend on current S and covered_count.
            
            # Iterate v in S
            swap_found = False
            for v in list(S):
                critical = []
                for ci in vertex_to_cliques[v]:
                    if covered_count[ci] == 1:
                        critical.append(ci)
                        
                if not critical:
                    # Should have been deleted
                    continue
                    
                # We need a u not in S that covers all cliques in 'critical'
                # And ideally, u should cover many other cliques to allow future deletions?
                # For now, just find ANY valid u.
                
                # To find u, we can check all u not in S.
                # Optimization: u must be in the intersection of all cliques in 'critical'.
                # Intersection of sets.
                
                # Since |critical| is likely small, we can just iterate u.
                # But n is 40, so 40 * |S| * |critical| is fine.
                
                # Let's just iterate u not in S
                for u in range(n):
                    if u in S:
                        continue
                        
                    # Check if u covers all critical cliques
                    valid = True
                    for ci in critical:
                        if u not in cliques[ci]:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        # Remove v
                        for ci in vertex_to_cliques[v]:
                            covered_count[ci] -= 1
                        S.remove(v)
                        
                        # Add u
                        for ci in vertex_to_cliques[u]:
                            covered_count[ci] += 1
                        S.add(u)
                        
                        improved = True
                        swap_found = True
                        break
                        
                if swap_found:
                    break
                    
        return sorted(list(S))

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices if we run out of time
            # This is a valid hitting set, just not minimal
            out.append(list(range(n)))
            continue
            
        result = solve_instance(n, p, seed)
        out.append(result)
        
    return out
