import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P ∪ X with max |P ∩ N(u)|
            pivot = None
            max_inter = -1
            for u in list(P) + list(X):
                inter = len(P & adj[u])
                if inter > max_inter:
                    max_inter = inter
                    pivot = u
            
            if pivot is None:
                for v in list(P):
                    R.add(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.remove(v)
                    P.remove(v)
                    X.add(v)
            else:
                # Only consider v in P not adjacent to pivot
                for v in list(P - adj[pivot]):
                    R.add(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.remove(v)
                    P.remove(v)
                    X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex hitting most remaining cliques."""
        if not cliques:
            return set()
        
        # Map each vertex to set of clique indices it belongs to
        vertex_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        remaining = set(range(len(cliques)))
        hit = set()
        
        while remaining:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in hit:
                    continue
                count = len(vertex_to_cliques[v] & remaining)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count <= 0:
                break
            
            hit.add(best_v)
            remaining -= vertex_to_cliques[best_v]
        
        return hit

    def local_search_minimize(hit, cliques, n, time_limit):
        """Local search: try deletions and swaps to minimize hitting set."""
        hit = set(hit)
        cliques_list = list(cliques)
        num_cliques = len(cliques_list)
        
        # Precompute: for each clique, which vertices in hit cover it
        # For efficiency, track for each vertex how many cliques it uniquely covers
        def is_valid(h):
            h_set = set(h)
            for clique in cliques_list:
                if not (clique & h_set):
                    return False
            return True
        
        def count_covered_by(v, h_set):
            """Count cliques that are hit ONLY by v among h_set."""
            count = 0
            for clique in cliques_list:
                if v in clique:
                    # Check if any other vertex in h_set is in this clique
                    others = clique & h_set
                    others.discard(v)
                    if not others:
                        count += 1
            return count
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 10:
            changed = False
            
            # Phase 1: Deletions
            for v in list(hit):
                hit_without = hit - {v}
                if is_valid(hit_without):
                    hit = hit_without
                    changed = True
            
            # Phase 2: Swaps - try replacing one vertex with another
            if (time.time() - start_time) < deadline - 10:
                for v in list(hit):
                    for u in range(n):
                        if u in hit or u == v:
                            continue
                        new_hit = (hit - {v}) | {u}
                        if len(new_hit) < len(hit) and is_valid(new_hit):
                            hit = new_hit
                            changed = True
                            break
                    if changed:
                        break
        
        return hit

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        # Filter to size >= 2 (already done in BK, but be safe)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Greedy hitting set
        hit = greedy_hitting_set(cliques, n)
        
        # Local search to minimize
        hit = local_search_minimize(hit, cliques, n, time_budget)
        
        return sorted(hit)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 5:
            # Not enough time, just use greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            cliques = [c for c in cliques if len(c) >= 2]
            if not cliques:
                sol = []
            else:
                sol = sorted(greedy_hitting_set(cliques, n))
        else:
            # Allocate time proportionally, leave buffer
            per_instance_time = (time_left - 5) / (3 - i)
            sol = solve_instance(n, p, seed, per_instance_time)
        
        results.append(sol)

    return results
