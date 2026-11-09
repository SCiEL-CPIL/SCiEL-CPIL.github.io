"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict
import math

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        if P:
            pivot_candidates = list(P)
        elif X:
            pivot_candidates = list(X)
        else:
            pivot_candidates = []
        
        if pivot_candidates:
            # Pick pivot with max degree into P
            pivot = max(pivot_candidates, key=lambda u: len(P & adj[u]))
            # Only recurse on P \ N(pivot)
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _sa_hitting_set(cliques, n, seed=42):
    """
    Finds a minimum hitting set using Simulated Annealing.
    State: set of vertices S.
    Cost: |S| + penalty * (# unhit cliques).
    """
    if not cliques:
        return []
    
    rng = random.Random(seed)
    
    # Precompute incidence: for each vertex, which cliques it hits
    vertex_to_cliques = defaultdict(set)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].add(i)
    
    num_cliques = len(cliques)
    
    def evaluate(S):
        """Returns (size, unhit_count)."""
        size = len(S)
        # Count unhit cliques
        # Efficiently: maintain a count of hits per clique? 
        # For simplicity in SA, we can just check.
        # But N is small.
        unhit = 0
        hit_flags = [False] * num_cliques
        for v in S:
            for c_idx in vertex_to_cliques[v]:
                hit_flags[c_idx] = True
        for i in range(num_cliques):
            if not hit_flags[i]:
                unhit += 1
        return size, unhit

    # Initial solution: Greedy
    S = set()
    uncovered = set(range(num_cliques))
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = 0
        for v in range(n):
            if v in S:
                continue
            cnt = len(uncovered & vertex_to_cliques[v])
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v is None:
            break
        S.add(best_v)
        uncovered -= vertex_to_cliques[best_v]
        
    current_size, current_unhit = evaluate(S)
    best_S = set(S)
    best_size, best_unhit = current_size, current_unhit
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 5000
    
    # Precompute list of all vertices
    all_vertices = list(range(n))
    
    while T > T_min:
        # Generate neighbor
        # 1. Add random vertex not in S
        # 2. Remove random vertex in S
        # 3. Swap
        move_type = rng.random()
        
        if move_type < 0.3 and len(S) < n:
            # Add
            candidates = [v for v in all_vertices if v not in S]
            if not candidates: continue
            v = rng.choice(candidates)
            new_S = S | {v}
            move_desc = "add"
        elif move_type < 0.6 and S:
            # Remove
            v = rng.choice(list(S))
            new_S = S - {v}
            move_desc = "remove"
        else:
            # Swap
            if not S:
                # Cannot swap if empty, add instead
                candidates = [v for v in all_vertices if v not in S]
                if not candidates: continue
                v = rng.choice(candidates)
                new_S = S | {v}
                move_desc = "add"
            else:
                v_in = rng.choice(list(S))
                candidates = [v for v in all_vertices if v not in S]
                if not candidates:
                    # Just remove?
                    v = rng.choice(list(S))
                    new_S = S - {v}
                    move_desc = "remove"
                else:
                    v_out = rng.choice(candidates)
                    new_S = (S - {v_in}) | {v_out}
                    move_desc = "swap"
        
        new_size, new_unhit = evaluate(new_S)
        
        # Cost function
        # We want to minimize size, but feasibility is paramount.
        # Penalty for unhit cliques must be large.
        penalty = 100
        current_cost = current_size + penalty * current_unhit
        new_cost = new_size + penalty * new_unhit
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            S = new_S
            current_size, current_unhit = new_size, new_unhit
            
            if current_unhit == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_S = set(S)
        # else: reject move, S remains unchanged
        
        T *= cooling
        
    # Final polish: Local search (remove redundant vertices)
    improved = True
    while improved:
        improved = False
        for v in list(best_S):
            trial = best_S - {v}
            # Check if trial hits all cliques
            ok = True
            # Efficient check:
            # We can just re-evaluate
            s, u = evaluate(trial)
            if u == 0:
                best_S = trial
                best_size = s
                improved = True
    
    return sorted(best_S)

def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = _generate_graph(n, p, seed)
        all_cliques = _bron_kerbosch(adj, n)
        # Filter to cliques with at least 2 vertices
        cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Use Simulated Annealing with multiple restarts to find better solution
        best_hit = None
        best_size = float('inf')
        
        # Run SA multiple times with different seeds
        for sa_seed in range(10):
            hit = _sa_hitting_set(cliques, n, seed=sa_seed)
            if len(hit) < best_size:
                best_size = len(hit)
                best_hit = hit
        
        results.append(best_hit)
    
    return results
# EVOLVE-BLOCK-END
