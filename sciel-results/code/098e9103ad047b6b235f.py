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
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        PX = P | X
        if PX:
            pivot = max(PX, key=lambda u: len(P & adj[u]))
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P.discard(v)
            X.add(v)
    
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
    vertex_to_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    num_cliques = len(cliques)
    
    def evaluate(S):
        """Returns (size, unhit_count) using hit counts."""
        if not S:
            return 0, num_cliques
        hit_count = [0] * num_cliques
        for v in S:
            for c_idx in vertex_to_cliques[v]:
                hit_count[c_idx] += 1
        unhit = sum(1 for h in hit_count if h == 0)
        return len(S), unhit

    # Initial solution: Greedy with random tie-breaking
    S = set()
    uncovered = set(range(num_cliques))
    while uncovered:
        best_v = None
        best_count = 0
        best_rand = float('inf')
        for v in range(n):
            if v in S:
                continue
            cnt = sum(1 for c_idx in vertex_to_cliques[v] if c_idx in uncovered)
            if cnt > best_count or (cnt == best_count and rng.random() < best_rand):
                best_count = cnt
                best_v = v
                best_rand = rng.random()
        if best_v is None or best_count == 0:
            break
        S.add(best_v)
        uncovered -= set(vertex_to_cliques[best_v])
        
    current_size, current_unhit = evaluate(S)
    best_S = set(S)
    best_size = current_size if current_unhit == 0 else float('inf')
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.998
    max_iter = 8000
    
    all_vertices = list(range(n))
    
    for _ in range(max_iter):
        if T <= T_min:
            break
        # Generate neighbor
        move_type = rng.random()
        
        if move_type < 0.35 and len(S) < n:
            # Add
            candidates = [v for v in all_vertices if v not in S]
            if not candidates: continue
            v = rng.choice(candidates)
            new_S = S | {v}
        elif move_type < 0.65 and S:
            # Remove
            v = rng.choice(list(S))
            new_S = S - {v}
        else:
            # Swap
            if not S:
                candidates = [v for v in all_vertices if v not in S]
                if not candidates: continue
                v = rng.choice(candidates)
                new_S = S | {v}
            else:
                v_in = rng.choice(list(S))
                candidates = [v for v in all_vertices if v not in S]
                if not candidates:
                    v = rng.choice(list(S))
                    new_S = S - {v}
                else:
                    v_out = rng.choice(candidates)
                    new_S = (S - {v_in}) | {v_out}
        
        new_size, new_unhit = evaluate(new_S)
        
        # Cost function with large penalty for infeasibility
        penalty = 1000
        current_cost = current_size + penalty * current_unhit
        new_cost = new_size + penalty * new_unhit
        
        delta = new_cost - current_cost
        
        if delta < 0 or (delta > 0 and T > 0 and rng.random() < math.exp(-delta / T)):
            S = new_S
            current_size, current_unhit = new_size, new_unhit
            
            if current_unhit == 0 and current_size < best_size:
                best_size = current_size
                best_S = set(S)
        
        T *= cooling
    
    # Final polish: Local search (remove redundant vertices)
    improved = True
    while improved:
        improved = False
        for v in list(best_S):
            trial = best_S - {v}
            s, u = evaluate(trial)
            if u == 0:
                best_S = trial
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
        for sa_seed in range(15):
            hit = _sa_hitting_set(cliques, n, seed=sa_seed)
            if len(hit) < best_size:
                best_size = len(hit)
                best_hit = hit
        
        results.append(best_hit)
    
    return results
# EVOLVE-BLOCK-END
