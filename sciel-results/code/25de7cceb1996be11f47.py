"""Structural change: Simulated Annealing with bitmask representation and fast coverage updates."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

_DEADLINE = 145.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            # Fallback: return all touched vertices for remaining instances
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance_sa(n, p, seed)
        out.append(sol)
    return out

def _solve_instance_sa(n, p, seed):
    # 1. Build Graph
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # 2. Enumerate Maximal Cliques (Bron-Kerbosch)
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot
        if P:
            # Choose pivot with max degree in P|X
            max_count = -1
            pivot = None
            PX = P | X
            for v in P:
                c = len(adj[v] & PX)
                if c > max_count:
                    max_count = c
                    pivot = v
            candidates = P - adj[pivot] if pivot is not None else set(P)
        else:
            candidates = set()
            
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    bronk(set(), set(range(n)), set())
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return []
        
    m = len(cliques)
    
    # 3. Precompute Vertex-Clique Incidence
    # vertex_cliques[v] = list of indices of cliques containing v
    vertex_cliques = [[] for _ in range(n)]
    for i, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].append(i)
            
    # 4. Initial Solution: Greedy Set Cover
    # We use a bitmask for the selected set
    # For fast updates, we maintain coverage counts
    def greedy_init():
        uncovered = [1] * m # 1 if uncovered, 0 if covered
        covered_count = 0
        selected_mask = 0
        selected_set = set()
        
        # Precompute for greedy: for each vertex, which cliques it covers
        # We just iterate
        remaining_cliques = set(range(m))
        
        while remaining_cliques:
            best_v = -1
            best_gain = 0
            for v in range(n):
                if selected_mask & (1 << v):
                    continue
                gain = 0
                for ci in vertex_cliques[v]:
                    if ci in remaining_cliques:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            selected_set.add(best_v)
            for ci in vertex_cliques[best_v]:
                if ci in remaining_cliques:
                    remaining_cliques.discard(ci)
                    
        # Ensure validity (should be valid by construction)
        return selected_mask, selected_set, [1 if i in remaining_cliques else 0 for i in range(m)]
        
    init_mask, init_set, init_cov = greedy_init()
    # init_cov: 1 if uncovered, 0 if covered. Wait, my greedy loop discards from remaining_cliques.
    # So if i in remaining_cliques, it is uncovered.
    # Let's re-verify coverage.
    coverage_counts = [0] * m
    for v in init_set:
        for ci in vertex_cliques[v]:
            coverage_counts[ci] += 1
            
    # Uncovered cliques are those with count 0
    uncovered_count = sum(1 for c in coverage_counts if c == 0)
    
    # 5. Simulated Annealing
    # Energy = alpha * uncovered_count + beta * popcount(mask)
    # We want to minimize energy.
    # Start with high alpha to ensure validity, then reduce?
    # Or just use a large penalty for uncovered.
    
    best_mask = init_mask
    best_energy = uncovered_count * 1000 + bin(init_mask).count('1')
    current_mask = init_mask
    current_coverage = coverage_counts[:]
    current_uncovered = uncovered_count
    current_size = bin(current_mask).count('1')
    
    T = 50.0 # Initial temperature
    cooling = 0.995
    t_start = time.time()
    
    while time.time() - t_start < 10.0: # 10s per instance
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
            
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        bit = 1 << v
        
        if current_mask & bit:
            # Remove v
            # Update coverage
            delta_uncovered = 0
            for ci in vertex_cliques[v]:
                current_coverage[ci] -= 1
                if current_coverage[ci] == 0:
                    delta_uncovered += 1
            new_uncovered = current_uncovered + delta_uncovered
            new_size = current_size - 1
            new_mask = current_mask ^ bit
            new_energy = new_uncovered * 1000 + new_size
        else:
            # Add v
            delta_uncovered = 0
            for ci in vertex_cliques[v]:
                if current_coverage[ci] == 0:
                    delta_uncovered -= 1 # Becomes covered
                current_coverage[ci] += 1
            new_uncovered = current_uncovered + delta_uncovered
            new_size = current_size + 1
            new_mask = current_mask | bit
            new_energy = new_uncovered * 1000 + new_size
            
        # Acceptance
        delta_E = new_energy - (current_uncovered * 1000 + current_size)
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept
            current_mask = new_mask
            current_uncovered = new_uncovered
            current_size = new_size
            
            if new_uncovered == 0 and new_size < bin(best_mask).count('1'):
                best_mask = new_mask
                best_energy = new_size
                
        T *= cooling
        
    # Final local search on best_mask to remove any redundant vertices
    # Convert best_mask to set
    best_set = {v for v in range(n) if best_mask & (1 << v)}
    
    # Redundancy removal
    changed = True
    while changed:
        changed = False
        for v in list(best_set):
            # Check if v is redundant
            # A vertex v is redundant if for every clique c containing v, there is another vertex in best_set \ {v} in c.
            # Equivalent: if we remove v, do we uncover any clique?
            # We can check by temporarily removing v and checking coverage.
            # Or simpler: for each clique c containing v, check if (c & (best_set - {v})) is non-empty.
            is_redundant = True
            for ci in vertex_cliques[v]:
                c = cliques[ci]
                # Check if any other vertex in c is in best_set
                found_other = False
                for u in c:
                    if u != v and (best_mask & (1 << u)):
                        found_other = True
                        break
                if not found_other:
                    is_redundant = False
                    break
            if is_redundant:
                best_set.discard(v)
                best_mask &= ~(1 << v)
                changed = True
                
    return sorted(best_set)

# EVOLVE-BLOCK-END
