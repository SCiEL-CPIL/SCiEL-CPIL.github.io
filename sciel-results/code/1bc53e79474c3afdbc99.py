"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    # Primary approach: Maximum Weight Independent Set via NetworkX
    # Build complement graph (edge = coprime), find max weight clique
    try:
        import networkx as nx
        
        G = nx.Graph()
        G.add_nodes_from(range(1, N + 1))
        
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                if is_compatible(a, b):
                    G.add_edge(a, b)
        
        # Set node weights to the integer values
        for node in G.nodes:
            G.nodes[node]['weight'] = node
        
        clique = nx.max_weight_clique(G, weight='weight')
        
        if clique:
            return [int(x) for x in clique]
    except Exception:
        pass
    
    # Fallback: greedy + local search
    solution = [1]
    for a in range(N, 1, -1):
        if all(is_compatible(a, b) for b in solution):
            solution.append(a)
    
    current_sum = sum(solution)
    solution_set = set(solution)
    
    def can_replace(a, b):
        for c in solution_set:
            if c == b:
                continue
            if not is_compatible(a, c):
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            a = solution[i]
            for b in range(N, a, -1):
                if b in solution_set:
                    continue
                if can_replace(b, a):
                    solution_set.remove(a)
                    solution_set.add(b)
                    solution[i] = b
                    current_sum += b - a
                    improved = True
                    break
    
    return [int(x) for x in solution]
# EVOLVE-BLOCK-END
