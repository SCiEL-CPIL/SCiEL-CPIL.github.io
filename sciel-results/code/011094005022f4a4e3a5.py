import itertools
import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_edges = None
    best_count = -1

    # Precompute all triples
    triples = list(itertools.combinations(range(n), 3))
    
    def run_greedy():
        # Initialize complete graph
        adj_mask = [(1 << n) - 2 for _ in range(n)]
        for i in range(n):
            adj_mask[i] &= ~(1 << i)
            
        edges = set(itertools.combinations(range(n), 2))
        
        while True:
            # Check time
            if time.time() > deadline:
                return list(edges)
                
            # Find a K3,3 violation
            violation_edge = None
            for B in triples:
                b1, b2, b3 = B
                # Common neighbors of b1, b2, b3
                common = adj_mask[b1] & adj_mask[b2] & adj_mask[b3]
                # Count bits
                k = bin(common).count('1')
                if k >= 3:
                    # Find 3 vertices in common
                    A_list = [i for i in range(n) if (common >> i) & 1]
                    # Pick 3 random from A_list
                    A_sample = random.sample(A_list, 3)
                    # The K3,3 edges are between A_sample and B
                    # Pick a random edge from these 9
                    a_rand = random.choice(A_sample)
                    b_rand = random.choice(B)
                    violation_edge = (min(a_rand, b_rand), max(a_rand, b_rand))
                    break
            
            if violation_edge is None:
                break
                
            # Remove the edge
            u, v = violation_edge
            edges.remove((u, v))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)

        return list(edges)

    # Run multiple times
    while time.time() < deadline:
        res = run_greedy()
        if len(res) > best_count:
            best_count = len(res)
            best_edges = res

    if best_edges is None:
        return []
    return [[u, v] for u, v in best_edges]
