"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    best_colors = 17
    best_result = None
    
    # Start with the known valid solution
    best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    
    # Precompute all K4s
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            # Find edge index
            for idx, (i, j) in enumerate(edge_idx):
                if (i == a and j == b) or (i == b and j == a):
                    edges.append(idx)
                    break
        k4_edges.append(edges)
    
    def count_colors_in_k4(colors, edge_indices):
        return len(set(colors[e] for e in edge_indices))
    
    def is_valid(colors):
        for edges in k4_edges:
            if count_colors_in_k4(colors, edges) < 5:
                return False
        return True
    
    def num_violations(colors):
        count = 0
        for edges in k4_edges:
            if count_colors_in_k4(colors, edges) < 5:
                count += 1
        return count
    
    # Try local search with different numbers of colors
    for num_c in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            
            # Local search: try to fix violations
            for iteration in range(20000):
                if time.time() > deadline - 5:
                    break
                
                # Find a violated K4
                violated = False
                for edges in k4_edges:
                    if count_colors_in_k4(colors, edges) < 5:
                        violated = True
                        # Try changing one of the edges in this K4
                        for e in edges:
                            old_c = colors[e]
                            new_c = random.randint(0, num_c - 1)
                            colors[e] = new_c
                            if is_valid(colors):
                                break
                            colors[e] = old_c
                        break
            
            if is_valid(colors):
                used = len(set(colors))
                if used < best_colors:
                    best_colors = used
                    best_result = colors[:]
                break
    
    # Final fallback: return the known valid 17-color solution if nothing better
    if best_colors > 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_result
# EVOLVE-BLOCK-END
