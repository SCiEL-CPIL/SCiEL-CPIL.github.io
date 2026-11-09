import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Known valid graph: 8 disjoint K4s, n=32, alpha=8
    # K4 is C5-free. Disjoint union of C5-free is C5-free.
    # alpha(K4) = 1. Sum of alphas = 8 < 9.
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33 graph with quick search
    # Given the difficulty and time limit, a random search is unlikely to succeed 
    # unless the graph is very sparse or structured.
    # We will skip the search to ensure we return a valid result within time.
    
    return {'n': best_n, 'edges': best_edges}
