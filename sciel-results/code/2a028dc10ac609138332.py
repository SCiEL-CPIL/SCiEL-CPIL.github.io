import time
import sys

def construct():
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Start with a small n, try to grow
    # We know n=9 is valid. Let's try to start from n=10 or 12?
    # Starting from scratch is safer for correctness.
    
    best_n = 1
    best_colors = []
    
    # We will try to find a valid coloring for n = 10, 11, ... 20
    # If we fail for n, we return the best found (n-1)
    
    # To make it faster, we can use a greedy approach first.
    
    def check_c6(u, v, c, adj, n):
        # Check if adding edge (u,v) to color c creates a C6
        # adj[c] is a list of sets or bitmasks?
        # For n<=20, bitmasks are very fast.
        pass

    # ... implementation ...
