"""Program for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import math
import sys
import time

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of
    every nonempty consecutive block are distinct. Maximize length.
    
    Approach: Depth-First Search (DFS) with Backtracking and Branch and Bound.
    We explore the space of valid sequences by trying to append each candidate
    integer. If a candidate leads to a conflict, we skip it. If we reach a
    dead end, we backtrack. We keep track of the longest sequence found.
    Branch and bound prunes branches that cannot exceed the current best length.
    A time limit ensures the program finishes within the evaluation window.
    """
    
    best_length = 0
    best_sequence = []
    start_time = time.time()
    TIME_LIMIT = 55.0  # Seconds
    
    def try_add(values, products, x):
        """
        Checks if x can be appended to values.
        Returns the new set of products if valid, else None.
        """
        new_products = []
        prod = x
        # Generate products ending at x: x, x*last, x*last*second_last, ...
        for a in reversed(values):
            prod *= a
            new_products.append(prod)
        
        # Check for conflicts with existing products
        for p in new_products:
            if p in products:
                return None
        
        # If no conflict, return the union of existing and new products
        # Using union creates a new set, which is safe for backtracking
        return products.union(new_products)

    def dfs(values, products, last_index):
        nonlocal best_length, best_sequence
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
            
        current_length = len(values)
        
        # Update best if we found a longer sequence
        if current_length > best_length:
            best_length = current_length
            best_sequence = list(values)
            
        # Branch and Bound:
        # The maximum number of elements we can still add is (128 - last_index)
        # If even adding all of them doesn't beat the best, prune
        remaining = 128 - last_index
        if current_length + remaining <= best_length:
            return
        
        # Try adding each subsequent number
        for x in range(last_index + 1, 129):
            new_products = try_add(values, products, x)
            if new_products is not None:
                values.append(x)
                dfs(values, new_products, x)
                values.pop()
                
    # Start DFS with empty sequence
    dfs([], set(), 1)
    
    return best_sequence

# EVOLVE-BLOCK-END
