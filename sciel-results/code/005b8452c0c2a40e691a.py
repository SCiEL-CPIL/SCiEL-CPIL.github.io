"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16,
    in lexicographic order.
    
    Uses vertex-labeling approach: assign each vertex a label from a small set,
    then derive edge colors from vertex labels using a nonlinear function.
    This allows finding structured solutions with fewer colors.
    """
    n = 16
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for 120s limit
    
    # Precompute all K4s (sets of 4 vertices)
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(edges)
    
    # Precompute edge_to_k4s for fast validity checking
    edge_to_k4s = {}
    for idx, edges in enumerate(k4s):
        for (u, v) in edges:
            if (u, v) not in edge_to_k4s:
                edge_to_k4s[(u, v)] = []
            edge_to_k4s[(u, v)].append(idx)
            if (v, u) not in edge_to_k4s:
                edge_to_k4s[(v, u)] = []
            edge_to_k4s[(v, u)].append(idx)
    
    # Function to check if a coloring is valid
    def is_valid(colors):
        for edges in k4s:
            c_set = set()
            for (u, v) in edges:
                c_set.add(colors[u][v])
            if len(c_set) < 5:
                return False
        return True
    
    # Function to count violations
    def count_violations(colors):
        violations = 0
        for edges in k4s:
            c_set = set()
            for (u, v) in edges:
                c_set.add(colors[u][v])
            if len(c_set) < 5:
                violations += 1
        return violations
    
    # Function to get number of distinct colours used
    def get_num_colours(colors):
        used = set()
        for i in range(n):
            for j in range(i+1, n):
                used.add(colors[i][j])
        return len(used)
    
    # Strategy: Try different vertex labeling schemes
    # Each vertex gets a label from a small set (e.g., 0..m-1)
    # Edge color = f(label[i], label[j]) for some function f
    
    # Try several labeling schemes and find the one with minimum colors
    
    best_colors = None
    best_num_colours = 100
    
    # Fallback: the known valid 17-colour construction
    def make_fallback():
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = (i + j) % 17
                colors[j][i] = colors[i][j]
        return colors
    
    # Try vertex labeling with labels from 0..m-1
    # Edge color = (label[i] * label[j]) % p for prime p
    # Or edge color = (label[i] + label[j]) % p
    # Or edge color = (label[i] ^ label[j])  # XOR
    
    # We'll try a few different functions and label sizes
    
    # First, try the simple (i+j)%17 construction as baseline
    colors_fallback = make_fallback()
    if is_valid(colors_fallback):
        best_colors = colors_fallback
        best_num_colours = get_num_colours(colors_fallback)
    
    # Now try vertex labeling approaches
    # Label vertices with values from 0..m-1
    # Use different functions to derive edge colors
    
    # Function 1: (label[i] + label[j]) % p
    # Function 2: (label[i] * label[j]) % p  
    # Function 3: (label[i] ^ label[j])  # XOR
    
    # We'll search over label assignments and function parameters
    
    # For efficiency, we'll use random restarts with local search on vertex labels
    
    # Define a function to compute edge colors from vertex labels
    def compute_edge_colors(labels, func_type, param):
        """
        labels: list of vertex labels
        func_type: 'sum', 'prod', 'xor'
        param: parameter for the function (e.g., modulus)
        """
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                li, lj = labels[i], labels[j]
                if func_type == 'sum':
                    c = (li + lj) % param
                elif func_type == 'prod':
                    c = (li * lj) % param
                elif func_type == 'xor':
                    c = li ^ lj
                else:
                    c = (li + lj) % param
                colors[i][j] = c
                colors[j][i] = c
        return colors
    
    # Try different label sizes and functions
    # We want to minimize the number of distinct edge colors
    
    # Strategy: 
    # 1. Try small label sets (m=3,4,5,...)
    # 2. For each, try different functions and parameters
    # 3. Use local search to optimize vertex labels
    
    # Let's try a systematic search over label assignments
    
    # For each vertex, assign a label from 0..m-1
    # We'll use simulated annealing or hill climbing on the vertex labels
    
    # Objective: minimize number of distinct edge colors, subject to validity
    
    # Let's implement a local search on vertex labels
    
    def vertex_local_search(initial_labels, func_type, param, max_iterations=1000):
        """
        Local search on vertex labels to minimize number of distinct edge colors
        while maintaining validity.
        """
        labels = initial_labels[:]
        colors = compute_edge_colors(labels, func_type, param)
        
        if not is_valid(colors):
            return None, None
        
        current_num_colours = get_num_colours(colors)
        best_labels = labels[:]
        best_colors = [row[:] for row in colors]
        best_num = current_num_colours
        
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout - 10:
                break
            
            # Pick a random vertex and change its label
            v = random.randint(0, n - 1)
            old_label = labels[v]
            new_label = random.randint(0, param - 1)
            
            if new_label == old_label:
                continue
            
            labels[v] = new_label
            colors = compute_edge_colors(labels, func_type, param)
            
            # Check validity
            if not is_valid(colors):
                labels[v] = old_label
                continue
            
            new_num_colours = get_num_colours(colors)
            
            # Accept if it reduces the number of colors
            if new_num_colours < current_num_colours:
                current_num_colours = new_num_colours
                if new_num_colours < best_num:
                    best_num = new_num_colours
                    best_labels = labels[:]
                    best_colors = [row[:] for row in colors]
            else:
                # Reject (or accept with some probability for simulated annealing)
                labels[v] = old_label
        
        return best_labels, best_colors
    
    # Try different configurations
    # We'll try various label ranges and functions
    
    # Configuration 1: labels from 0..4 (5 labels), sum mod 13
    # Configuration 2: labels from 0..5 (6 labels), prod mod 13
    # etc.
    
    configs = [
        ('sum', 13),
        ('sum', 17),
        ('prod', 13),
        ('prod', 17),
        ('xor', 16),
        ('xor', 32),
    ]
    
    # For each config, try different initial label assignments
    for func_type, param in configs:
        if time.time() - start_time > timeout - 5:
            break
        
        # Try several random initial assignments
        for trial in range(50):
            if time.time() - start_time > timeout - 5:
                break
            
            # Random initial labels
            initial_labels = [random.randint(0, param - 1) for _ in range(n)]
            
            # Run local search
            best_labels, best_colors_found = vertex_local_search(initial_labels, func_type, param, max_iterations=500)
            
            if best_colors_found is not None:
                num_colours = get_num_colours(best_colors_found)
                if num_colours < best_num_colours:
                    best_num_colours = num_colours
                    best_colors = [row[:] for row in best_colors_found]
    
    # If we didn't find anything better, use the fallback
    if best_colors is None:
        best_colors = make_fallback()
    
    # Final verification
    if not is_valid(best_colors):
        best_colors = make_fallback()
    
    # Flatten to list in lexicographic order
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(best_colors[i][j])
            
    return result

# EVOLVE-BLOCK-END
