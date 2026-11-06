import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Try n from 34 down to 1
    # We know n=34 is possible.
    for n in range(34, 0, -1):
        if time.time() > deadline:
            break
            
        # Precompute triple indices for this n
        # Map (a,b,c) to index
        triple_to_idx = {}
        idx = 0
        triples_list = []
        for a, b, c in combinations(range(n), 3):
            triple_to_idx[(a, b, c)] = idx
            triples_list.append((a, b, c))
            idx += 1
        num_triples = idx
        
        # Precompute K4 and K5 violation structures
        # K4: 4 vertices, 4 triples
        k4_indices = []
        for quad in combinations(range(n), 4):
            inds = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        inds.append(triple_to_idx[(quad[i], quad[j], quad[k])])
            k4_indices.append(inds)
        k4_indices = np.array(k4_indices)
        
        # K5: 5 vertices, 10 triples
        k5_indices = []
        for quint in combinations(range(n), 5):
            inds = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        inds.append(triple_to_idx[(quint[i], quint[j], quint[k])])
            k5_indices.append(inds)
        k5_indices = np.array(k5_indices)
        
        # If no K4 or K5 possible (n small), it's trivially valid
        if n < 4:
             return {'n': n, 'colors': [0]*num_triples}
        if n < 5:
             # Only K4 constraint
             pass # Handled by search below or trivial if n<4

        # Initialize coloring
        # Heuristic: Use modular arithmetic as a starting point
        # Color 0 (red) if (a+b+c) % 3 == 0
        colors = np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples_list], dtype=np.uint8)
        
        best_colors = colors.copy()
        best_violations = float('inf')
        
        # Local Search
        # We will try to minimize violations = count_red_K4 + count_blue_K5
        # We use vertex flips: flip all triples containing vertex v
        
        def count_violations(c):
            if len(k4_indices) == 0:
                red_v = 0
            else:
                red_v = np.all(c[k4_indices] == 0, axis=1).sum()
            if len(k5_indices) == 0:
                blue_v = 0
            else:
                blue_v = np.all(c[k5_indices] == 1, axis=1).sum()
            return red_v + blue_v
        
        current_violations = count_violations(colors)
        if current_violations == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        best_colors = colors.copy()
        best_violations = current_violations
        
        # Simulated Annealing / Local Search
        # Time budget per n
        time_per_n = 5.0
        start_n_time = time.time()
        
        while time.time() - start_n_time < time_per_n:
            # Pick a random vertex
            v = np.random.randint(n)
            
            # Identify all triples containing v
            # We can precompute this for speed
            # But for simplicity, let's just flip a random triple first to see if it works?
            # No, vertex flip is better.
            
            # Find indices of triples containing v
            # This is O(num_triples) which is ~6000. Acceptable.
            flip_indices = []
            for i, (a,b,c) in enumerate(triples_list):
                if a == v or b == v or c == v:
                    flip_indices.append(i)
            
            if not flip_indices:
                continue
                
            # Flip colors
            colors[flip_indices] = 1 - colors[flip_indices]
            
            # Count violations
            new_violations = count_violations(colors)
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                if new_violations == 0:
                    return {'n': n, 'colors': colors.tolist()}
            else:
                # Revert if not better (simple hill climbing)
                # Or accept with probability?
                # For now, simple hill climbing to avoid getting stuck in bad states
                colors[flip_indices] = 1 - colors[flip_indices]
                
        # If we didn't find 0 violations for this n, try smaller n?
        # Or return the best found for this n? 
        # The task requires a VALID coloring.
        # If we don't find a valid one for n, we must drop n.
        
    # Fallback: Return a small valid n
    # n=4 is valid if we color all blue (no red K4) or all red (no blue K5, but red K4 exists).
    # If n=4, we must avoid red K4. So at least one triple must be blue.
    # If n=5, we must avoid blue K5. So at least one triple must be red.
    # And avoid red K4.
    
    # Let's just return n=1 if all else fails
    return {'n': 1, 'colors': []}
