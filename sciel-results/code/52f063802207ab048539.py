import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Fallback solution
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Precompute K4s
    # Vertices 0..15
    # K4s are combinations of 4 vertices
    # We need the 6 edges for each K4
    # Edge index for (i, j) with i < j:
    # idx = i * 15 - i * (i+1) // 2 + (j - i - 1)
    # Or simpler: map (i,j) to index
    edge_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_idx[(i,j)] = idx
            idx += 1
            
    k4s = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                        edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]
                    ]
                    k4s.append(edges)
                    
    def is_valid(colours):
        # colours is a list or array of length 120
        for edges in k4s:
            # Get colours of the 6 edges
            c0 = colours[edges[0]]
            c1 = colours[edges[1]]
            c2 = colours[edges[2]]
            c3 = colours[edges[3]]
            c4 = colours[edges[4]]
            c5 = colours[edges[5]]
            # Check if at least 5 distinct
            # Fast check:
            if c0 == c1 == c2 == c3 == c4 == c5: return False
            # Count distinct
            # Using set is easy but might be slow in tight loop
            # Manual check:
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
            if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
            
            if distinct < 5:
                return False
        return True

    # Search for better solutions
    for K in [15, 14, 13]:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid solution with K colours
        # Random restarts
        for restart in range(10000):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize random
            colours = np.random.randint(0, K, 120)
            
            # Check validity
            if is_valid(colours):
                return colours.tolist()
                
            # Local search
            # We can do a few steps of local search
            # But pure random restarts might be better if the space is rugged
            # Let's try a few steps of local search
            for step in range(100):
                if time.time() > deadline - 2.0:
                    break
                    
                if is_valid(colours):
                    return colours.tolist()
                    
                # Pick a random K4
                k4_idx = np.random.randint(len(k4s))
                edges = k4s[k4_idx]
                
                # Check if this K4 is violated
                c0 = colours[edges[0]]
                c1 = colours[edges[1]]
                c2 = colours[edges[2]]
                c3 = colours[edges[3]]
                c4 = colours[edges[4]]
                c5 = colours[edges[5]]
                
                distinct = 1
                if c1 != c0: distinct += 1
                if c2 != c0 and c2 != c1: distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                
                if distinct >= 5:
                    continue
                    
                # Change one of the edges in this K4
                # Pick a random edge
                e_idx = np.random.choice(edges)
                # Change to a random colour
                colours[e_idx] = np.random.randint(0, K)
                
    return fallback
