"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n = 20
    m = 10
    K = 20  # number of available colours
    
    left = list(range(m))
    right = list(range(m, 2*m))
    
    # Precompute all 4-cycles in K_{m,m}
    # A 4-cycle is (l1, l2, r1, r2) with l1!=l2, r1!=r2
    # Cycle: l1-r1-l2-r2-l1
    cycles = []
    for l1 in range(m):
        for l2 in range(l1+1, m):
            for r1 in range(m):
                for r2 in range(r1+1, m):
                    cycles.append((l1, l2, r1, r2))
    
    def check_colouring(colour_matrix):
        """Check if all 4-cycles are rainbow. colour_matrix[l][r] = color."""
        for l1, l2, r1, r2 in cycles:
            c1 = colour_matrix[l1][r1]
            c2 = colour_matrix[l2][r1]
            c3 = colour_matrix[l2][r2]
            c4 = colour_matrix[l1][r2]
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True
    
    # Strategy 1: Try to find valid K_{10,10} coloring via random + repair
    best_edges = None
    best_count = 0
    
    # First, try structured approaches
    # Try C[l][r] = (a*l + b*r) % K for various a, b
    found_valid = False
    for a in range(1, K):
        for b in range(1, K):
            if time.time() > deadline - 10:
                break
            colour_matrix = [[(a*l + b*r) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid:
            break
    
    if not found_valid:
        # Try C[l][r] = (l * r + c) % K
        for c in range(K):
            if time.time() > deadline - 10:
                break
            colour_matrix = [[(l * r + c) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid:
            pass
    
    if not found_valid:
        # Random search with repair on K_{10,10}
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colour_matrix = [[random.randint(0, K-1) for _ in range(m)] for _ in range(m)]
            
            # Repair loop
            for repair_iter in range(200):
                if time.time() > deadline - 5:
                    break
                # Find a violated cycle
                violated = None
                for l1, l2, r1, r2 in cycles:
                    c1 = colour_matrix[l1][r1]
                    c2 = colour_matrix[l2][r1]
                    c3 = colour_matrix[l2][r2]
                    c4 = colour_matrix[l1][r2]
                    if len({c1, c2, c3, c4}) < 4:
                        violated = (l1, l2, r1, r2)
                        break
                
                if violated is None:
                    # Valid!
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
                
                l1, l2, r1, r2 = violated
                # Change one of the four colors randomly
                edge_choice = random.randint(0, 3)
                if edge_choice == 0:
                    colour_matrix[l1][r1] = random.randint(0, K-1)
                elif edge_choice == 1:
                    colour_matrix[l2][r1] = random.randint(0, K-1)
                elif edge_choice == 2:
                    colour_matrix[l2][r2] = random.randint(0, K-1)
                else:
                    colour_matrix[l1][r2] = random.randint(0, K-1)
            
            if found_valid:
                break
    
    # Strategy 2: If K_{10,10} coloring not found, use C4-free greedy bipartite
    if best_edges is None:
        # Build C4-free bipartite graph greedily
        adjL = [set() for _ in range(m)]
        adjR = [set() for _ in range(m)]
        
        edges = []
        all_edges = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(all_edges)
        
        for i, j in all_edges:
            if time.time() > deadline - 2:
                break
            
            # Check if adding (i, j) creates a C4
            creates_c4 = False
            for i_prime in adjR[j]:
                if adjL[i] & adjL[i_prime]:
                    creates_c4 = True
                    break
            
            if not creates_c4:
                adjL[i].add(j)
                adjR[j].add(i)
                c = random.randint(0, K - 1)
                edges.append([left[i], right[j], c])
        
        best_edges = edges
        best_count = len(edges)
    
    return best_edges
# EVOLVE-BLOCK-END
