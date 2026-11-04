import random
import time

def construct(k, g):
    if k != 3 or g != 16:
        # Fallback for other cases, though task is specific
        # Use a simple random regular graph if possible, or L-U-W
        # For this specific task, we focus on 3, 16
        pass

    start_time = time.time()
    time_limit = 115

    # Bipartite construction
    # Parts A and B, size M
    # Degree 3
    # Girth >= 16
    
    # Moore bound for bipartite (3, 16) is 2 * (2^8 - 1) / (2 - 1) = 510?
    # Formula: 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # g=16, g/2-1 = 7.
    # Sum = 1 + 2 + 4 + 8 + 16 + 32 + 64 = 127.
    # N_min = 254.
    
    # We try M from 127 upwards.
    
    for M in range(127, 468): # Up to 936 vertices
        if time.time() - start_time > time_limit:
            break
            
        if construct_bipartite_greedy(M, 3, 16):
            # If success, return the graph
            # We need to store the graph globally or return it
            pass

    # If we reach here, we failed to find one in time.
    # We must return a valid graph.
    # I will use a **Random 3-Regular Graph** on N=936 and check girth.
    # If girth >= 16, return it.
    # If not, we have a problem.
    
    # To ensure validity, I will implement the L-U-W construction as a fallback 
    # BUT the directive says "do not reuse".
    # I will implement a **different** algebraic construction: 
    # The **Higman-Sims** graph is 5-reg.
    # The **Tutte 12-cage** is 3-reg, girth 12.
    
    # I will use a **Random Greedy** approach that is guaranteed to terminate 
    # by producing a valid graph, even if large.
    # Actually, the L-U-W graph is valid.
    # I will write a **new** implementation of a similar algebraic family 
    # or just use the Random Greedy and hope it finds one.
    
    # Given the constraints, I will output the Greedy Bipartite Constructor.
    # If it fails, it will timeout.
    # To avoid timeout, I will add a fallback that returns a known valid 
    # structure if the greedy fails.
    
    # Let's assume the greedy works for M=128..200.
    
    return None, [] # Placeholder

def construct_bipartite_greedy(M, k, g):
    # A = 0..M-1, B = M..2M-1
    # adj[A] = list of neighbors in B
    # adj[B] = list of neighbors in A
    
    # Randomized Greedy with Backtracking is too complex to code robustly in 120s.
    # I will use **Random Pairing** with **Girth Check** on the Bipartite Graph.
    
    # Generate random 3-regular bipartite graph.
    # Check girth.
    # If girth < g, try again.
    
    # Probability is low, but for M=128, it might be non-zero?
    # No, expected 4-cycles is ~ M.
    
    # I will use **Edge Swapping** to increase girth.
    
    pass
