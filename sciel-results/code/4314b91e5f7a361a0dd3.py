import time
import random
from itertools import product

def check_c6_in_set(S, p):
    """
    Check if there is a 6-cycle in the graph defined by differences in S over Z_p.
    A 6-cycle exists if there are d1..d6 in S such that sum(di) = 0 
    and partial sums are distinct (mod p).
    """
    # To avoid duplicates, we can fix d1 = min(S) or just iterate all.
    # 4^6 = 4096 combinations. Very fast.
    for d1 in S:
        for d2 in S:
            for d3 in S:
                for d4 in S:
                    for d5 in S:
                        for d6 in S:
                            if d1 + d2 + d3 + d4 + d5 + d6 == p: # sum is 0 mod p
                                # Check distinct vertices
                                # v0 = 0
                                # v1 = d1
                                # v2 = d1+d2
                                # ...
                                # v6 = 0
                                verts = [0]
                                s = 0
                                for d in [d1, d2, d3, d4, d5, d6]:
                                    s = (s + d) % p
                                    verts.append(s)
                                if len(set(verts)) == 7:
                                    return True
    return False

def try_partition(S0, S1, S2, S3, p):
    sets = [S0, S1, S2, S3]
    for S in sets:
        if check_c6_in_set(S, p):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, keep 20s margin
    
    # Target n = 17
    n = 17
    p = n
    
    # Partition of F_p^* into 4 sets
    # Residues: 1, 2, 4, 8, 9, 13, 15, 16
    # Non-Residues: 3, 5, 6, 7, 10, 11, 12, 14
    
    # Try a few specific partitions
    partitions = [
        # Partition 1: Powers of 2 vs others
        ([1, 2, 4, 8], [9, 13, 15, 16], [3, 5, 6, 7], [10, 11, 12, 14]),
        # Partition 2: Interleaved
        ([1, 4, 9, 16], [2, 8, 13, 3], [5, 10, 6, 11], [7, 12, 14, 15]),
        # Partition 3: Random-ish heuristic
        ([1, 3, 4, 10], [2, 5, 8, 11], [6, 7, 9, 12], [13, 14, 15, 16])
    ]
    
    best_n = 9
    best_colors = [min(j - i, n - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for S0, S1, S2, S3 in partitions:
        if time.time() > deadline:
            break
        if try_partition(S0, S1, S2, S3, p):
            # Construct colors
            colors = []
            map_diff_to_color = {}
            for d in S0: map_diff_to_color[d] = 0
            for d in S1: map_diff_to_color[d] = 1
            for d in S2: map_diff_to_color[d] = 2
            for d in S3: map_diff_to_color[d] = 3
            
            for i in range(p):
                for j in range(i + 1, p):
                    d = (j - i) % p
                    # Note: difference can be d or p-d. 
                    # In a directed sense, edge (i,j) has diff d. 
                    # But color of (i,j) must be consistent.
                    # Usually, we define color based on (j-i) mod p.
                    # Is color(i,j) == color(j,i)? Yes, undirected.
                    # But (j-i) and (i-j) = -d. 
                    # If S is not symmetric (S != -S), this is a problem.
                    # For a proper edge coloring of K_n, we need c(i,j) = c(j,i).
                    # So if d in S_c, then -d must also be in S_c? 
                    # Or we can define the color of (i,j) based on the set containing d OR -d?
                    # Standard construction: Partition F_p^* into sets S_c such that S_c = -S_c? 
                    # If p=17, -1 = 16. 
                    # If 1 is in S0, 16 must be in S0 for symmetry? 
                    # Let's check my partitions.
                    # P1: S0={1,2,4,8}. -S0={16,15,13,9} = S1. Not symmetric.
                    # So this construction defines a *directed* coloring or requires S_c to be symmetric.
                    # If S_c is not symmetric, we can't simply map d to color.
                    # We need c(i,j) = c(j,i).
                    # So we must have: if d in S_c, then -d in S_c.
                    
                    # Let's check if we can make symmetric sets.
                    # Size of each set is 4. 
                    # Pairs {d, -d}: 
                    # (1,16), (2,15), (3,14), (4,13), (5,12), (6,11), (7,10), (8,9)
                    # We have 8 pairs. We need 4 sets of size 4.
                    # Each set must contain 2 pairs (4 elements).
                    # So we partition the 8 pairs into 4 groups of 2 pairs.
                    
                    pairs = [(1,16), (2,15), (3,14), (4,13), (5,12), (6,11), (7,10), (8,9)]
                    
                    # Try a few partitions of pairs
                    pair_partitions = [
                        # Group 1: pairs 0,1. Group 2: pairs 2,3. Group 3: pairs 4,5. Group 4: pairs 6,7
                        ([(1,16), (2,15)], [(3,14), (4,13)], [(5,12), (6,11)], [(7,10), (8,9)]),
                        # Group 1: pairs 0,2. Group 2: pairs 1,3. Group 3: pairs 4,6. Group 4: pairs 5,7
                        ([(1,16), (3,14)], [(2,15), (4,13)], [(5,12), (7,10)], [(6,11), (8,9)]),
                        # Random mix
                        ([(1,16), (8,9)], [(2,15), (7,10)], [(3,14), (6,11)], [(4,13), (5,12)])
                    ]
                    
                    for pp in pair_partitions:
                        S0 = sorted([x for pair in pp[0] for x in pair])
                        S1 = sorted([x for pair in pp[1] for x in pair])
                        S2 = sorted([x for pair in pp[2] for x in pair])
                        S3 = sorted([x for pair in pp[3] for x in pair])
                        
                        if try_partition(S0, S1, S2, S3, p):
                            colors = []
                            map_diff_to_color = {}
                            for d in S0: map_diff_to_color[d] = 0
                            for d in S1: map_diff_to_color[d] = 1
                            for d in S2: map_diff_to_color[d] = 2
                            for d in S3: map_diff_to_color[d] = 3
                            
                            for i in range(p):
                                for j in range(i + 1, p):
                                    d = (j - i) % p
                                    colors.append(map_diff_to_color[d])
                            
                            return {'n': n, 'colors': colors}
    
    # If n=17 failed, try n=16? Or just return seed?
    # n=16 is not a prime, so field construction is harder.
    # We can try a backtracking search for n=16 if time permits.
    # But for now, if 17 fails, return the best known seed (n=9) or a slightly better one if found.
    # Actually, let's try a greedy construction for n=16 if 17 fails.
    
    return {'n': best_n, 'colors': best_colors}
