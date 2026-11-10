import numpy as np
from itertools import combinations, permutations

# Precompute Q3 structure
# Vertices 0..7
# Edges:
Q3_edges = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# For the check "Does G[S] + (u,v) contain Q3 using (u,v)?",
# we assume u maps to 0 and v maps to 1.
# We need to map T (6 vertices) to {2,3,4,5,6,7}.
# Required edges in G[S] (excluding u-v which is being added):
# 0-2, 0-4
# 1-3, 1-5
# 2-3, 2-6
# 3-7
# 4-5, 4-6
# 5-7
# 6-7

# Let's define the constraints on the mapping of T to {2,3,4,5,6,7}
# Let m be a mapping from {2,3,4,5,6,7} to T.
# Conditions:
# 1. m[2] is neighbor of u in G
# 2. m[4] is neighbor of u in G
# 3. m[3] is neighbor of v in G
# 4. m[5] is neighbor of v in G
# 5. m[2] connected to m[3]
# 6. m[2] connected to m[6]
# 7. m[3] connected to m[7]
# 8. m[4] connected to m[5]
# 9. m[4] connected to m[6]
# 10. m[5] connected to m[7]
# 11. m[6] connected to m[7]

def construct():
    n = 16
    # Initialize with K3,13
    # Vertices 0,1,2 are in A, 3..15 in B
    A = np.zeros((n, n), dtype=bool)
    for a in range(3):
        for b in range(3, 16):
            A[a, b] = True
            A[b, a] = True
            
    # List of missing edges
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not A[i, j]:
                missing_edges.append((i, j))
                
    # Shuffle to randomize
    import random
    random.seed(42)
    random.shuffle(missing_edges)
    
    # Function to check if adding edge (u, v) creates a Q3
    def can_add(u, v):
        # Get neighbors of u and v in current graph A
        # Note: u and v are NOT connected in A
        nu = np.where(A[u])[0]
        nv = np.where(A[v])[0]
        
        # Candidates for T are vertices not u, v
        all_vertices = [x for x in range(n) if x != u and x != v]
        
        # Iterate over all 6-subsets of all_vertices
        # Optimization: Pre-filter subsets based on degrees?
        # For a subset T to work, it must contain at least 2 neighbors of u and 2 neighbors of v.
        
        for T in combinations(all_vertices, 6):
            T_set = set(T)
            
            # Check if T has enough neighbors of u and v
            nu_in_T = [x for x in T if x in nu]
            nv_in_T = [x for x in T if x in nv]
            
            if len(nu_in_T) < 2 or len(nv_in_T) < 2:
                continue
                
            # We need to assign T to {2,3,4,5,6,7}
            # m[2], m[4] must be in nu_in_T
            # m[3], m[5] must be in nv_in_T
            
            # Let's pick candidates for 2 and 4 from nu_in_T
            # And 3 and 5 from nv_in_T
            # The remaining 2 vertices in T will be 6 and 7.
            
            # To avoid duplicates, we can fix the order?
            # Since 2 and 4 are symmetric in the constraints?
            # 2 is connected to 0,3,6. 4 is connected to 0,5,6.
            # 3 is connected to 1,2,7. 5 is connected to 1,4,7.
            # 6 is connected to 2,4,7. 7 is connected to 3,5,6.
            
            # Let's iterate over pairs for (2,4) and (3,5)
            for pair_24 in combinations(nu_in_T, 2):
                for pair_35 in combinations(nv_in_T, 2):
                    # Check if pair_24 and pair_35 are disjoint
                    if set(pair_24).intersection(pair_35):
                        continue
                        
                    # Determine remaining vertices for 6,7
                    used = set(pair_24).union(pair_35)
                    remaining = [x for x in T if x not in used]
                    if len(remaining) != 2:
                        continue
                        
                    # Assign 2, 4
                    # Two possibilities for (2,4) order?
                    # 2 and 4 are not fully symmetric because 2 connects to 3, 4 connects to 5.
                    # But 3 is in pair_35, 5 is in pair_35.
                    # Let's try both permutations for 2,4 and 3,5?
                    # Actually, we can just check the constraints for both assignments.
                    
                    # Let c2, c4 be the two vertices in pair_24
                    # Let c3, c5 be the two vertices in pair_35
                    # Let c6, c7 be the two vertices in remaining
                    
                    # We need to check if there is an assignment of {c2,c4} to {2,4}, {c3,c5} to {3,5}, {c6,c7} to {6,7}
                    # such that all edges exist.
                    
                    # There are 2 * 2 * 2 = 8 permutations.
                    for p24 in permutations(pair_24):
                        for p35 in permutations(pair_35):
                            for p67 in permutations(remaining):
                                # Mapping:
                                # 0->u, 1->v
                                # 2->p24[0], 4->p24[1]
                                # 3->p35[0], 5->p35[1]
                                # 6->p67[0], 7->p67[1]
                                
                                m = {
                                    0: u, 1: v,
                                    2: p24[0], 4: p24[1],
                                    3: p35[0], 5: p35[1],
                                    6: p67[0], 7: p67[1]
                                }
                                
                                # Check edges
                                valid = True
                                # Edges involving 0 (u)
                                if not A[u, m[2]]: valid = False; break
                                if not A[u, m[4]]: valid = False; break
                                # Edges involving 1 (v)
                                if not A[v, m[3]]: valid = False; break
                                if not A[v, m[5]]: valid = False; break
                                # Other edges
                                if not A[m[2], m[3]]: valid = False; break
                                if not A[m[2], m[6]]: valid = False; break
                                if not A[m[3], m[7]]: valid = False; break
                                if not A[m[4], m[5]]: valid = False; break
                                if not A[m[4], m[6]]: valid = False; break
                                if not A[m[5], m[7]]: valid = False; break
                                if not A[m[6], m[7]]: valid = False; break
                                
                                if valid:
                                    return False # Q3 found, cannot add
                                    
            # If we finished loops without returning False, this T is safe
            # But we need to check ALL T. If ANY T creates a Q3, we cannot add.
            # Wait, the logic above returns False immediately if a Q3 is found.
            # If we finish all T without finding a Q3, it is safe.
            
        return True

    # Greedy addition
    added_edges = 0
    # We iterate through missing edges. If we add one, we update A.
    # We should probably re-shuffle or just iterate.
    # Since adding an edge might make other edges invalid, we process sequentially.
    
    # To be more effective, we can use a queue or list and filter.
    # But simple iteration is okay for now.
    
    current_missing = list(missing_edges)
    # We might want to try multiple times with different shuffles?
    # Given time limit, let's stick to one good pass.
    
    for u, v in current_missing:
        if can_add(u, v):
            A[u, v] = True
            A[v, u] = True
            added_edges += 1
            
    # Convert back to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j]:
                edges.append([i, j])
                
    return edges
