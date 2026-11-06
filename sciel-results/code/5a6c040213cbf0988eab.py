import random
import numpy as np
import itertools
import time

def construct():
    N8 = 8
    N16 = 16
    best_edges_8 = 0
    best_graph_8 = None
    start_time = time.time()
    time_limit = 100
    
    def contains_q3_8(adj):
        """
        Check if graph on 8 vertices contains Q3.
        Returns True if found.
        """
        vertices = list(range(8))
        four_sets = list(itertools.combinations(vertices, 4))
        # Map set to index?
        # We need disjoint A, B.
        # If A is a 4-set, B is its complement.
        # There are 70 such pairs (each pair counted twice? No, A and B are ordered in our check?
        # Q3 is bipartite. (A, B) and (B, A) are the same subgraph.
        # We just need to check one direction.
        
        for A in four_sets:
            B = tuple(v for v in vertices if v not in A)
            # Check if non-edges between A and B contain a perfect matching
            non_edge_mat = np.zeros((4, 4), dtype=bool)
            for i in range(4):
                for j in range(4):
                    if not adj[A[i]][B[j]]:
                        non_edge_mat[i][j] = True
                        
            # Check perfect matching in non_edge_mat
            # Brute force 24 perms
            for perm in itertools.permutations(range(4)):
                match = True
                for k in range(4):
                    if not non_edge_mat[k][perm[k]]:
                        match = False
                        break
                if match:
                    return True
        return False

    def optimize_8():
        best_e = 0
        best_g = None
        random.seed(1)
        for restart in range(200):
            if time.time() - start_time > time_limit:
                break
            # Start with K8
            adj = np.ones((8, 8), dtype=bool)
            np.fill_diagonal(adj, False)
            
            # Remove random edges until Q3-free?
            # Or start with random p?
            # Let's start with K8 and remove edges greedily.
            
            # To speed up, we can just try random removals.
            # But K8 has many Q3s.
            
            # Let's use the same local search logic.
            # Find Q3, remove edge.
            
            # To find Q3 faster, we can cache the list of Q3s?
            # On 8 vertices, finding a Q3 is very fast.
            
            while True:
                if time.time() - start_time > time_limit:
                    break
                # Find a Q3
                q3_found = False
                vertices = list(range(8))
                four_sets = list(itertools.combinations(vertices, 4))
                for A in four_sets:
                    B = tuple(v for v in vertices if v not in A)
                    non_edge_mat = np.zeros((4, 4), dtype=bool)
                    for i in range(4):
                        for j in range(4):
                            if not adj[A[i]][B[j]]:
                                non_edge_mat[i][j] = True
                            
                    for perm in itertools.permutations(range(4)):
                        match = True
                        for k in range(4):
                            if not non_edge_mat[k][perm[k]]:
                                match = False
                                break
                        if match:
                            q3_found = True
                            # Found Q3 defined by A, B, perm
                            # Remove an edge
                            # Edges in Q3 are (A[i], B[j]) for i != perm[j]?
                            # No, non-edges are (A[k], B[perm[k]]).
                            # So edges are all other pairs.
                            edge_candidates = []
                            for i in range(4):
                                for j in range(4):
                                    if i != perm[j]: # Wait, perm is index of B for non-edge at i?
                                         # My perm definition: non_edge_mat[k][perm[k]] is True.
                                         # So (A[k], B[perm[k]]) is a NON-edge.
                                         # So (A[i], B[j]) is an EDGE if i != perm[j]?
                                         # No.
                                         # non_edge_mat[i][j] is True if (A[i], B[j]) is NOT an edge.
                                         # The matching is on the non-edges.
                                         # So (A[k], B[perm[k]]) is a non-edge.
                                         # All other pairs (A[i], B[j]) with i != perm[j] are edges in the Q3?
                                         # No, they are edges in G.
                                         # The Q3 subgraph consists of the 12 edges of G that are NOT in the perfect matching of non-edges.
                                         # So we can remove any of these 12 edges.
                                         pass
                            # Let's just pick a random edge from the 16 possible, ensuring it's in G?
                            # Or pick one that is "central".
                            # Let's pick a random edge from the 12 edges of the Q3.
                            edges_in_q3 = []
                            for i in range(4):
                                for j in range(4):
                                    if not non_edge_mat[i][j]:
                                        edges_in_q3.append((A[i], B[j]))
                            if edges_in_q3:
                                u, v = random.choice(edges_in_q3)
                                adj[u][v] = False
                                adj[v][u] = False
                                break
                    if q3_found:
                        break
                if not q3_found:
                    break
            
            # Count edges
            e_count = int(np.sum(adj)) // 2
            if e_count > best_e:
                best_e = e_count
                best_g = adj.copy()
                
        return best_e, best_g

    e8, g8 = optimize_8()
    
    # Construct graph on 16 vertices
    edges = []
    if g8 is not None:
        # Component 1: 0..7
        for i in range(8):
            for j in range(i+1, 8):
                if g8[i][j]:
                    edges.append([i, j])
        # Component 2: 8..15
        for i in range(8):
            for j in range(i+1, 8):
                if g8[i][j]:
                    edges.append([i+8, j+8])
    else:
        # Fallback
        for i in range(7):
            for j in range(i + 1, 7):
                edges.append([i, j])
        for i in range(7, 14):
            for j in range(i + 1, 14):
                edges.append([i, j])
        edges.append([14, 15])
        edges.append([0, 7])
        edges.append([1, 8])

    return edges
