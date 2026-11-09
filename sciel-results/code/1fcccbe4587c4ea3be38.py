import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 16
    # Total possible edges in K16
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute all 35 bipartitions of 8 items into two sets of 4
    # For checking Q3, we need to check if a set of 8 vertices contains a Q3.
    # Q3 is K_{4,4} minus a perfect matching.
    # So for a set of 8 vertices, we check all ways to split them into A (size 4) and B (size 4).
    # Then we check if the edges between A and B in the graph form a superset of K_{4,4} minus a matching.
    # Actually, it's simpler: 
    # A graph G contains Q3 iff there exists a set of 8 vertices V' and a bijection from Q3 vertices to V'
    # such that all edges of Q3 are present in G.
    
    # Let's precompute the structure of Q3.
    # Q3 vertices: 0..7
    # Edges of Q3:
    q3_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    
    # For fast checking, we can precompute all 8-vertex subsets? No, that's 12870.
    # For each 8-vertex subset, we check all 35 splits.
    # Total checks per graph: 12870 * 35 * (check if 12 specific edges are present).
    # This is too slow for every fitness evaluation if done naively.
    
    # Better approach for fitness:
    # Count number of Q3s.
    # To count Q3s efficiently:
    # Iterate over all 8-vertex subsets (12870).
    # For each subset, build the 8x8 adjacency matrix.
    # Check if it contains Q3.
    # To check if an 8-vertex induced subgraph contains Q3:
    # We can try all 35 splits (A, B).
    # For a split (A, B), the edges of Q3 correspond to a specific set of 12 edges between A and B.
    # Wait, Q3 is not just bipartite. It has edges within A and within B?
    # No, Q3 is bipartite. The vertices can be colored black/white.
    # Let's check the bipartition of Q3.
    # Q3 is the cube graph. It is bipartite.
    # Let's find the bipartition of Q3 vertices 0..7.
    # 0 is even parity (0 bits). 1,2,4 are odd.
    # So bipartition is {0,3,5,6} and {1,2,4,7}.
    # Let's verify.
    # Edges:
    # 0-1 (0 even, 1 odd) OK
    # 0-2 (0 even, 2 odd) OK
    # 0-4 (0 even, 4 odd) OK
    # 1-3 (1 odd, 3 even) OK
    # 1-5 (1 odd, 5 even) OK
    # 2-3 (2 odd, 3 even) OK
    # 2-6 (2 odd, 6 even) OK
    # 3-7 (3 even, 7 odd) OK
    # 4-5 (4 odd, 5 even) OK
    # 4-6 (4 odd, 6 even) OK
    # 5-7 (5 even, 7 odd) OK
    # 6-7 (6 even, 7 odd) OK
    # Yes.
    # So Q3 is a bipartite graph with parts A_q3 = {0,3,5,6} and B_q3 = {1,2,4,7}.
    # The edges are exactly the 12 edges between these two sets in the cube.
    # Which edges are missing?
    # All 16 possible edges between A_q3 and B_q3.
    # Missing:
    # 0-7? No edge in cube. (0:000, 7:111) dist 3. No.
    # 3-1? No edge. (3:011, 1:001) dist 1. Wait. 3 is 011, 1 is 001. They differ in one bit. So 1-3 IS an edge.
    # Let's list edges from 0: 1, 2, 4. Not 3, 5, 6, 7.
    # Edges from 3 (011): 1 (001), 2 (010), 7 (111). Not 0, 4, 5, 6.
    # Edges from 5 (101): 1 (001), 4 (100), 7 (111). Not 0, 2, 3, 6.
    # Edges from 6 (110): 2 (010), 4 (100), 7 (111). Not 0, 1, 3, 5.
    
    # So for a specific assignment of Q3 vertices to the 8 graph vertices,
    # we need to check if the 12 specific edges are present.
    
    # To speed up:
    # Precompute all 35 splits of 8 positions into A and B.
    # For each split, precompute the 12 pairs of indices (i, j) such that i in A, j in B, and (i, j) is an edge in Q3.
    # Wait, the mapping from Q3 vertices to graph vertices is arbitrary.
    # So for a fixed set of 8 graph vertices, we try all 35 splits.
    # For a split (A_idx, B_idx), we check if there is a bijection from A_q3 to A_idx and B_q3 to B_idx such that all edges are present.
    # This is equivalent to checking if the bipartite graph induced by A_idx and B_idx contains the bipartite graph of Q3 as a subgraph.
    # The bipartite graph of Q3 has 12 edges.
    # We can precompute the adjacency matrix of Q3's bipartite structure.
    # Let A_q3 = [0, 3, 5, 6] and B_q3 = [1, 2, 4, 7].
    # Edges in Q3 between A_q3 and B_q3:
    # 0-1, 0-2, 0-4
    # 3-1, 3-2, 3-7
    # 5-1, 5-4, 5-7
    # 6-2, 6-4, 6-7
    
    # Let's represent this as a 4x4 matrix M where M[i][j] = 1 if A_q3[i] is connected to B_q3[j] in Q3.
    # A_q3 indices: 0->0, 3->1, 5->2, 6->3
    # B_q3 indices: 1->0, 2->1, 4->2, 7->3
    
    # M[0] (v0): connected to 1,2,4 -> indices 0,1,2. So [1,1,1,0]
    # M[1] (v3): connected to 1,2,7 -> indices 0,1,3. So [1,1,0,1]
    # M[2] (v5): connected to 1,4,7 -> indices 0,2,3. So [1,0,1,1]
    # M[3] (v6): connected to 2,4,7 -> indices 1,2,3. So [0,1,1,1]
    
    q3_adj = [
        [1,1,1,0],
        [1,1,0,1],
        [1,0,1,1],
        [0,1,1,1]
    ]
    
    # Precompute all 35 splits of range(8) into two sets of 4.
    splits = []
    seen_splits = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen_splits:
            seen_splits.add(key)
            splits.append((A, B))
            
    # For each split, we need to check if the graph induced by A and B contains Q3's bipartite graph.
    # This is a subgraph isomorphism problem for bipartite graphs of size 4x4.
    # Since 4 is small, we can just check all 4! * 4! = 576 permutations? No, that's too slow for every split.
    # Better: Use bitmask checks.
    # For a specific assignment of A_q3 to A_idx and B_q3 to B_idx, we check 12 edges.
    # But we don't know the assignment.
    
    # Alternative: Precompute for every possible 4x4 bipartite adjacency matrix (2^16 = 65536 possibilities)
    # whether it contains Q3's bipartite graph as a subgraph.
    # If it does, then any graph with that induced bipartite structure contains Q3.
    
    # Let's precompute a lookup table: `has_q3_subgraph[bitmask]`
    # bitmask represents the 16 possible edges between A and B.
    # Bit i*4+j corresponds to edge between A[i] and B[j].
    
    has_q3_subgraph = [False] * (1 << 16)
    
    # Generate all 576 permutations of A and B?
    # Actually, we just need to check if there exists a permutation of rows and columns such that
    # the permuted matrix covers q3_adj.
    # i.e., for all i,j where q3_adj[i][j]==1, permuted_matrix[i][j]==1.
    
    # Let's iterate over all 4! * 4! = 576 permutations.
    # For each permutation (row_perm, col_perm), construct the required bitmask.
    # If the candidate bitmask (from the graph) is a superset of the required bitmask, then it contains Q3.
    
    required_masks = set()
    for row_perm in permutations(range(4)):
        for col_perm in permutations(range(4)):
            mask = 0
            for i in range(4):
                for j in range(4):
                    if q3_adj[i][j]:
                        # The edge in the candidate graph is between A[row_perm[i]] and B[col_perm[j]]
                        # Wait, the candidate graph's A and B are fixed sets of vertices.
                        # The permutation maps Q3's A_q3 to the candidate's A vertices.
                        # Let's say A_q3[k] is mapped to A_idx[row_perm[k]].
                        # And B_q3[l] is mapped to B_idx[col_perm[l]].
                        # Then the edge A_q3[i]-B_q3[j] in Q3 corresponds to edge A_idx[row_perm[i]] - B_idx[col_perm[j]] in candidate.
                        # In the bitmask, bit (row_perm[i])*4 + col_perm[j] must be set.
                        mask |= (1 << (row_perm[i] * 4 + col_perm[j]))
            required_masks.add(mask)
            
    # Now, for any candidate bitmask, it contains Q3 if (candidate_mask & req_mask) == req_mask for any req_mask in required_masks.
    # We can precompute this for all 65536 masks.
    for m in range(1 << 16):
        for rm in required_masks:
            if (m & rm) == rm:
                has_q3_subgraph[m] = True
                break
                
    # Now we can count Q3s in a graph represented by a 120-bit integer.
    def count_q3s(graph_mask):
        # graph_mask is an integer where bit i is 1 if edge all_edges[i] is present.
        count = 0
        # Iterate over all 8-vertex subsets
        for v_subset in combinations(range(n), 8):
            # Build the 8x8 adjacency matrix for this subset
            # We need to map global vertices to local indices 0..7
            # local_adj[i][j] is 1 if edge (v_subset[i], v_subset[j]) is in graph
            # We can build a 64-bit mask for the 8x8 matrix? No, 28 edges.
            # Let's just build a 8x8 boolean matrix or a list of edges.
            
            # For each of the 35 splits:
            for A_idx, B_idx in splits:
                # A_idx and B_idx are tuples of local indices (0..7)
                # We need to construct the 16-bit mask for edges between A_idx and B_idx
                mask = 0
                for i, ai in enumerate(A_idx):
                    for j, bj in enumerate(B_idx):
                        # Get global vertices
                        u = v_subset[ai]
                        v = v_subset[bj]
                        # Find edge index
                        e_idx = edge_to_idx[(min(u,v), max(u,v))]
                        if (graph_mask >> e_idx) & 1:
                            mask |= (1 << (i * 4 + j))
                
                if has_q3_subgraph[mask]:
                    count += 1
        return count

    # This is still likely too slow for GA fitness evaluation.
    # 12870 subsets * 35 splits * (construct mask + lookup)
    # Constructing mask takes 16 lookups.
    # 12870 * 35 * 16 ~ 7 million operations per fitness eval.
    # If GA does 1000 evals, that's 7 billion operations. Too slow in Python.
    
    # We need a faster fitness function.
    # Idea: Only check Q3s that are "close" to being formed?
    # Or, use a heuristic penalty.
    # Or, precompute the list of all Q3 embeddings in K16.
    # How many Q3 embeddings in K16?
    # Choose 8 vertices: 12870.
    # For each, 35 splits.
    # For each split, check if it's a Q3.
    # Total potential Q3s is at most 12870 * 35 = 450450.
    # We can precompute the list of edge masks for each potential Q3.
    # There are at most 450450 such masks.
    # Then, for a given graph, count how many of these masks are subsets of the graph mask.
    # This is: for each q3_mask, check if (graph_mask & q3_mask) == q3_mask.
    # 450k checks per fitness eval.
    # 450k * 1000 evals = 450 million checks. Still might be slow in Python, but feasible if optimized?
    # 450 million bitwise ANDs in Python might take ~20-40 seconds.
    # We have 115 seconds.
    # If we limit GA to 200 generations * 50 population = 10000 evals? No, 1000 evals is better.
    # Let's try to optimize the counting.
    
    # Precompute all potential Q3 masks.
    q3_masks = []
    for v_subset in combinations(range(n), 8):
        for A_idx, B_idx in splits:
            mask = 0
            for i, ai in enumerate(A_idx):
                for j, bj in enumerate(B_idx):
                    u = v_subset[ai]
                    v = v_subset[bj]
                    e_idx = edge_to_idx[(min(u,v), max(u,v))]
                    mask |= (1 << e_idx)
            # Check if this mask corresponds to a Q3
            # We need to check if the bipartite graph formed by this mask is a supergraph of Q3.
            # Wait, the mask contains ALL edges between A and B that are present in K16.
            # No, the mask I constructed above contains ALL 16 edges between A and B.
            # That's not right.
            # The mask should only contain the edges of the Q3.
            # But the Q3 edges depend on the permutation.
            # So for a fixed v_subset and split, there are multiple possible Q3 masks (one for each valid permutation).
            # We should add ALL valid Q3 masks to the list.
            
            # For each of the 576 permutations, check if it forms a valid Q3 subgraph in K16 (which it always does, since K16 is complete).
            # But we want the mask of the Q3 edges.
            # So for each perm, construct the mask of the 12 edges.
            for row_perm in permutations(range(4)):
                for col_perm in permutations(range(4)):
                    q3_mask = 0
                    for i in range(4):
                        for j in range(4):
                            if q3_adj[i][j]:
                                ai = A_idx[row_perm[i]]
                                bj = B_idx[col_perm[j]]
                                u = v_subset[ai]
                                v = v_subset[bj]
                                e_idx = edge_to_idx[(min(u,v), max(u,v))]
                                q3_mask |= (1 << e_idx)
                    q3_masks.append(q3_mask)
                    
    # Remove duplicates
    q3_masks = list(set(q3_masks))
    print(f"Number of unique Q3 masks: {len(q3_masks)}")
    
    # Now fitness function:
    def fitness(graph_mask):
        # Count edges
        edge_count = bin(graph_mask).count('1')
        # Count Q3s
        q3_count = 0
        for q3_mask in q3_masks:
            if (graph_mask & q3_mask) == q3_mask:
                q3_count += 1
        # Penalty
        penalty = 100 * q3_count
        return edge_count - penalty

    # Genetic Algorithm
    population_size = 50
    # Initialize population
    # Start with a few good seeds and random ones
    population = []
    
    # Seed 1: K3,13 + clique
    mask1 = 0
    for a in range(3):
        for b in range(3, 16):
            e_idx = edge_to_idx[(a,b)]
            mask1 |= (1 << e_idx)
    for i in range(3):
        for j in range(i+1, 3):
            e_idx = edge_to_idx[(i,j)]
            mask1 |= (1 << e_idx)
    population.append(mask1)
    
    # Seed 2: Random dense
    for _ in range(population_size - 1):
        m = 0
        for i in range(num_edges):
            if random.random() < 0.6:
                m |= (1 << i)
        population.append(m)
        
    best_mask = 0
    best_fit = -1000000
    
    # Evaluate initial population
    for m in population:
        f = fitness(m)
        if f > best_fit:
            best_fit = f
            best_mask = m
            
    # GA Loop
    while time.time() < deadline:
        # Selection: Tournament selection
        new_population = []
        for _ in range(population_size):
            # Pick 3 random individuals, choose best
            p1 = random.choice(population)
            p2 = random.choice(population)
            p3 = random.choice(population)
            
            f1 = fitness(p1)
            f2 = fitness(p2)
            f3 = fitness(p3)
            
            best_p = p1
            best_f = f1
            if f2 > best_f:
                best_p = p2
                best_f = f2
            if f3 > best_f:
                best_p = p3
                best_f = f3
                
            new_population.append(best_p)
            
        # Crossover and Mutation
        next_population = []
        for i in range(0, population_size, 2):
            p1 = new_population[i]
            p2 = new_population[i+1]
            
            # Uniform crossover
            child1 = 0
            child2 = 0
            for bit in range(num_edges):
                if random.random() < 0.5:
                    if (p1 >> bit) & 1:
                        child1 |= (1 << bit)
                    if (p2 >> bit) & 1:
                        child2 |= (1 << bit)
                else:
                    if (p1 >> bit) & 1:
                        child2 |= (1 << bit)
                    if (p2 >> bit) & 1:
                        child1 |= (1 << bit)
                        
            # Mutation
            # Flip a few bits
            for _ in range(5):
                bit = random.randint(0, num_edges - 1)
                child1 ^= (1 << bit)
                child2 ^= (1 << bit)
                
            next_population.append(child1)
            next_population.append(child2)
            
        # Evaluate new population
        population = next_population
        for m in population:
            f = fitness(m)
            if f > best_fit:
                best_fit = f
                best_mask = m
                
    # Convert best_mask to edge list
    edges = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            edges.append(list(all_edges[i]))
            
    return edges

from itertools import permutations

if __name__ == "__main__":
    res = construct()
    print(len(res))
