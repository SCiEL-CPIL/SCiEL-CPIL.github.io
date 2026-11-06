import time
import random

def construct():
    start_time = time.time()
    
    # We search for a valid 4-coloring of K_n for n up to 19.
    # A known lower bound for R_4(C6) is 18, meaning there exists a 4-coloring of K_17.
    # It is unknown if K_18 or K_19 can be colored. 
    # We will attempt to find a coloring for n=19, 18, 17, 16...
    # 
    # Strategy:
    # Try to construct a coloring based on cyclic differences in Z_n.
    # For a cyclic coloring, the color of edge (i, j) depends on (j - i) mod n.
    # We partition the non-zero elements of Z_n into 4 sets S_0, S_1, S_2, S_3.
    # A color class c contains a 6-cycle if there exist distinct differences d1..d6 in S_c
    # such that they form a cycle. Specifically, in Z_n, a 6-cycle in the graph defined by S_c
    # corresponds to a sequence of 6 steps summing to 0.
    # 
    # For n=17, the non-zero elements are {1..16}.
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16} (8 elements)
    # Non-residues: {3, 5, 6, 7, 10, 11, 12, 14} (8 elements)
    # 17 is prime.
    # 
    # Actually, let's just use a randomized local search (Simulated Annealing or Hill Climbing)
    # to find a valid coloring for n=17, 18, 19.
    # Objective: Minimize the number of monochromatic C6s.
    # 
    # Since n is small (<= 20), we can afford some search.
    # However, detecting C6s is O(n^6) which is ~ 60M for n=20, too slow for many iterations.
    # We need a faster check or a smarter heuristic.
    #
    # Alternative: Use the known explicit construction for n=17 if possible, or 
    # try to verify a random coloring quickly.
    #
    # Let's try a specific algebraic construction for n=17 first.
    # If that fails or we can't code it up perfectly, we fall back to n=9 seed or simple random search.
    #
    # Actually, the prompt says "The best known colouring has 17 vertices".
    # So if we can return n=17, we get score 17/9 = 1.88.
    # If we return n=18 or 19, it's a breakthrough.
    #
    # Let's try to hardcode or generate a valid n=17 coloring.
    # A common source for Ramsey numbers is the "Ramsey Theory" literature.
    # For R_4(C6), a construction using finite fields is standard.
    #
    # Let's try a simple random search with a fast C6 detector for n=17, 18, 19.
    # Fast C6 detector:
    # For each color c, build the adjacency matrix A_c.
    # A 6-cycle exists if trace(A_c^6) > 0 (roughly, needs correction for shorter cycles and closed walks).
    # Actually, trace(A^6) counts closed walks of length 6.
    # A simple cycle of length 6 contributes 12 to trace(A^6) (6 rotations * 2 directions).
    # Triangles contribute to closed walks of length 3, and can be extended to length 6.
    # This is complex.
    #
    # Simpler: Just check for C6 directly but optimized.
    # For n=17, number of edges per color ~ 136/4 = 34.
    # Number of paths of length 5 is manageable?
    #
    # Let's try to find a known solution online in my memory...
    # I recall that R_4(C6) is between 18 and 20? Or 17 and 19?
    # The prompt says "The best known colouring has 17 vertices; 18 or 19 vertices would be new."
    # This implies R_4(C6) > 17 is known, but R_4(C6) > 18 is unknown.
    # So finding n=17 is a guaranteed improvement over the seed (n=9).
    #
    # I will implement a random search that tries to find a valid coloring for n=17.
    # If it finds one within time, return it.
    # If not, return the seed n=9.
    #
    # To make it fast:
    # 1. Generate random coloring.
    # 2. Check validity.
    # 3. If invalid, flip a random edge in a color class that has a C6?
    #    Or just restart?
    #
    # Given 150s, we can do many random trials.
    # Checking a single coloring for n=17:
    # 4 colors. For each color, check if graph has C6.
    # Graph has ~34 edges.
    # We can use networkx? No, standard library only.
    #
    # I will write a specific C6 checker.
    # A graph has a C6 if there is a path of length 5 that closes.
    # Or simply: iterate all subsets of 6 vertices? C(17,6) = 12376.
    # For each subset, check if induced subgraph has a 6-cycle?
    # Or if the 6 vertices form a cycle in the color class.
    #
    # Efficient check for "does color class c contain a C6":
    # Let G_c be the graph.
    # If G_c has a C6, it contains a cycle of length 6.
    # We can check all 6-tuples? No.
    #
    # Let's use the fact that we only need to find ONE valid coloring.
    # Maybe I can construct one using the "cyclic" method with specific difference sets for n=17.
    #
    # Difference sets for n=17:
    # We need to partition {1..16} into 4 sets of size 4.
    # Let the sets be D0, D1, D2, D3.
    # We need no set Di to contain a 6-cycle in Z_17.
    # A 6-cycle in Z_17 using differences from D means d1+...+d6 = 0 mod 17.
    #
    # Let's try a random partition of {1..16} into 4 sets of size 4 and check.
    # There are 16! / (4!^4 * 4!) ~ 63 million partitions.
    # We can sample a few thousand.
    
    def has_c6_in_color(colors, n, c):
        # Build adjacency list for color c
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                idx = i*n - i*(i+1)//2 + (j-i-1) # This is wrong, let's just use a 2D array or list of lists from flat list
                # Let's stick to a flat list index calculation or just build a 2D matrix
                pass
        
        # Better: build matrix
        mat = [[False]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == c:
                    mat[i][j] = True
                    mat[j][i] = True
                idx += 1
        
        # Check for C6
        # A C6 is a sequence v0-v1-v2-v3-v4-v5-v0 with all distinct.
        # We can iterate v0, v1, v2, v3, v4, v5? 17^6 is too big.
        # Iterate v0, v1, v2, v3. v4 must be neighbor of v3 and v5.
        #
        # Optimization:
        # For each pair (u, v) in the graph, count common neighbors.
        # A 4-cycle exists if common neighbors > 0.
        # A 6-cycle...
        
        # Simplest robust check for small n:
        # Use DFS to find a cycle of length 6.
        # Or just check all permutations of 6 distinct vertices?
        # C(17,6) * 6! / 12 is the number of cycles.
        # C(17,6) = 12376.
        # For each set of 6 vertices, check if they form a C6 in the subgraph.
        # This is fast enough for 17 vertices.
        
        import itertools
        vertices = list(range(n))
        for combo in itertools.combinations(vertices, 6):
            # Check if these 6 vertices contain a 6-cycle in mat
            # A 6-cycle in a 6-vertex set means the induced subgraph is a C6 or contains a C6.
            # Since the set has size 6, a C6 uses all 6 vertices.
            # So we just need to check if the induced subgraph on these 6 vertices has a Hamiltonian cycle.
            # Or simply, does it contain a cycle of length 6?
            # In a 6-vertex graph, a cycle of length 6 is a Hamiltonian cycle.
            
            # Check if there is a permutation of combo that forms a cycle
            # We can fix v0=combo[0] to reduce symmetry.
            v0 = combo[0]
            # We need to visit all other 5 and return to v0.
            # This is TSP on 6 nodes. 5! = 120 checks.
            # 12376 * 120 ~ 1.5M operations per color.
            # 4 colors * 1.5M = 6M operations per trial.
            # This is doable in Python if optimized, but might be slow for many trials.
            
            # Faster:
            # Count edges in induced subgraph. Must be at least 6.
            # Actually, just check if the induced subgraph has a cycle of length 6.
            # Since n=6, we can just check if the graph is connected and has 6 edges and is 2-regular?
            # No, it could have chords.
            # But if it has a C6, it has a cycle of length 6.
            
            # Let's just do the permutation check but break early.
            rest = [v for v in combo if v != v0]
            for perm in itertools.permutations(rest):
                # v0 -> perm[0] -> ... -> perm[4] -> v0
                if (mat[v0][perm[0]] and mat[perm[0]][perm[1]] and mat[perm[1]][perm[2]] and 
                    mat[perm[2]][perm[3]] and mat[perm[3]][perm[4]] and mat[perm[4]][v0]):
                    return True
        return False

    def is_valid(colors, n):
        for c in range(4):
            if has_c6_in_color(colors, n, c):
                return False
        return True

    # Try to find a valid coloring for n=17, 18, 19
    # We will try random cyclic colorings (differences) first, as they are structured.
    # If that fails, random arbitrary colorings.
    
    # Deadline guard
    deadline = start_time + 140.0
    
    # Try n=19
    for n in [19, 18, 17]:
        if time.time() > deadline:
            break
            
        # Try random difference sets for cyclic coloring
        # Number of non-zero elements is n-1.
        # We need to partition 1..n-1 into 4 sets.
        # If n-1 is not divisible by 4, we can't have equal size.
        # For n=17, 16/4 = 4.
        # For n=18, 17/4 = 4.25. Not equal.
        # For n=19, 18/4 = 4.5. Not equal.
        
        # So for n=18, 19, cyclic difference coloring with equal sizes is not possible.
        # We can still try random partitions of the differences, but the "cyclic" symmetry is broken if sizes differ?
        # No, a cyclic coloring is defined by a function f: Z_n -> {0,1,2,3} where color(i,j) = f((j-i)%n).
        # This always works, but the sets of differences can have different sizes.
        
        # Let's try random cyclic colorings.
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            if n == 17:
                # Partition 1..16 into 4 sets of size 4
                diffs = list(range(1, 17))
                random.shuffle(diffs)
                s0 = set(diffs[0:4])
                s1 = set(diffs[4:8])
                s2 = set(diffs[8:12])
                s3 = set(diffs[12:16])
                sets = [s0, s1, s2, s3]
            else:
                # Random assignment of each difference to a color
                sets = [set() for _ in range(4)]
                for d in range(1, n):
                    sets[random.randint(0, 3)].add(d)
            
            # Build colors list
            colors = [0] * (n*(n-1)//2)
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    d = (j - i) % n
                    # d is in 1..n-1
                    for c in range(4):
                        if d in sets[c]:
                            colors[idx] = c
                            break
                    idx += 1
            
            if is_valid(colors, n):
                return {'n': n, 'colors': colors}
                
        # Try random arbitrary colorings if cyclic failed
        for attempt in range(100):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
            if is_valid(colors, n):
                return {'n': n, 'colors': colors}
                
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
