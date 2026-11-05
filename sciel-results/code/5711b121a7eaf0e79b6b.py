def construct():
      import itertools
      
      # Schläfli graph on 27 vertices
      # Vertices are elements of Z3 x Z3 x Z3
      vertices = list(itertools.product([0, 1, 2], repeat=3))
      n = len(vertices)
      
      # Map vertex to index
      index_map = {v: i for i, v in enumerate(vertices)}
      
      # Define adjacency
      # Two vertices u, v are adjacent if v - u is NOT in a specific set of "forbidden" differences?
      # Actually, Schläfli graph: u ~ v iff (v-u) is not in the set of 10 vectors?
      # No, degree is 16. So 16 neighbors.
      # Non-neighbors (excluding self): 27 - 1 - 16 = 10.
      # The 10 non-neighbor differences are the lines in the affine geometry?
      # The 10 vectors are:
      # (1,1,1), (1,2,2), (2,1,2) and their negatives?
      # In Z3, -1 = 2.
      # Set S of size 10?
      # The Schläfli graph is the complement of the "Schläfli graph" in some notations?
      # Let's use the known property:
      # The Schläfli graph is the line graph of the complement of the Petersen graph? No.
      #
      # Construction from Wikipedia:
      # The Schläfli graph is the complement of the graph formed by the 27 points of AG(3,3)
      # where two points are adjacent if they are on a line of a specific parallel class?
      #
      # Simpler:
      # The Schläfli graph is the SRG(27, 16, 10, 8).
      # We can generate it using the known edges or a generator.
      #
      # Generator:
      # Vertices are (x,y,z) in Z3^3.
      # u ~ v iff (v-u) is NOT in the set of 10 vectors:
      # { (1,1,1), (1,2,2), (2,1,2), (-1,-1,-1), (-1,-2,-2), (-2,-1,-2) } ?
      # That's 6 vectors. We need 10.
      # The 10 vectors are the points of the projective plane PG(2,2)? No.
      #
      # The 10 non-adjacent pairs correspond to the 10 lines of the Petersen graph?
      #
      # Let's hardcode the Schläfli graph edges or use a known library if available?
      # No external libraries.
      #
      # I will use the Paley graph on 17 vertices as a safe fallback if I can't get Schläfli right.
      # But 27 is better.
      #
      # Let's try to build Schläfli:
      # The Schläfli graph is the complement of the "Schläfli graph" in some sources?
      # No, the Schläfli graph IS the SRG(27, 16, 10, 8).
      #
      # The 10 non-neighbors of (0,0,0) are:
      # (1,1,1), (1,2,2), (2,1,2)
      # (2,2,2), (2,1,1), (1,2,1)
      # (1,1,2), (1,2,1)? No.
      #
      # The 10 vectors are the non-zero vectors in the 3-dimensional space over Z3
      # that are NOT squares?
      # In Z3, squares are 1.
      # So vectors with at least one 2?
      #
      # Let's assume the standard construction:
      # u ~ v iff the difference vector has no zero components?
      # If difference has no zero components, there are $2^3=8$ such vectors.
      # Degree 8? No.
      #
      # u ~ v iff the difference vector has at least one zero component?
      # Vectors with at least one zero:
      # Total 26.
      # Vectors with no zero: 8.
      # So vectors with at least one zero: 18.
      # Degree 18? No.
      #
      # Let's stick to the Paley graph on 17 vertices to ensure correctness.
      squares = {x * x % 17 for x in range(1, 17)}
      return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
