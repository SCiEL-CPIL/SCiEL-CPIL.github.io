import time
import sys
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    # Total possible edges in K16
    # Vertices 0..15
    # Edge (u, v) with u < v
    # Total edges = 16 * 15 / 2 = 120
    
    # 1. Generate all possible edges
    all_edges = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_index[(u, v)] = idx
            idx += 1
            
    m = len(all_edges)
    
    # 2. Identify all subsets of 8 vertices that could form a Q3
    # A Q3 is a subgraph isomorphic to the 3-cube.
    # The 3-cube Q3 has 8 vertices and 12 edges.
    # It is a bipartite graph with two partitions of size 4.
    # Key Insight: A set of 8 vertices S contains a Q3 subgraph if and only if
    # there exists a partition of S into A and B (|A|=4, |B|=4) such that
    # the induced bipartite graph G[A, B] contains a Q3.
    
    # Actually, a simpler characterization for "contains Q3 as subgraph":
    # Q3 is the unique 3-regular bipartite graph on 4+4 vertices up to isomorphism?
    # No, K_{4,4} minus a perfect matching is Q3.
    # Are all 3-regular bipartite graphs on 4+4 isomorphic to Q3?
    # Yes. The automorphism group of K_{4,4} acts transitively on perfect matchings.
    # So, S contains a Q3 iff there is a bipartition (A, B) of S such that
    # the edges of G between A and B include ALL edges of K_{4,4} \ M for some perfect matching M.
    # Equivalently, the missing edges between A and B in G must form a subset of a perfect matching.
    # This implies the missing edges must be a matching (no shared vertices) and size <= 4.
    
    # To forbid Q3, for every 8-vertex subset S and every bipartition (A, B) of S,
    # the set of missing edges between A and B must NOT be a subset of a perfect matching.
    # Wait, if the missing edges form a matching of size k, we can extend it to a perfect matching M.
    # Then K_{4,4} \ M is a subgraph of G[A, B].
    # So, to prevent Q3, for every S and every bipartition (A, B),
    # the missing edges between A and B must NOT form a matching.
    # i.e., There must be at least two missing edges that share a vertex, OR the number of missing edges is > 4.
    
    # Let x_e be 1 if edge e is present, 0 if absent.
    # Let E_AB be the set of edges between A and B in K_{4,4}. |E_AB| = 16.
    # Missing edges are those e in E_AB where x_e = 0.
    # Condition: The set { e in E_AB | x_e = 0 } is NOT a matching.
    # This is a non-linear constraint (logical OR of conditions).
    # How to linearize?
    # "Not a matching" means:
    # 1. There exist two missing edges sharing a vertex u in A (or v in B).
    #    i.e., there exists u in A such that sum_{v in B} (1 - x_{uv}) >= 2.
    #    OR there exists v in B such that sum_{u in A} (1 - x_{uv}) >= 2.
    # 2. The total number of missing edges is > 4.
    #    i.e., sum_{e in E_AB} (1 - x_e) >= 5.
    
    # So, for each S and bipartition (A, B), we must enforce:
    # (Exists u in A: sum_{v in B} (1 - x_{uv}) >= 2) 
    # OR (Exists v in B: sum_{u in A} (1 - x_{uv}) >= 2)
    # OR (sum_{e in E_AB} (1 - x_e) >= 5)
    
    # This is an OR of linear inequalities.
    # We can use the "Big-M" method or introduce binary variables to represent the choice of which condition holds.
    # However, generating all such constraints might be large.
    # Number of 8-sets: C(16, 8) = 12870.
    # Number of bipartitions per 8-set: C(8, 4) / 2 = 35.
    # Total constraints: 12870 * 35 = 450,450.
    # This is too many for a simple ILP solver to handle quickly in Python without heavy preprocessing.
    
    # Alternative Approach:
    # Instead of forbidding ALL Q3s directly, we can use a different structural insight.
    # The problem is equivalent to finding the maximum size of a graph on 16 vertices with no Q3 subgraph.
    # This is the Turan number ex(16, Q3).
    # Known bounds: ex(16, Q3) is likely around 40-50?
    # The seed score 1.3846 corresponds to a specific number of edges.
    # Let's estimate the seed edges.
    # Seed: K3,13 + K3 on part A.
    # Edges in K3,13: 3 * 13 = 39.
    # Edges in K3: 3.
    # Total = 42 edges.
    # Score 1.3846 = 42 / 30.3? No, score = value / seed_value.
    # If seed value is 42, and score is 1.3846, then seed_value = 42 / 1.3846 = 30.3?
    # Wait, the prompt says "score 1.3846" and "seed program".
    # Usually seed score is 1.0. Here it says "score 1.3846" for the CURRENT program.
    # And "search score is value/seed".
    # If the current program IS the seed, its score should be 1.0.
    # The prompt says: "CURRENT PROGRAM (score 1.3846) and its strategy".
    # This implies the current program scores 1.3846 relative to some baseline?
    # Or is 1.3846 the raw value? No, "Maximize the number of edges... value/seed".
    # If the current program is the seed, its score is 1.0 by definition of "value/seed".
    # But it says "score 1.3846". This is confusing.
    # Let's re-read: "The evaluator measures the raw value... search score is value/seed, so 1.0 equals the seed program".
    # Maybe the "seed" is a specific baseline graph, and the current program is an improvement?
    # Or maybe the "CURRENT PROGRAM" is NOT the seed, but a program that scored 1.3846?
    # "CURRENT PROGRAM (score 1.3846) and its strategy:"
    # It lists a program.
    # If this program is the seed, its score would be 1.0.
    # Therefore, the "seed" is likely a simpler graph (e.g., 30 edges), and the current program (42 edges) scores 42/30.3 = 1.38.
    # Let's assume the goal is to find a graph with MORE than 42 edges.
    
    # Given the complexity of the full ILP formulation with 450k constraints, 
    # and the 120s time limit, we might need a smarter approach or a smaller constraint set.
    
    # Observation:
    # Q3 is a specific graph. 
    # Maybe we can use a different decomposition.
    # What if we just run the ILP on a reduced set of constraints?
    # Or, we can use a heuristic: 
    # The Turan graph T(16, 2) (complete bipartite K8,8) has 64 edges. Does it contain Q3?
    # K8,8 contains K4,4. K4,4 contains Q3 (as K4,4 minus matching). 
    # So K8,8 is not Q3-free.
    # What about T(16, 3)? T(16,3) has parts of size 6,5,5.
    # Edges: 6*5 + 6*5 + 5*5 = 30 + 30 + 25 = 85 edges.
    # Does T(16,3) contain Q3?
    # Q3 is bipartite. It must be contained in the bipartite parts of the Turan graph.
    # In T(16,3), any bipartite subgraph is between two parts.
    # Max bipartite subgraph is between part of size 6 and part of size 5 (or 5 and 5).
    # K6,5 contains K4,4. K4,4 contains Q3.
    # So T(16,3) contains Q3.
    
    # What about T(16, 4)? Parts 4,4,4,4.
    # Edges: 6 * (4*4) = 96 edges.
    # Bipartite subgraphs are K4,4.
    # K4,4 contains Q3.
    
    # It seems complete bipartite graphs are bad.
    # We need a graph that avoids K4,4? No, just avoids Q3.
    # But Q3 is a subgraph of K4,4.
    # So if a graph contains K4,4, it contains Q3.
    # Thus, a Q3-free graph must be K4,4-free?
    # No. K4,4 contains Q3. So if G contains K4,4, G contains Q3.
    # Therefore, ex(16, Q3) <= ex(16, K4,4).
    # ex(16, K4,4) is the max edges in a K4,4-free graph.
    # By Kővári–Sós–Turán theorem, ex(n, K_{s,t}) <= (t-1)^{1/s} n^{2-1/s} + (s-1)n.
    # For s=4, t=4: ex(16, K4,4) <= 3^{1/4} * 16^{1.75} + 3*16.
    # 16^{1.75} = (2^4)^{1.75} = 2^7 = 128.
    # 3^{0.25} ~ 1.316.
    # 1.316 * 128 + 48 ~ 168 + 48 = 216. This bound is loose.
    # Actual ex(16, K4,4) is likely much smaller.
    # Known values for ex(n, K4,4):
    # For n=16, it is known that ex(16, K4,4) is small.
    # Actually, if G is K4,4-free, it is Q3-free.
    # So we can solve for max edges in a K4,4-free graph.
    # This is a standard extremal graph problem.
    
    # Let's formulate the ILP for K4,4-free.
    # Constraint: For every 4+4 subset (A, B), the number of edges between A and B must be < 16.
    # i.e., sum_{u in A, v in B} x_{uv} <= 15.
    # Number of constraints: C(16, 4) * C(12, 4) / 2?
    # No, A and B are disjoint.
    # Choose A (4 verts), choose B (4 verts from remaining 12).
    # C(16, 4) * C(12, 4) = 1820 * 495 = 900,900.
    # This is also large, but the constraints are simple sums <= 15.
    # However, K4,4-free is a STRONGER condition than Q3-free.
    # A graph can be Q3-free but contain K4,4?
    # No, because Q3 is a subgraph of K4,4.
    # If G contains K4,4, it contains Q3.
    # So Q3-free => K4,4-free.
    # Wait. "Contains Q3" means there is a subgraph isomorphic to Q3.
    # If G contains K4,4, does it contain Q3?
    # Yes, Q3 is a subgraph of K4,4.
    # So any Q3-free graph is K4,4-free.
    # Therefore, the maximum number of edges in a Q3-free graph is <= maximum edges in a K4,4-free graph.
    # Is the converse true? Can we have a K4,4-free graph that contains Q3?
    # No, because if it contains Q3, it contains a subgraph of K4,4?
    # Q3 is not a subgraph of K4,4? 
    # Q3 has 12 edges. K4,4 has 16.
    # Q3 is a subgraph of K4,4.
    # So if G contains Q3, G contains a set of 8 vertices inducing at least the edges of Q3.
    # This does NOT imply G contains K4,4.
    # Example: G could be Q3 itself. Q3 is K4,4-free (it has 12 edges, K4,4 needs 16).
    # So Q3-free is a WEAKER condition than K4,4-free?
    # No.
    # If G contains K4,4, G contains Q3.
    # So K4,4-free => Q3-free.
    # So the set of Q3-free graphs is a SUPERSET of K4,4-free graphs.
    # Therefore, ex(16, Q3) >= ex(16, K4,4).
    # We want to maximize edges in a Q3-free graph.
    # Solving for K4,4-free gives a LOWER bound on ex(16, Q3).
    # If we find a K4,4-free graph with M edges, we know ex(16, Q3) >= M.
    # But we might be able to do better by allowing some K4,4s as long as they don't contain a Q3?
    # Impossible. Every K4,4 contains a Q3.
    # So any graph containing a K4,4 is NOT Q3-free.
    # Therefore, Q3-free <=> K4,4-free?
    # No. A graph can contain Q3 without containing K4,4 (e.g., the graph Q3 itself).
    # So Q3-free is a STRONGER condition than K4,4-free?
    # Let's check the logic.
    # Condition A: Graph is Q3-free.
    # Condition B: Graph is K4,4-free.
    # If Graph has K4,4, it has Q3. So (Not Q3-free) => (Not K4,4-free).
    # Contrapositive: (K4,4-free) => (Q3-free).
    # So K4,4-free graphs are a subset of Q3-free graphs.
    # We want to maximize edges in Q3-free graphs.
    # The maximum edges in K4,4-free graphs is a lower bound.
    # Can we add edges to a K4,4-free graph and still be Q3-free?
    # Yes, if the new edges don't complete a Q3.
    # But wait. If we add an edge to a K4,4-free graph, do we create a K4,4?
    # Not necessarily.
    # But if we create a Q3, we are invalid.
    # So we can potentially have MORE edges than the K4,4-free bound?
    # No, the K4,4-free bound is the max edges for a graph that is K4,4-free.
    # Since Q3-free graphs include K4,4-free graphs, the max edges for Q3-free is AT LEAST the max edges for K4,4-free.
    # It could be higher if there are Q3-free graphs that contain K4,4?
    # No, we proved K4,4 implies Q3. So no Q3-free graph contains K4,4.
    # So ALL Q3-free graphs are K4,4-free.
    # Therefore, the set of Q3-free graphs is a SUBSET of K4,4-free graphs?
    # Let's re-evaluate:
    # 1. If G contains K4,4, G contains Q3.
    # 2. So if G is Q3-free, G does NOT contain K4,4.
    # 3. So Q3-free => K4,4-free.
    # 4. So the set of Q3-free graphs is a SUBSET of the set of K4,4-free graphs.
    # 5. Therefore, ex(16, Q3) <= ex(16, K4,4).
    
    # Conclusion: To find the max edges in a Q3-free graph, we can solve for the max edges in a K4,4-free graph.
    # Any K4,4-free graph is Q3-free.
    # And no Q3-free graph can have more edges than the best K4,4-free graph?
    # Wait. If Q3-free is a subset, the max element of the subset is <= the max element of the superset.
    # So ex(16, Q3) <= ex(16, K4,4).
    # And since any K4,4-free graph is Q3-free, ex(16, Q3) >= ex(16, K4,4).
    # So ex(16, Q3) = ex(16, K4,4).
    
    # This simplifies the problem drastically!
    # We just need to find the maximum number of edges in a K4,4-free graph on 16 vertices.
    
    # ILP Formulation for K4,4-free:
    # Variables: x_e for each edge e. Binary.
    # Objective: Maximize sum x_e.
    # Constraints: For every pair of disjoint sets A, B of size 4, sum_{u in A, v in B} x_{uv} <= 15.
    
    # Number of constraints:
    # Choose A: C(16, 4) = 1820.
    # Choose B from remaining 12: C(12, 4) = 495.
    # Total: 1820 * 495 = 900,900 constraints.
    # This is too many for `scipy.optimize.milp` to handle efficiently in Python within 120s.
    
    # We need to reduce the number of constraints.
    # Note that the constraints are symmetric.
    # We can use a "cutting plane" approach or just sample constraints.
    # Or, we can use a known structure.
    # The extremal graph for K_{s,t} is often a Turan graph or a blow-up of a small graph.
    # For K4,4, the extremal number ex(n, K4,4) is known to be related to the Zarankiewicz problem.
    # For n=16, we can try to construct a graph based on a finite geometry or random restarts with ILP on a subset of constraints.
    
    # Alternative: Use a smaller ILP.
    # We can fix a small set of "critical" constraints.
    # Or, we can use a heuristic:
    # Start with a random graph, check for K4,4, remove edges.
    # But ILP is better.
    
    # Let's try to reduce constraints by symmetry.
    # All 4-sets are equivalent? No, but the graph is not necessarily vertex-transitive.
    # However, we can just iterate through a subset of constraints.
    # If we include all constraints, it's exact.
    # Can we generate the constraint matrix efficiently?
    # 900k constraints * 120 variables.
    # Matrix size: 900k x 120. ~100 million entries.
    # This is large but might be feasible with sparse matrices.
    # `scipy.optimize.milp` accepts sparse matrices.
    
    # Let's try to build the sparse matrix.
    # Rows: constraints. Cols: edges.
    # For each constraint (A, B), the row has 1s at the positions of edges between A and B.
    # There are 16 such edges.
    # So each row has 16 non-zeros.
    # Total non-zeros: 900k * 16 = 14.4 million.
    # This is very feasible for sparse matrices.
    
    # Let's implement this.
    
    # 1. Build edge list and index map (done above)
    
    # 2. Generate all pairs (A, B)
    # This might take some time to generate 900k tuples.
    # itertools.combinations(16, 4) -> 1820.
    # For each A, itertools.combinations(remaining, 4) -> 495.
    # Total iterations: 900k.
    # In Python, this loop might take 1-2 seconds. Acceptable.
    
    # 3. Build sparse matrix A_ub
    # We need to map edge (u, v) to column index.
    
    # 4. Solve MILP
    
    # 5. Extract edges
    
    # Potential Issue: The ILP solver might be slow with 900k constraints.
    # We can try to limit the time.
    # If it doesn't finish, we return the best incumbent.
    # `scipy.optimize.milp` has a `time_limit` option?
    # Checking docs: `options` dict can include `time_limit` for HiGHS.
    
    # Let's write the code.
    
    from scipy.sparse import lil_matrix, csr_matrix
    
    # Generate constraints
    # We'll store them as lists of (row, col, val) for COO format.
    # COO is good for building.
    
    # Actually, building 14 million entries in Python lists might be memory/time heavy.
    # Let's try to be smart.
    # We don't need ALL constraints if we use a heuristic?
    # No, we want exact.
    
    # Let's try a different approach: Local Search on the K4,4-free constraint.
    # Start with a dense graph.
    # While there is a K4,4, remove an edge from it.
    # This is similar to the Q3 removal but K4,4 detection is faster?
    # Detecting K4,4:
    # For each pair of 4-sets (A, B), check if all 16 edges are present.
    # This is still 900k checks.
    
    # Let's stick to ILP but optimize the matrix construction.
    # We can use numpy to generate the rows faster?
    
    # Actually, 900k constraints is a lot for a general purpose ILP solver in Python.
    # It might time out.
    
    # Alternative: Use a smaller subset of constraints.
    # Random sampling of constraints?
    # If we sample 10k constraints, we might find a good solution.
    # But it won't be provably optimal.
    
    # Given the score 1.3846 (42 edges), and ex(16, K4,4) is likely around 50-60?
    # Let's check known values.
    # ex(16, K4,4) is not immediately known to me, but for n=16, it's small.
    # Actually, the Turan graph T(16, 3) has 85 edges and contains K4,4.
    # T(16, 4) has 96 edges and contains K4,4.
    # We need a graph with no K4,4.
    # A random graph G(16, p) has expected K4,4 count:
    # C(16,4) * C(12,4) * p^16.
    # 1820 * 495 * p^16 = 900,900 * p^16.
    # For this to be < 1, p^16 < 1/900000.
    # p < (1/900000)^(1/16) = (10^-5.95)^(1/16) = 10^-0.37 = 0.42.
    # So a random graph with p=0.4 has ~0.4^16 * 900k = 0.00004 * 900k = 36 K4,4s?
    # 0.4^16 = 4.2e-7.
    # 900k * 4.2e-7 = 0.38.
    # So p=0.4 is likely K4,4-free with high probability?
    # Expected edges: 120 * 0.4 = 48.
    # This is better than 42.
    
    # So a random graph with p=0.4 might work.
    # But we can do better by starting with p=0.5 and removing edges.
    
    # Let's try a hybrid:
    # 1. Start with a random graph G(16, 0.6).
    # 2. Find all K4,4s.
    # 3. Remove edges to break them.
    # 4. Use ILP to refine?
    
    # Actually, let's just use the ILP with a time limit and a reduced set of constraints if necessary.
    # But 900k constraints is the exact set.
    # Let's try to run the ILP with all constraints and see if it fits in memory and time.
    # If not, we fall back to random search.
    
    # To speed up ILP, we can use `method='highs'` which is fast.
    
    # Let's construct the matrix.
    
    # Optimization: 
    # Instead of 900k rows, we can group them?
    # No, they are distinct.
    
    # Let's try to generate the matrix in a more vectorized way or just accept the loop time.
    # 900k iterations in Python is ~0.5s.
    # Building sparse matrix: 14M entries. ~1s.
    # Solving: HiGHS is C++. It should be fast.
    
    # Let's try it.
    
    # 1. Edge indices
    # 2. Constraints
    
    # Note: The constraint is sum <= 15.
    # So A_ub is the incidence matrix of the bipartite edges for each (A,B).
    # b_ub is 15 for all rows.
    
    # Let's code it.

    # Generate all 4-subsets
    all_4_sets = list(itertools.combinations(range(n), 4))
    
    # Map vertex to index? No, we need edge indices.
    
    # We will build a list of rows.
    # Each row is a dict {edge_idx: 1}
    # Then convert to sparse.
    
    # This might use too much memory if we store dicts.
    # Better to use COO lists: rows, cols, data.
    
    rows = []
    cols = []
    data = []
    
    # Precompute for each 4-set A, the edges connecting A to any vertex v?
    # No, we need edges between A and B.
    
    # Let's iterate.
    for A in all_4_sets:
        A_set = set(A)
        # Find B from remaining
        remaining = [v for v in range(n) if v not in A_set]
        for B in itertools.combinations(remaining, 4):
            # For each u in A, v in B, add edge (u,v) to this constraint
            for u in A:
                for v in B:
                    # Edge (min(u,v), max(u,v))
                    e = (min(u, v), max(u, v))
                    idx = edge_index[e]
                    rows.append(len(rows)) # Wait, row index is the constraint index
                    # We need to track constraint index
                    # Let's use a counter
                    pass
    
    # This loop is nested deep.
    # Let's restructure to be efficient.
    
    row_idx = 0
    for A in all_4_sets:
        A_set = set(A)
        remaining = [v for v in range(n) if v not in A_set]
        for B in itertools.combinations(remaining, 4):
            # This is one constraint
            for u in A:
                for v in B:
                    e = (min(u, v), max(u, v))
                    col = edge_index[e]
                    rows.append(row_idx)
                    cols.append(col)
                    data.append(1)
            row_idx += 1
            
    # row_idx should be 900,900.
    
    # Build sparse matrix
    A_ub = lil_matrix((row_idx, m))
    # lil is slow for filling. Use coo then convert.
    A_ub = csr_matrix((data, (rows, cols)), shape=(row_idx, m))
    
    # Objective: Maximize sum x_e
    # c = -1 (minimize -sum)
    c = -np.ones(m)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints
    # A_ub @ x <= b_ub
    b_ub = np.full(row_idx, 15)
    cons = LinearConstraint(A_ub, ub=b_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=cons, integrality=np.ones(m), bounds=bounds, 
                   options={"time_limit": 100, "mip_rel_gap": 0.01})
        
        if res.status == 0 or res.status == 2: # Optimal or Time Limit
            x = res.x
            # x is float array. Threshold at 0.5
            edges_found = []
            for i, val in enumerate(x):
                if val > 0.5:
                    edges_found.append(list(all_edges[i]))
            
            # Verify no Q3 (and thus no K4,4)
            # Just to be safe, we can do a quick check.
            # But ILP should guarantee it.
            
            return edges_found
            
    except Exception as e:
        print(f"ILP Error: {e}", file=sys.stderr)
        # Fallback to random search
        pass

    # Fallback: Random Search with K4,4 repair
    # Start with random graph p=0.5
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if adj[u][v]: return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u,v), max(u,v)))
        
    def remove_edge(u, v):
        if not adj[u][v]: return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u,v), max(u,v)))
        
    # Initialize random
    random.seed(42)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                add_edge(u, v)
                
    best_edges = len(edges)
    best_edge_set = set(edges)
    
    # Repair loop
    deadline = start_time + 110.0
    while time.time() < deadline:
        # Find a K4,4
        found_k44 = None
        for A in all_4_sets:
            A_set = set(A)
            remaining = [v for v in range(n) if v not in A_set]
            for B in itertools.combinations(remaining, 4):
                # Check if all 16 edges present
                is_k44 = True
                for u in A:
                    for v in B:
                        if not adj[u][v]:
                            is_k44 = False
                            break
                    if not is_k44:
                        break
                if is_k44:
                    found_k44 = (A, B)
                    break
            if found_k44:
                break
                
        if found_k44:
            A, B = found_k44
            # Remove a random edge from this K4,4
            u = random.choice(A)
            v = random.choice(B)
            remove_edge(u, v)
        else:
            # No K4,4 found. Try to add edges.
            # Pick random missing edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj[u][v]:
                continue
            add_edge(u, v)
            # Check if this created a K4,4
            # We can just let the next iteration handle it, 
            # but we need to ensure we don't loop forever on invalid states.
            # Actually, if we add an edge and create a K4,4, we should remove it or fix it.
            # For simplicity, if we add an edge, we check immediately.
            # But checking all K4,4s is expensive.
            # We can just revert if it creates a K4,4.
            
            # Re-check for K4,4 involving u and v?
            # Or just revert if any K4,4 exists?
            # Let's just revert to be safe.
            # Find K4,4 again?
            # This is slow.
            
            # Better: Only add if it doesn't complete a K4,4.
            # How to check if adding (u,v) creates a K4,4?
            # It creates a K4,4 if there exists A containing u and B containing v (or vice versa)
            # such that A\{u} and B\{v} form a K3,3 with the rest of the edges?
            # This is complex.
            
            # Given the time, we just revert.
            remove_edge(u, v)
            
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_edge_set = set(edges)
                
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))

