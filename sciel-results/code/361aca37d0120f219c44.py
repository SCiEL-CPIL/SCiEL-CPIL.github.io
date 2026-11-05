import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Pre-compute triples for fast validity checking
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def is_valid_fast(adj_bits):
        """
        Checks if the graph represented by adj_bits is K3,3-free.
        Condition: No three vertices have three common neighbors.
        """
        for i, j, k in triples:
            # Common neighbors of i, j, k
            if (adj_bits[i] & adj_bits[j] & adj_bits[k]).bit_count() >= 3:
                return False
        return True

    def count_edges(adj_bits):
        return sum(adj_bits[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj_bits):
        edges = []
        for u in range(n):
            v = adj_bits[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # --- Strategy: Deterministic Construction using Finite Geometry ---
    # The Hermitian Unital in PG(2,4) has 21 points.
    # It is a Steiner system S(2, 4, 21).
    # The polarity graph (or related incidence graph) of a unital is known to be K3,3-free.
    # Specifically, the graph where vertices are points and edges connect points that are "orthogonal" 
    # with respect to the Hermitian polarity is a strongly regular graph.
    # For q=4, the parameters are v=21, k=6, lambda=1, mu=2.
    # Wait, let's verify the K3,3-free property.
    # In a graph with parameters (v, k, lambda, mu), if mu < 3, is it K3,3-free?
    # A K3,3 requires two disjoint sets of 3 vertices with all cross-edges.
    # If we pick 3 vertices in one part, their common neighbors must be < 3.
    # In a strongly regular graph, the number of common neighbors of two adjacent vertices is lambda.
    # The number of common neighbors of two non-adjacent vertices is mu.
    # For 3 vertices {u,v,w}, the common neighbors are N(u) ∩ N(v) ∩ N(w).
    # This is not directly given by lambda/mu.
    # However, it is a known result that the polarity graph of the Hermitian unital H(2, q^2) is K_{3,3}-free.
    # Reference: "K_{3,3}-free graphs" by various authors.
    # For q=4, v=21. We need n=20. We can simply remove one vertex.
    
    # Let's construct the Hermitian unital graph on 21 vertices.
    # Field GF(4). Elements: 0, 1, omega, omega^2 (where omega^2 + omega + 1 = 0).
    # We can represent GF(4) elements as integers 0, 1, 2, 3.
    # Multiplication table for GF(4):
    # 0: 0,0,0,0
    # 1: 0,1,2,3
    # 2 (omega): 0,2,3,1  (omega^2 = omega+1=3, omega^3=1)
    # 3 (omega^2): 0,3,1,2 (omega^4=omega, omega^5=omega^2=3, omega^6=1)
    
    gf4_mult = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    # Addition is XOR for GF(2^k)
    # Hermitian form: H(x, y) = x0*y0^4 + x1*y1^4 + x2*y2^4 ...
    # In PG(2, 4), points are lines through origin in GF(4)^3.
    # There are (4^3 - 1) / (4 - 1) = 21 points.
    # We can enumerate the 21 points as normalized vectors (1, a, b), (0, 1, a), (0, 0, 1).
    
    points = []
    # Type 1: (1, a, b) for a,b in GF(4) -> 16 points
    for a in range(4):
        for b in range(4):
            points.append([1, a, b])
    # Type 2: (0, 1, a) for a in GF(4) -> 4 points
    for a in range(4):
        points.append([0, 1, a])
    # Type 3: (0, 0, 1) -> 1 point
    points.append([0, 0, 1])
    
    # Total 21 points.
    
    # Hermitian polarity in PG(2, q^2) with q=2 (so field size 4):
    # Points x, y are orthogonal if sum x_i * y_i^q = 0.
    # Here q=2. So x0*y0^2 + x1*y1^2 + x2*y2^2 = 0 in GF(4).
    # Note: In GF(4), x^2 is the Frobenius automorphism.
    # Squaring table for GF(4):
    # 0^2 = 0
    # 1^2 = 1
    # 2^2 = 3 (omega^2)
    # 3^2 = 2 (omega^4 = omega)
    gf4_sq = [0, 1, 3, 2]
    
    def gf4_add(a, b):
        return a ^ b
        
    def gf4_mul(a, b):
        return gf4_mult[a][b]
        
    def gf4_pow2(x):
        return gf4_sq[x]
        
    def hermitian_dot(x, y):
        # x, y are lists of 3 elements in GF(4)
        # Sum x_i * y_i^2
        s = 0
        for i in range(3):
            term = gf4_mul(x[i], gf4_pow2(y[i]))
            s = gf4_add(s, term)
        return s
        
    # Construct adjacency matrix for the 21 vertices
    adj_21 = [0] * 21
    for i in range(21):
        for j in range(i + 1, 21):
            if hermitian_dot(points[i], points[j]) == 0:
                adj_21[i] |= (1 << j)
                adj_21[j] |= (1 << i)
                
    # This graph has 21 vertices. We need 20.
    # Remove vertex 20 (or any vertex). The induced subgraph on 20 vertices is also K3,3-free.
    # Let's check the edge count.
    # Regular graph? The polarity graph of a unital is regular of degree q+1 = 5?
    # Wait, for q=2, degree is q+1 = 3? No.
    # The Hermitian unital has parameters (v, k) = (q^3+1, q+1).
    # The associated graph (orthogonality) is strongly regular.
    # Let's just calculate the edges in the 20-vertex subgraph.
    
    # Remove vertex 20
    adj_20 = [0] * 20
    for i in range(20):
        # Mask out bit 20
        adj_20[i] = adj_21[i] & ((1 << 20) - 1)
        
    # Ensure symmetry (just in case)
    for i in range(20):
        for j in range(i):
            if adj_20[i] & (1 << j):
                adj_20[j] |= (1 << i)
            else:
                adj_20[j] &= ~(1 << i)
                
    # Verify validity
    if not is_valid_fast(adj_20):
        # Fallback: If my geometric construction is wrong or implementation error,
        # use a simple valid graph.
        # A path is valid.
        adj_20 = [0] * 20
        for i in range(19):
            adj_20[i] |= (1 << (i+1))
            adj_20[i+1] |= (1 << i)
            
    current_adj = adj_20[:]
    
    # --- Refinement Phase: Greedy Additions ---
    # The geometric graph might not be maximal. We can try to add edges.
    # Since the graph is sparse (degree ~3-5), there is likely room to add edges.
    # We use a randomized greedy approach to add edges if they don't create a K3,3.
    
    # To speed up checking "does adding (u,v) create a K3,3?",
    # we can check if there exists a triple {a,b,c} such that {u,v} are in N(a)∩N(b)∩N(c)?
    # No, adding (u,v) affects the neighborhood of u and v.
    # A new K3,3 must involve u or v in one of the parts.
    # Case 1: u is in part A (size 3), v is in part B (size 3).
    #   Then u and v are already connected? No, we are adding it.
    #   If u in A, v in B, then all edges between A and B must exist.
    #   This implies u is connected to all 3 in B, and v is connected to all 3 in A.
    #   Since we are adding (u,v), this edge is new.
    #   So we need:
    #   1. v is connected to 3 vertices in A (including u? No, u is in A).
    #      Wait, if u in A, then u is not in B.
    #      Edges required: (u, b1), (u, b2), (u, b3), (a1, v), (a2, v), (a3, v), (a1, b1)...
    #      This is complex to check incrementally.
    #
    # Simpler: Just try adding (u,v), check full validity.
    # With n=20, full validity check is ~1140 ops.
    # 190 possible edges. 190 * 1140 ~ 200k ops per sweep.
    # We can do many sweeps.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Randomize order to escape local minima
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if current_adj[u] & (1 << v):
                continue
            # Try adding
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            if is_valid_fast(current_adj):
                improved = True
            else:
                # Revert
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
    # --- Refinement Phase 2: Swap-based Local Search ---
    # Try to swap edges to increase density?
    # Swap (u,v) and (x,y) -> remove (u,v), add (x,y)?
    # Or remove (u,v), add (w,z)?
    # Let's try "remove one, add one" moves.
    
    # This is a 2-opt style move.
    # We want to maintain validity.
    # Given the time, simple greedy addition is likely sufficient for a big jump from the seed.
    # The seed score 2.27 suggests ~100+ edges.
    # The geometric graph has ~30-40 edges.
    # Wait, if the geometric graph has only 30 edges, and the seed has 100, 
    # my "deterministic construction" is a *worse* start!
    
    # Let's check the edge count of the geometric graph.
    # If it's low, we should start from the BEST known valid graph from the seed logic?
    # No, the seed logic is SA. 
    # The "inspiration" says: "Adding complete-graph start with aggressive repair... measured 2.25 -> 2.306".
    # This implies starting dense and repairing is better.
    
    # My current geometric graph is sparse. 
    # Let's check its edge count.
    # If it's < 100, I should probably use a different strategy or combine.
    
    # Actually, let's look at the "Inspiration 1" which scored 2.2778 (same as current).
    # The current program *is* the SA program.
    
    # Let's re-evaluate the geometric graph.
    # Hermitian Unital H(2, 4) polarity graph.
    # Parameters: v=21.
    # Degree k = q+1 = 3?
    # If k=3, edges = 21*3/2 = 31.5 -> 31 or 32 edges.
    # This is very sparse. The seed has ~113 edges (score 2.27 * 50? No, score is value/seed.
    # If seed value is ~50, score 1.0.
    # If score is 2.27, value is ~113.
    # Max edges in K3,3-free graph on 20 vertices?
    # Ex(20, K3,3) is likely around 100-120.
    
    # So my geometric construction starts at ~30 edges. 
    # Greedy addition will have to add ~80 edges.
    # Is that possible?
    # K3,3-free allows high density.
    # The greedy addition from a sparse valid graph might get stuck.
    
    # Alternative Strategy: 
    # Use the "Complete Graph Start with Repair" strategy mentioned in the cell findings.
    # Start with K20 (190 edges).
    # Remove edges to fix violations.
    # This is a "Max-Cut" like problem or "Maximum K3,3-free subgraph".
    
    # Let's implement the "Dense Start + Repair" strategy as the primary engine, 
    # using the geometric graph as a *fallback* or *initialization* if repair fails?
    # No, repair from dense is hard.
    
    # Let's try a hybrid:
    # 1. Start with a random dense graph (p=0.8).
    # 2. Repair it by removing edges that cause violations.
    # 3. Then try to add edges back.
    
    # Repair Algorithm:
    # While invalid:
    #   Find a violating triple {a,b,c} with >= 3 common neighbors {x,y,z}.
    #   Remove one edge from the K3,3 structure.
    #   Which edge? Randomly pick one of the 9 cross edges.
    #   This is slow if we do it one by one.
    
    # Better Repair:
    # Find all violations.
    # For each violation, mark the edges involved as "bad".
    # Remove all "bad" edges? Too aggressive.
    # Remove one edge per violation, prioritizing edges in multiple violations.
    
    # Let's implement "Dense Start + Greedy Repair + Greedy Add".
    
    def find_violation(adj_bits):
        """Returns (i, j, k, common_neighbors_list) if violation exists, else None."""
        for i, j, k in triples:
            cn = adj_bits[i] & adj_bits[j] & adj_bits[k]
            if cn.bit_count() >= 3:
                # Get the list of common neighbors
                neighs = []
                c = cn
                while c:
                    low = c & (-c)
                    v = low.bit_length() - 1
                    neighs.append(v)
                    c ^= low
                return (i, j, k, neighs)
        return None

    def repair_graph(adj_bits):
        """Removes edges until valid. Returns valid adj_bits."""
        # Use a set of edges to remove
        edges_to_remove = set()
        
        # Iterative removal
        # To avoid infinite loops or getting stuck, we limit iterations
        max_iter = 1000
        for _ in range(max_iter):
            v = find_violation(adj_bits)
            if v is None:
                break
            i, j, k, neighs = v
            # The K3,3 is formed by {i,j,k} and any 3 from neighs.
            # We need to break this.
            # Pick a random neighbor x from neighs.
            x = random.choice(neighs)
            # The edges (i,x), (j,x), (k,x) are part of the violation.
            # Remove one of them.
            # Randomly choose which of i,j,k to disconnect from x.
            target = random.choice([i, j, k])
            
            # Remove edge (target, x)
            if adj_bits[target] & (1 << x):
                adj_bits[target] &= ~(1 << x)
                adj_bits[x] &= ~(1 << target)
            
        return adj_bits

    # Initialize with Dense Random
    random.seed(42)
    p_init = 0.9
    adj_dense = [0] * 20
    for u, v in all_pairs:
        if random.random() < p_init:
            adj_dense[u] |= (1 << v)
            adj_dense[v] |= (1 << u)
            
    # Repair
    adj_repaired = repair_graph(adj_dense[:])
    
    # Check if repaired is better than geometric
    edges_repaired = count_edges(adj_repaired)
    edges_geom = count_edges(adj_20)
    
    # Use the one with more edges as the base for addition
    if edges_repaired > edges_geom:
        current_adj = adj_repaired
    else:
        current_adj = adj_20[:]
        
    # Final Greedy Addition
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if current_adj[u] & (1 << v):
                continue
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            if is_valid_fast(current_adj):
                improved = True
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
    # Final check and return
    if not is_valid_fast(current_adj):
        # Emergency repair
        current_adj = repair_graph(current_adj)
        
    return adj_to_edges(current_adj)

# Test locally
if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    # Verify
    n = 20
    adj = [0]*n
    for u,v in edges:
        adj[u] |= 1<<v
        adj[v] |= 1<<u
    valid = is_valid_fast(adj)
    print(f"Valid: {valid}")
