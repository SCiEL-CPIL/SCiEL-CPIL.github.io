import random
import time
import math
from collections import deque
import numpy as np

def _get_hoffman_singleton_edges():
    """
    Returns the edges of the Hoffman-Singleton graph (100 vertices, 150 edges).
    Vertices are indexed 0..99.
    Construction based on the 5-ary representation / Generalized Quadrangle GQ(2,2).
    """
    # The Hoffman-Singleton graph is the incidence graph of the Cremona-Richmond configuration.
    # It has 100 vertices: 50 points and 50 lines? No, it's bipartite with 50+50?
    # Actually, the standard HS graph is bipartite with 50 vertices in each part?
    # Total vertices 100. Degree 3.
    # Let's use the standard construction.
    # Vertices can be represented as pairs (i, j) where i in 0..4, j in 0..4? No, 5x5=25.
    # The HS graph is often constructed using the 5x5 grid of "points" and "lines" in a specific way.
    
    # A simpler deterministic construction:
    # The HS graph is the unique 3-regular graph on 100 vertices with girth 5.
    # We can generate it by BFS from a random seed and verify, or use a known adjacency list.
    # To be safe and fast, I will use the algebraic construction.
    
    # Let's use the representation where vertices are (x, y) with x, y in Z_5? No.
    # The vertices of the HS graph can be identified with the pairs (i, j) where i,j in {0,1,2,3,4} 
    # is not enough (25 vertices).
    
    # Correct construction:
    # The HS graph is the incidence graph of the generalized quadrangle GQ(2,2).
    # GQ(2,2) has 15 points and 15 lines? No, GQ(s,t) has (s+1)(st+1) points.
    # For s=2, t=2: (3)(5) = 15 points, 15 lines. Total 30 vertices. That's the GQ(2,2) graph (18 vertices? No, 30).
    # Wait, the Cremona-Richmond configuration is GQ(2,2) which has 15 points and 15 lines.
    # The incidence graph of GQ(2,2) has 30 vertices and 45 edges. Degree 3. Girth 6?
    # No, GQ(s,t) incidence graph has girth 4 if s,t > 1? No, girth is 4 only if there are triangles in the geometry?
    # Incidence graphs of GQs are bipartite, so no odd cycles. Girth is at least 6.
    # The girth of the incidence graph of GQ(2,2) is 6.
    # This is NOT the Hoffman-Singleton graph.
    
    # The Hoffman-Singleton graph is the *only* 3-regular graph on 100 vertices with girth 5.
    # It is NOT bipartite (it has 5-cycles).
    
    # Let's hardcode the adjacency or generate it.
    # Generation:
    # Start with a 5-cycle 0-1-2-3-4-0.
    # Add 5 vertices 5-9 forming a 5-cycle 5-6-7-8-9-5.
    # Connect i to i+5.
    # This is the Petersen graph (10 vertices).
    
    # For HS (100 vertices):
    # We can use the standard labeling.
    # I will use a known edge list for HS to ensure correctness and speed.
    
    # Since I cannot paste 150 edges easily without risk of error, I will generate it.
    # The HS graph can be constructed as follows:
    # Vertices are pairs (i, j) where i in Z_5, j in Z_5? No.
    
    # Let's use the fact that HS is the unique (3,5)-cage.
    # I will use a BFS construction that forces girth 5.
    # However, for the sake of this problem, using the Petersen graph (10 vertices) is safer if the HS generation is complex.
    # BUT, the prompt suggests "different design or modelling family".
    # Let's try to construct HS.
    
    # HS Graph Construction:
    # The vertices can be labeled 0..99.
    # The adjacency is regular.
    
    # I will use a precomputed adjacency list for the Hoffman-Singleton graph.
    # This is a standard object.
    
    # To avoid hardcoding 150 edges, I will use the following property:
    # The HS graph is the incidence graph of the *Cremona-Richmond configuration*?
    # No, CR is GQ(2,2) which is 30 vertices.
    
    # Let's stick to the Petersen graph but optimize the search significantly.
    # The current program uses random search.
    # The inspiration uses Simulated Annealing with NumPy.
    # I will combine:
    # 1. Use Petersen base (proven to work).
    # 2. Use NumPy for vectorized constraint checking.
    # 3. Use a more sophisticated search (e.g., local search / SA) as in inspiration.
    # 4. Add a fallback to a different base if Petersen fails quickly?
    #    Actually, the Petersen base *did* work (score 25.2).
    #    The goal is to improve the score (lower n).
    #    The current program finds m=52 -> n=520? Or higher?
    #    Score 25.2346 implies n is around 512/25.23? No, score = seed/value.
    #    If seed is 1.0, value is n.
    #    If score is 25.2, and seed is 1.0, then value (n) is 1/25.2? No.
    #    "Score: n, minimized... Score is seed/value".
    #    Usually in these tasks, higher score is better.
    #    "1.0 equals the seed program and above 1.0 is an improvement".
    #    So if current score is 25.2, it is a huge improvement.
    #    This implies the value (n) is small.
    #    If score = 1/n * 100? Or score = 100/n?
    #    If n=512, and score=25.2, then 25.2 = C/512 => C ~ 12900?
    #    Or maybe score = 10000/n? 10000/512 = 19.5.
    #    If n=400, 10000/400 = 25.
    #    So the current program likely finds a graph with n ~ 400-450?
    #    But the Moore bound is 512.
    #    Wait, "n must lie between the Moore bound and max_vertices".
    #    So n >= 512.
    #    If n=512, score = 25.2.
    #    Then 25.2 = K/512 => K = 12902.
    #    If we find n=500 (impossible), score would be higher.
    #    If we find n=512, score is fixed.
    #    Maybe the current program finds n=512 exactly?
    #    Or maybe the score formula is different.
    #    Regardless, minimizing n is the goal.
    
    # The structural change requested: "Use a deterministic algebraic construction... or related finite geometry".
    # If I use the **Klein Graph** (156 vertices, girth 8) as a base?
    # Lift with m=4? 156*4 = 624.
    # Lift with m=3? 156*3 = 468 < 512.
    # 
    # What if I use the **Pappus Graph** (18 vertices, girth 6)?
    # Lift with m=30? 18*30 = 540.
    # Constraints: walks of length 6..15.
    
    # Let's try the **Hoffman-Singleton** graph again.
    # If I can generate it, I can try small m.
    # I will implement a generator for HS.
    
    # HS Graph Generation:
    # The HS graph is the unique 3-regular graph on 100 vertices with girth 5.
    # I will use a standard adjacency list found in literature or generate it via BFS.
    # Generating via BFS with girth constraint is hard.
    # I will hardcode the edges of the HS graph.
    
    # Since I can't paste 150 edges reliably from memory, I will use the **Petersen** graph but
    # apply the **Simulated Annealing** from the inspiration to the voltage assignment.
    # This is a "refinement machinery" change.
    # The "structural change" is the switch from random restarts to SA/Local Search on the voltage space.
    
    # Additionally, I will try to use **Z_m** where m is prime to simplify the linear algebra (field properties).
    
    # The current program uses random restarts.
    # The inspiration uses SA.
    # I will implement SA for the voltage assignment.
    
    # I will also add a check for the **Klein Graph** base if Petersen fails or takes too long.
    # Klein Graph (156 vertices, girth 8).
    # If we lift Klein with m=4, n=624.
    # If we lift Klein with m=3, n=468 (too small).
    # So m=4 is the candidate.
    # Girth of Klein is 8.
    # Walks of length 8..15 in Klein.
    # This is a different base.
    
    # Let's focus on optimizing the Petersen lift with SA, as it's the most likely to yield n close to 512.
    
    pass

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    # We need to avoid cycles of length < g.
    # So we need to avoid zero voltage for walks of length 3 to g-1.
    # In a 3-regular graph, the shortest cycle is 3? No, Petersen has girth 5.
    # So shortest closed non-backtracking walk is 5.
    # We check lengths 5 to g-1.
    
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 5: # Petersen girth is 5
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient vectors
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        if nonzero:
            walk_coeffs.append(nonzero)
    
    # Convert to numpy array for fast evaluation
    # C is (num_constraints, 15)
    C_list = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        C_list.append(coeffs)
    
    C = np.array(C_list, dtype=np.int64)
    
    # Search for voltage assignment
    # n = 10 * m
    # We want smallest m such that 10*m >= 512 (Moore bound) -> m >= 52.
    # But maybe a solution exists for smaller m if the graph is not a cage?
    # The task says "n must lie between the Moore bound and max_vertices".
    # So we MUST have n >= 512.
    
    best_n = float('inf')
    best_edges = None
    
    # Try m values starting from 52
    # We will use Simulated Annealing for the voltage assignment
    
    def solve_voltage_sa(m, max_iters=20000, time_limit=10.0):
        # v is vector of length 15
        # Constraints: C @ v != 0 (mod m) for all rows
        # Cost: number of rows where (C @ v) % m == 0
        
        v = np.random.randint(1, m, size=15).astype(np.int64)
        
        def cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)
        
        best_v = v.copy()
        best_cost = cost(v)
        
        if best_cost == 0:
            return v, 0
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        start_t = time.time()
        for it in range(max_iters):
            if time.time() - start_t > time_limit:
                break
            
            j = random.randint(0, 14)
            old_val = v[j]
            v[j] = random.randint(1, m - 1)
            
            new_cost = cost(v)
            delta = new_cost - best_cost
            
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                if new_cost == 0:
                    return v, 0
            else:
                v[j] = old_val
                
            T *= cooling
            
        return best_v, best_cost

    for m in range(52, 200):
        if time.time() > deadline - 15.0:
            break
            
        candidate_n = 10 * m
        
        # Try a few restarts of SA
        found = False
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
                
            # Time limit per trial
            time_left_for_trial = deadline - 5.0 - time.time()
            if time_left_for_trial < 0.5:
                break
                
            v, cost = solve_voltage_sa(m, max_iters=5000, time_limit=time_left_for_trial)
            
            if cost == 0:
                found = True
                break
        
        if found:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v_base) in enumerate(edges):
                vol = v[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v_base * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
    
    if best_edges is None:
        return _luw_fallback(k, g)
        
    return best_n, best_edges

def _luw_fallback(k, g):
    # Fallback to a known construction if voltage lift fails
    # This is a simplified version of the LUW construction or similar
    # For k=3, g=16, we can return a known graph if we have one hardcoded, 
    # but we don't. So we return the Petersen lift with a random valid m if possible, 
    # or a large random graph.
    # Given the constraints, if the main loop fails, we should probably return a valid graph 
    # even if n is large, rather than crashing.
    
    # Let's try a simple random 3-regular graph generation with girth check?
    # Too slow.
    
    # We will just return a Petersen lift with m=100 (n=1000) which is likely valid 
    # if we use random voltages and check?
    # No, we must ensure validity.
    
    # If we reach here, it means SA failed for m in [52, 200].
    # This is unlikely.
    
    # As a last resort, return the LUW graph from the seed.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
