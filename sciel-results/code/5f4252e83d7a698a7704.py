"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 18
    
    def is_k4_free(edge_set, n=18):
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if all((min(p[0],p[1]), max(p[0],p[1])) in edge_set for p in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]):
                return False
        return True

    def find_largest_triangle_free(edge_set, n=18, timeout=1.0):
        """Find a large triangle-free induced subgraph via greedy + local search."""
        start = time.time()
        best_set = set()
        best_size = 0
        
        # Greedy: try to build a large triangle-free set
        remaining = list(range(n))
        current = set()
        
        while remaining:
            if time.time() - start > timeout:
                break
            # Pick vertex with fewest edges to current set
            best_v = None
            best_edges_to_current = float('inf')
            for v in remaining:
                edges_to_current = sum(1 for u in current if (min(u,v), max(u,v)) in edge_set)
                # Also count triangles that would form
                tri_count = 0
                for i, u in enumerate(current):
                    if (min(u,v), max(u,v)) in edge_set:
                        for w in current:
                            if w != u and (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                                tri_count += 1
                                break
                if tri_count > 0:
                    continue  # Can't add v
                if edges_to_current < best_edges_to_current:
                    best_edges_to_current = edges_to_current
                    best_v = v
            
            if best_v is None:
                break
            current.add(best_v)
            remaining.remove(best_v)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        # Try to extend best_set by adding vertices
        for v in range(n):
            if v in best_set:
                continue
            can_add = True
            for i, u in enumerate(best_set):
                if (min(u,v), max(u,v)) in edge_set:
                    for w in best_set:
                        if w != u and (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                new_set = best_set | {v}
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
        
        return best_set, best_size

    def max_triangle_free_size(edge_set, n=18, timeout=2.0):
        best_set, best_size = find_largest_triangle_free(edge_set, n, timeout)
        return best_size

    def score(edge_set):
        return max_triangle_free_size(edge_set)

    random.seed(42)
    
    # Generate Clebsch graph on 16 vertices using folded 5-cube
    # Vertices 0..15, u~v if popcount(u^v) in {1,2} and u^v != 0
    # Actually, for folded Q5: u~v if dist(u,v) = 1 or 4 in Q5
    # But we identify antipodes, so effectively u~v if popcount(u^v) = 1 (since dist 4 maps to dist 1)
    # Let's use: u~v if popcount(u^v) == 1 OR popcount(u^v) == 2
    # Check: for u=0, neighbors are all weight-1 and weight-2 vectors: 4+6=10. Too many.
    # Correct: Clebsch is (16,5,0,2). 
    # Use: u~v if popcount(u^v) in {1,2} but only if... 
    # Let me just hardcode a known Clebsch graph.
    
    # Clebsch graph edges (verified from known sources)
    clebsch_16 = [
        (0,1),(0,2),(0,4),(0,8),(0,12),
        (1,3),(1,5),(1,9),(1,13),
        (2,3),(2,6),(2,10),(2,14),
        (3,7),(3,11),(3,15),
        (4,5),(4,6),(4,10),(4,14),
        (5,7),(5,11),(5,15),
        (6,7),(6,11),(6,15),
        (7,9),(7,13),
        (8,9),(8,10),(8,14),
        (9,12),(9,16),  # 16 is out of range, fix
    ]
    # Let me use a simpler approach: use a random regular graph with degree 5 on 18 vertices
    # that is likely K4-free, then optimize.
    
    # Start with a random 5-regular graph
    def generate_random_regular(n, d, seed=42):
        rng = random.Random(seed)
        edges = set()
        degrees = [0] * n
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        rng.shuffle(all_pairs)
        
        for i,j in all_pairs:
            if degrees[i] < d and degrees[j] < d:
                edges.add((i,j))
                degrees[i] += 1
                degrees[j] += 1
            if all(deg == d for deg in degrees):
                break
        
        return edges
    
    # Try multiple starting points
    best_edges = None
    best_score = 999
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline:
            break
        
        # Generate random graph with moderate density
        e = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.35:
                    e.add((i,j))
        
        if not is_k4_free(e):
            continue
        
        s = score(e)
        if s < best_score:
            best_score = s
            best_edges = set(e)
    
    if best_edges is None:
        # Fallback: K6,6,6
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = set()
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.add((min(a,b), max(a,b)))
        best_edges = edges
        best_score = score(best_edges)
    
    current_edges = set(best_edges)
    current_score = best_score
    
    all_possible_edges = set((i,j) for i in range(n) for j in range(i+1, n))
    
    # Targeted local search: find largest triangle-free set, then try to add edges within it
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Find the largest triangle-free set
        tf_set, tf_size = find_largest_triangle_free(current_edges, n, timeout=0.5)
        
        if tf_size <= current_score:
            current_score = tf_size
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        # Try to add edges within the triangle-free set to break it up
        improved = False
        tf_list = sorted(tf_set)
        
        for i in range(len(tf_list)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(tf_list)):
                u, v = tf_list[i], tf_list[j]
                edge = (min(u,v), max(u,v))
                if edge in current_edges:
                    continue
                
                new_edges = current_edges | {edge}
                if not is_k4_free(new_edges):
                    continue
                
                s = score(new_edges)
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    break
            if improved:
                break
        
        if not improved:
            # Try removing edges to allow more additions later
            candidates = list(current_edges)
            random.shuffle(candidates)
            for e in candidates[:10]:
                if time.time() > deadline:
                    break
                new_edges = current_edges - {e}
                s = score(new_edges)
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    break
            
            if not improved:
                # Random restart
                e = set()
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.35:
                            e.add((i,j))
                if is_k4_free(e):
                    s = score(e)
                    if s < current_score:
                        current_edges = e
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best_edges = set(current_edges)
    
    return [[u, v] for u, v in sorted(best_edges)]
# EVOLVE-BLOCK-END
