"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Uses Greedy Construction followed by Local Search (Hill Climbing) with Random Restarts.
    """
    
    def neighbors(p):
        a, b = p
        return [(a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)]

    def count_edges(pset):
        edges = 0
        for p in pset:
            for n in neighbors(p):
                if n in pset:
                    edges += 1
        return edges // 2

    def greedy_build(random_seed):
        """Build a set of 32 points greedily."""
        random.seed(random_seed)
        pset = set()
        
        # Start with a central point
        pset.add((0, 0))
        
        # Candidate locations: A box around the origin
        # We want to stay compact. Let's search in a box of [-10, 10]
        # But we need to ensure we can find 32 points.
        # A hex of radius 4 has 37 points. So [-4, 4] is sufficient for the bulk.
        
        # To make it robust, we generate candidates dynamically or use a fixed large box.
        # Let's use a box [-10, 10] for candidates, but prefer closer ones.
        
        # Actually, a better greedy: 
        # 1. Start with (0,0).
        # 2. At each step, consider all empty lattice points in a bounding box.
        # 3. Pick the one with max neighbors.
        
        # Optimization: Instead of checking the whole box every time, 
        # we can maintain a list of "frontier" points.
        
        # For simplicity and speed within 120s, we can just iterate a reasonable box.
        # 32 points fit easily in a 10x10 box.
        
        bounds = 10
        candidates = []
        for a in range(-bounds, bounds + 1):
            for b in range(-bounds, bounds + 1):
                if (a, b) not in pset:
                    candidates.append((a, b))
        
        # We will pick 31 more points
        for _ in range(31):
            best_score = -1
            best_pos = None
            # Shuffle candidates to break ties randomly
            random.shuffle(candidates)
            
            for pos in candidates:
                # Count neighbors in pset
                deg = 0
                for n in neighbors(pos):
                    if n in pset:
                        deg += 1
                
                # Prefer points closer to origin if scores are equal? 
                # The shuffle handles random tie-breaking.
                # But we should prefer compactness. 
                # Let's add a small penalty for distance from origin to keep it compact.
                dist = abs(pos[0]) + abs(pos[1])
                score = deg * 10 - dist
                
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos is None:
                break
                
            pset.add(best_pos)
            # Remove best_pos from candidates
            candidates.remove(best_pos)
            
        return pset

    def local_search(pset, max_iter=1000):
        """Improve the set using hill climbing moves."""
        pset = set(pset)
        current_edges = count_edges(pset)
        
        # Pre-compute neighbors for all points? 
        # We need to check moves: move point P to empty location Q.
        # Delta = deg(Q) - deg(P) + (1 if P-Q is edge else 0) ... wait.
        # If we move P to Q:
        # Lose edges from P to its neighbors.
        # Gain edges from Q to its neighbors.
        # If P and Q were adjacent, that edge is lost (counted in deg(P) and deg(Q)? No, Q is empty, so P-Q edge is not in current graph. 
        # Wait, if we move P to Q, the edge (P, X) is lost for all X in N(P). 
        # The edge (Q, Y) is gained for all Y in N(Q).
        # Note: If X == Q, the edge (P, Q) is lost. But Q is empty, so (P, Q) is not an edge.
        # So delta = sum(1 for y in N(Q) if y in pset) - sum(1 for x in N(P) if x in pset).
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try to move each point
            points_list = list(pset)
            random.shuffle(points_list) # Randomize order
            
            for P in points_list:
                deg_P = sum(1 for n in neighbors(P) if n in pset)
                
                # Candidate locations for Q:
                # 1. Neighbors of P (local move)
                # 2. Random locations in a box around P or global?
                # Let's try neighbors of P first, then a few random ones.
                
                candidates_Q = neighbors(P)
                # Add some random points in a small radius
                for _ in range(5):
                    da = random.randint(-3, 3)
                    db = random.randint(-3, 3)
                    candidates_Q.append((P[0]+da, P[1]+db))
                
                best_delta = 0
                best_Q = None
                
                for Q in candidates_Q:
                    if Q == P:
                        continue
                    if abs(Q[0]) > 64 or abs(Q[1]) > 64:
                        continue
                    if Q in pset:
                        continue
                    
                    deg_Q = sum(1 for n in neighbors(Q) if n in pset)
                    # Check if P and Q are adjacent? 
                    # If P and Q are adjacent, the edge (P,Q) is currently NOT in the graph because Q is empty.
                    # When we move P to Q, we lose edges from P to N(P). 
                    # We gain edges from Q to N(Q).
                    # The edge (P,Q) doesn't exist, so no special adjustment needed for "losing an edge between P and Q".
                    # However, if some neighbor X of P is also a neighbor of Q, 
                    # the edge (P,X) is lost and (Q,X) is gained. This is handled by deg_P and deg_Q.
                    
                    delta = deg_Q - deg_P
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_Q = Q
                
                if best_Q is not None and best_delta > 0:
                    pset.remove(P)
                    pset.add(best_Q)
                    current_edges += best_delta
                    improved = True
                    # Restart loop to update degrees? 
                    # Since we only changed one point, we could update incrementally, 
                    # but for simplicity and correctness, we break and restart the while loop 
                    # or just continue. Continuing is fine, but degrees of other points 
                    # relative to P's old/new position change. 
                    # Actually, deg of other points changes only if they were neighbors of P or Q.
                    # For robustness, let's just restart the inner loop or break out.
                    break
                    
        return pset, current_edges

    best_edges = -1
    best_result = []
    
    # Run multiple restarts
    for seed in range(50):
        pset = greedy_build(seed)
        pset, edges = local_search(pset)
        
        if edges > best_edges:
            best_edges = edges
            best_result = list(pset)
            
        # Early stop if we hit the theoretical max
        if best_edges == 76:
            break

    result = [[a, b] for a, b in best_result]
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
# EVOLVE-BLOCK-END
