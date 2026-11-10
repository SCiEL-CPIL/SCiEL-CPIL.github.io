"""No-isosceles set: maximize points in [0,N)^2 with no isosceles triangles."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 300  # Safety margin before 330s kill
    best_points = []
    
    # Precompute all squared distances between all pairs of grid points
    # For efficiency, we'll work with squared distances to avoid sqrt
    
    # We'll use a greedy approach with random restarts
    # Key data structure: for each point in current set, track which other points 
    # are "forbidden" because they'd form an isosceles triangle
    
    def greedy_construction(order, max_time):
        """Greedily add points in the given order, skipping conflicts."""
        selected = []
        selected_set = set()
        
        # For each selected point, we track distances to it
        # A new point P conflicts if there exist A, B in selected such that
        # dist(P,A) == dist(P,B) or dist(P,A) == dist(A,B) or dist(P,B) == dist(A,B)
        # i.e., P, A, B form isosceles with P as apex, or A as apex, or B as apex
        
        # More precisely: three points P, A, B form isosceles if any two of the three 
        # pairwise distances are equal.
        
        # Efficient check: for new point P, for each pair (A, B) in selected,
        # check if d(P,A)==d(P,B) or d(P,A)==d(A,B) or d(P,B)==d(A,B)
        
        # But this is O(k^2) per insertion where k = len(selected).
        # Better: for each existing point A in selected, and for each other existing point B,
        # the "forbidden" points from triangle (P,A,B) being isosceles are:
        #   - Points equidistant from A and B (perpendicular bisector) -> d(P,A)==d(P,B)
        #   - Points at distance d(A,B) from A -> d(P,A)==d(A,B)  
        #   - Points at distance d(A,B) from B -> d(P,B)==d(A,B)
        
        # Let's use a different approach: maintain a set of "forbidden" points.
        # When we add point P to selected, for each existing point A in selected:
        #   - All points Q where d(Q,A) == d(Q,P) are forbidden (Q,P,A isosceles at Q)
        #   - All points Q where d(Q,P) == d(A,P) are forbidden (Q,A,P isosceles at A... wait)
        
        # Let me think more carefully. We have existing set S. We want to add P.
        # Conflict exists if there exist A, B in S such that {P,A,B} is isosceles.
        # {P,A,B} is isosceles iff:
        #   d(P,A) == d(P,B)  [apex at P]
        #   d(P,A) == d(A,B)  [apex at A]  
        #   d(P,B) == d(A,B)  [apex at B]
        
        # So for each pair (A,B) in S, P is forbidden if any of those conditions hold.
        # That's O(k^2) checks per candidate, which is too slow for large k.
        
        # Alternative: for each A in S, precompute which points P would cause conflict 
        # with A and some B in S.
        
        # Let's try: for each A in S, and for each B in S (B != A):
        #   Points P with d(P,A) == d(P,B): P lies on perpendicular bisector of AB
        #   Points P with d(P,A) == d(A,B): P lies on circle centered at A with radius d(A,B)
        #   Points P with d(P,B) == d(A,B): P lies on circle centered at B with radius d(A,B)
        
        # This is getting complex. Let's just do the O(k^2) check but with early termination
        # and hope that k stays manageable.
        
        # Actually, let's use a smarter approach:
        # For each selected point A, maintain a dict: dist_to_A -> count of selected points at that distance
        # No wait, that doesn't directly help.
        
        # Let me just do the direct check but optimize with numpy or just accept O(k^2)
        # For N=64, grid has 4096 points. If we can select ~50-100 points, k^2 = 10000 checks per candidate.
        # 4096 candidates * 10000 = 40M operations per restart. That might be too slow.
        
        # Let's use a different strategy: maintain a "conflict count" for each grid point.
        # conflict_count[P] = number of pairs (A,B) in selected such that {P,A,B} is isosceles.
        # When we add a new point P to selected, for each existing point A:
        #   For each existing point B (B != A):
        #     If d(P,A)==d(P,B): increment conflict_count for all Q where... no this is circular.
        
        # OK let me just do the straightforward O(k^2) check per candidate point.
        # With random ordering and early termination, it should work.
        
        start_time = time.time()
        
        for P in order:
            if time.time() > max_time:
                break
            px, py = P
            if P in selected_set:
                continue
            
            # Check if P conflicts with any pair in selected
            conflict = False
            k = len(selected)
            for i in range(k):
                ax, ay = selected[i]
                d_pa = (px - ax) ** 2 + (py - ay) ** 2
                for j in range(i + 1, k):
                    bx, by = selected[j]
                    d_pb = (px - bx) ** 2 + (py - by) ** 2
                    d_ab = (ax - bx) ** 2 + (ay - by) ** 2
                    if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                selected.append(P)
                selected_set.add(P)
        
        return selected
    
    # Generate all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Try multiple random orderings
    random.seed(42)
    best_size = 0
    
    while time.time() < deadline - 5:
        # Shuffle points
        order = all_points[:]
        random.shuffle(order)
        
        # Try greedy with this order
        result = greedy_construction(order, deadline - 10)
        
        if len(result) > best_size:
            best_size = len(result)
            best_points = result
    
    # Also try a few deterministic orderings
    for dx in range(0, N, 8):
        if time.time() > deadline - 10:
            break
        order = [(x, y) for y in range(N) for x in range(N) if (x + y) % dx == 0 or dx == 0]
        result = greedy_construction(order, deadline - 10)
        if len(result) > best_size:
            best_size = len(result)
            best_points = result
    
    return best_points
# EVOLVE-BLOCK-END
