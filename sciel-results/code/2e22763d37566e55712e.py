import numpy as np
import time
import itertools
import random

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)

# Precompute squared distances
coords = np.array(POINTS, dtype=np.int32)
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

def construct():
    start_time = time.time()
    deadline = 280  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # Helper: Check if a set (list of indices) is valid (no isosceles)
    def is_valid_set(S_list):
        if len(S_list) <= 2:
            return True
        n = len(S_list)
        # D_S is n x n
        D_S = D[np.ix_(S_list, S_list)]
        # For each pair (i, j) with i < j, check if there exists k != i, j such that
        # dist(i, j) == dist(i, k) or dist(i, j) == dist(j, k) or dist(i, k) == dist(j, k)
        # Actually, the condition is: no three points form an isosceles triangle.
        # This means for any distinct i, j, k in S:
        # d(i,j) != d(i,k)
        # d(i,j) != d(j,k)
        # d(i,k) != d(j,k)
        # Equivalently, for each i, the distances from i to all other points in S must be unique.
        # Wait, is that sufficient?
        # If distances from i are unique, then d(i,j) != d(i,k) for all j!=k.
        # Does this prevent d(i,j) == d(j,k)?
        # No. Example: i=(0,0), j=(1,0), k=(0,1).
        # d(i,j)=1, d(i,k)=1. Distances from i are not unique (1,1). So this is caught.
        # Example: i=(0,0), j=(2,0), k=(1,1).
        # d(i,j)=4, d(i,k)=2. Unique from i.
        # d(j,i)=4, d(j,k)=2. Unique from j.
        # d(k,i)=2, d(k,j)=2. NOT unique from k.
        # So checking that distances from every point to others are unique is sufficient and necessary.
        
        for i in range(n):
            d_i = D_S[i]
            # Exclude self (d_i[i] = 0)
            d_i_others = np.delete(d_i, i)
            if len(np.unique(d_i_others)) < len(d_i_others):
                return False
        return True

    # Helper: Check if a set is maximal (no point can be added)
    def is_maximal(S_list):
        S_set = set(S_list)
        for p in range(N_POINTS):
            if p in S_set:
                continue
            # Try adding p
            new_S = S_list + [p]
            if is_valid_set(new_S):
                return False
        return True

    # Helper: Get all "bad" points for a set S (points that can be added)
    def get_addable_points(S_list):
        S_set = set(S_list)
        addable = []
        for p in range(N_POINTS):
            if p in S_set:
                continue
            new_S = S_list + [p]
            if is_valid_set(new_S):
                addable.append(p)
        return addable

    # Strategy 1: Randomized Local Search (Walk-SAT style)
    # We want to find a small maximal set.
    # Start with a random valid set (maybe not maximal).
    # Or start with a maximal set and try to shrink it?
    # Shrinking a maximal set breaks maximality.
    # We need to move in the space of sets.
    
    # Let's try a different approach:
    # 1. Generate random valid sets.
    # 2. Check if maximal.
    # 3. If not maximal, try to "fix" it by adding points? But we want to MINIMIZE size.
    # So we want maximal sets.
    #
    # Alternative:
    # Start with a large set (e.g., all points is not valid).
    # Start with a random subset, make it valid by removing points.
    # Then check maximality.
    # If not maximal, we have "free" points we can add. Adding them increases size, which is bad.
    # But we are stuck if we can't remove points without breaking validity.
    #
    # Let's try Simulated Annealing on the set of points.
    # State: Boolean vector x of length N_POINTS.
    # Constraints:
    # 1. No isosceles: For all i,j,k, not (x_i & x_j & x_k & (d_ij == d_ik or d_ij == d_jk or d_ik == d_jk))
    #    This is a 3-SAT like clause.
    # 2. Maximal: For all p, if x_p is False, then adding p violates constraint 1.
    #    This is harder to express as a simple penalty.
    
    # Let's stick to the constructive approach but improve the search.
    # The current program uses random permutation greedy.
    # Let's try:
    # 1. Greedy construction with random order.
    # 2. Post-processing: Try to remove points from the constructed maximal set.
    #    If we remove a point, the set is no longer maximal (that point becomes addable).
    #    To restore maximality, we might need to add other points?
    #    Actually, if we remove a point p from a maximal set S, the new set S' = S \ {p} is valid.
    #    Is it maximal? No, because p can be added back.
    #    So S' is NOT maximal.
    #    Therefore, we cannot simply remove points from a maximal set and expect the result to be maximal.
    #
    # However, we can try to "swap" points.
    # Remove p, add q.
    # If S \ {p} U {q} is valid and maximal, and size is same, we might find a different maximal set.
    # This doesn't reduce size.
    #
    # To reduce size, we need to find a maximal set that is a subset of the current one?
    # No, maximal sets are minimal with respect to inclusion? No, maximal means you can't add more.
    # A smaller maximal set is a subset of a larger maximal set? Not necessarily.
    #
    # Let's look at the structure.
    # If S is maximal, and S' is a valid subset of S, is S' maximal?
    # If S' is not maximal, there is p not in S' such that S' U {p} is valid.
    # Since S' is subset of S, p is either in S \ S' or not in S.
    # If p in S \ S', then S' U {p} is a subset of S. Since S is valid, S' U {p} is valid.
    # So if S \ S' is non-empty, S' is not maximal (we can add back any point from S \ S').
    # Thus, a valid subset of a maximal set is maximal IF AND ONLY IF it IS the maximal set itself.
    # This means we cannot find a smaller maximal set by removing points from a larger one.
    
    # So we must search the space of maximal sets directly.
    # Or find valid sets and then "complete" them to maximal sets, hoping the completion is small?
    # No, completion adds points, increasing size.
    
    # The only way to get a smaller maximal set is to find a different structure.
    
    # Let's try a different heuristic for the greedy step.
    # Instead of random order, use a score based on "constraint tightness".
    # Points that are "far" from others might be safer to add?
    # Or points that create few isosceles triangles with existing points?
    
    # Let's implement a greedy algorithm with backtracking (limited depth).
    # Or simply many random restarts with a smarter greedy.
    
    # Let's try to optimize the current random restart approach by making the check faster and allowing more restarts.
    # The current check in the loop is O(|S|^2) or so.
    
    # Let's try to use the "distance uniqueness" property more efficiently.
    
    # Revised Plan:
    # 1. Precompute for each point p, the list of other points q such that dist(p,q) is unique to p? No.
    # 2. Use a faster validity check.
    
    # Let's try a different search:
    # Start with an empty set.
    # Add points one by one.
    # When adding a point, check validity.
    # If we get stuck (no point can be added), we have a maximal set.
    # This is exactly the greedy construction.
    # The order matters.
    
    # Let's try to use a genetic algorithm on the permutation of points.
    # Or just increase the number of random restarts by making the inner loop faster.
    
    # Optimization:
    # Instead of checking all pairs in S for each new point, we can maintain a data structure.
    # But for N=16, |S| is likely small (< 20).
    
    # Let's try to use the fact that if d(p, s1) == d(p, s2) for s1, s2 in S, we can't add p.
    # Also if d(s1, s2) == d(p, s1) or d(s1, s2) == d(p, s2).
    
    # Let's run many restarts.
    
    def greedy_with_order(order):
        S = []
        S_set = set()
        for p_idx in order:
            if len(S) == 0:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
            if len(S) == 1:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
            
            # Check if p can be added to S
            valid = True
            # 1. Distances from p to S are unique
            d_p_s = D[p_idx, S]
            if len(np.unique(d_p_s)) < len(d_p_s):
                valid = False
                # Note: 0 is not in d_p_s because p is not in S
            else:
                # 2. No isosceles with existing points
                # For each pair (s1, s2) in S, check if d(s1, s2) is in {d(p, s1), d(p, s2)}
                # Actually, the condition is:
                # d(p, s1) != d(p, s2) (checked above)
                # d(s1, s2) != d(p, s1)
                # d(s1, s2) != d(p, s2)
                # This is equivalent to: d(s1, s2) not in {d(p, s1), d(p, s2)}
                # Which means d(s1, s2) != d(p, s1) AND d(s1, s2) != d(p, s2)
                
                # We can check this by:
                # For each s1 in S:
                #   d_ps1 = d(p, s1)
                #   For each s2 in S, s2 != s1:
                #     if d(s1, s2) == d_ps1: fail
                #     if d(s1, s2) == d(p, s2): fail
                
                # Vectorized check:
                # D_S = D[np.ix_(S, S)]
                # d_p_s is vector of distances from p to S.
                # We need to check if any D_S[i, j] == d_p_s[i] for i != j?
                # No, the condition is for the triangle (p, s_i, s_j).
                # Sides: d(p, s_i), d(p, s_j), d(s_i, s_j).
                # Isosceles if any two are equal.
                # d(p, s_i) == d(p, s_j) -> checked by unique(d_p_s)
                # d(p, s_i) == d(s_i, s_j)
                # d(p, s_j) == d(s_i, s_j)
                
                # So we need to check if d_p_s[i] == D_S[i, j] for any i != j.
                # And if d_p_s[j] == D_S[i, j] for any i != j.
                
                D_S = D[np.ix_(S, S)]
                # Create mask for i != j
                # We want to check D_S[i, j] == d_p_s[i]
                # This is element-wise comparison of D_S and d_p_s[:, None]
                # But we must exclude i == j because D_S[i, i] = 0 and d_p_s[i] > 0
                # So D_S[i, i] != d_p_s[i] is always true.
                
                # Check 1: D_S[i, j] == d_p_s[i]
                # This means d(s_i, s_j) == d(p, s_i)
                if np.any(D_S == d_p_s[:, None]):
                    # We need to ensure we don't count the diagonal?
                    # D_S[i, i] = 0. d_p_s[i] > 0. So 0 != d_p_s[i].
                    # So the diagonal is safe.
                    valid = False
                
                if valid:
                    # Check 2: D_S[i, j] == d_p_s[j]
                    # This means d(s_i, s_j) == d(p, s_j)
                    if np.any(D_S == d_p_s[None, :]):
                        valid = False
            
            if valid:
                S.append(p_idx)
                S_set.add(p_idx)
        
        return S

    # Main loop
    while time.time() - start_time < deadline:
        perm = np.random.permutation(N_POINTS)
        S = greedy_with_order(perm)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
            
            # Optional: Local search from this best set
            # Try to swap points to see if we can find a smaller one?
            # As discussed, subsets of maximal sets are not maximal.
            # So we can't just remove points.
            
    if best_set is None:
        # Fallback
        best_set = [POINTS[0]]
        
    return best_set
