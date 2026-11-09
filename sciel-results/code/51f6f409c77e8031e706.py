import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # Pre-generate all grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # Helper to compute squared distance
    def dist2(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    # Fast can_add using precomputed pair distances
    # S: list of points
    # pair_dists: list of lists, pair_dists[i][j] is dist2(S[i], S[j])
    def can_add_fast(p, S, pair_dists):
        K = len(S)
        if K < 2:
            return True
        
        px, py = p
        # Compute distances from p to all points in S
        d_p = [0] * K
        for i, a in enumerate(S):
            d_p[i] = (px - a[0])**2 + (py - a[1])**2
        
        # Check all pairs in S
        for i in range(K):
            for j in range(i + 1, K):
                dab2 = pair_dists[i][j]
                # Isosceles conditions:
                # 1. Apex at p: d(p, S[i]) == d(p, S[j])
                if d_p[i] == d_p[j]:
                    return False
                # 2. Apex at S[i]: d(p, S[i]) == d(S[i], S[j])
                if d_p[i] == dab2:
                    return False
                # 3. Apex at S[j]: d(p, S[j]) == d(S[i], S[j])
                if d_p[j] == dab2:
                    return False
        return True

    def add_point(S, pair_dists, P):
        K = len(S)
        # Compute distances from P to all points in S
        new_dists = [0] * K
        for i, a in enumerate(S):
            new_dists[i] = (P[0] - a[0])**2 + (P[1] - a[1])**2
        
        # Create new pair_dists matrix
        # Current size K x K
        # New size (K+1) x (K+1)
        new_pair_dists = [row[:] for row in pair_dists]
        
        # Append new column to existing rows
        for i in range(K):
            new_pair_dists[i].append(new_dists[i])
        
        # Append new row
        new_row = new_dists[:] + [0]
        new_pair_dists.append(new_row)
        
        S.append(P)
        return new_pair_dists

    def remove_point(S, pair_dists, P):
        # Find index of P
        idx = S.index(P)
        K = len(S)
        
        # Remove from S
        S.pop(idx)
        
        # Rebuild pair_dists? Or remove row/idx and col/idx
        # Removing row/col from a list of lists is O(K^2) due to list shifting, 
        # but K is small.
        new_pair_dists = []
        for i in range(K):
            if i == idx:
                continue
            new_row = []
            for j in range(K):
                if j == idx:
                    continue
                new_row.append(pair_dists[i][j])
            new_pair_dists.append(new_row)
            
        return new_pair_dists

    # Random seed for reproducibility in local tests, but evaluator might vary.
    # We will use a fixed seed to ensure consistent behavior if the evaluator
    # expects deterministic output for a given input, but the task implies
    # stochastic search. However, using a fixed seed ensures we don't get
    # unlucky with a bad random sequence.
    random.seed(42)

    while time.time() < deadline:
        # Create a shuffled list of points for this restart
        points = grid_points[:]
        random.shuffle(points)
        
        S = []
        pair_dists = []
        
        # Greedy Construction
        for P in points:
            if can_add_fast(P, S, pair_dists):
                pair_dists = add_point(S, pair_dists, P)
        
        # Local Search Phase
        # We will try to add points that were rejected, or swap.
        # To avoid O(N^2) checks in local search, we can just iterate through
        # a random subset or the whole list again.
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > 20:
                break # Give up on this restart
            
            # Try to add a point from the original shuffled list
            # We can just iterate through the whole list again
            # But to save time, we can pick random candidates
            # Let's try a few random candidates first
            candidates_tried = 0
            while candidates_tried < 50:
                P = random.choice(grid_points)
                if P in S:
                    candidates_tried += 1
                    continue
                
                if can_add_fast(P, S, pair_dists):
                    pair_dists = add_point(S, pair_dists, P)
                    improved = True
                    no_improve_count = 0
                    break
                candidates_tried += 1
            
            if not improved:
                # Try a swap
                # Pick random Q from S, random P from grid
                if len(S) > 0:
                    Q = random.choice(S)
                    P = random.choice(grid_points)
                    if P == Q or P in S:
                        continue
                    
                    # Check if P can be added if Q is removed
                    S_temp = S[:]
                    pair_dists_temp = pair_dists
                    # We need to remove Q from S_temp and update pair_dists_temp
                    # For simplicity and speed, since K is small, we can just
                    # create a new list and check can_add on S_temp without
                    # full pair_dists update if we implement a slower can_add
                    # for this specific check?
                    # Or we can use the fast method by temporarily removing.
                    
                    # Let's use the fast method by constructing the temp state
                    # This is expensive.
                    # Alternative: Just use the O(K^2) check for the swap candidate.
                    # Since we only do this when adds fail, it's infrequent.
                    
                    S_temp_list = [x for x in S if x != Q]
                    
                    # Check if P can be added to S_temp_list
                    # We can use a simple O(K^2) check here
                    K_temp = len(S_temp_list)
                    valid = True
                    if K_temp >= 2:
                        px, py = P
                        d_p = [0] * K_temp
                        for i, a in enumerate(S_temp_list):
                            d_p[i] = (px - a[0])**2 + (py - a[1])**2
                        
                        for i in range(K_temp):
                            for j in range(i + 1, K_temp):
                                a = S_temp_list[i]
                                b = S_temp_list[j]
                                dab2 = (a[0] - b[0])**2 + (a[1] - b[1])**2
                                if d_p[i] == d_p[j] or d_p[i] == dab2 or d_p[j] == dab2:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        # Perform swap
                        # Remove Q, Add P
                        pair_dists = remove_point(S, pair_dists, Q)
                        pair_dists = add_point(S, pair_dists, P)
                        improved = True
                        no_improve_count = 0

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    return best_set
# EVOLVE-BLOCK-END
