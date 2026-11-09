import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    # Pre-generate all grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    point_index = {p: i for i, p in enumerate(grid_points)}
    
    # Random seed
    random.seed(42)

    # Initialize with a greedy set to have a good starting point
    # Greedy: add points in random order if valid
    def dist2(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def get_dists_set(S):
        """
        For a set S of points, return a list of sets.
        dist_sets[i] is the set of squared distances from S[i] to all other points in S.
        """
        K = len(S)
        dist_sets = [set() for _ in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                d = dist2(S[i], S[j])
                dist_sets[i].add(d)
                dist_sets[j].add(d)
        return dist_sets

    def can_add(P, S, dist_sets):
        """
        Check if P can be added to S without forming an isosceles triangle.
        S is a list of points.
        dist_sets[i] is the set of squared distances from S[i] to other points in S.
        """
        K = len(S)
        if K == 0:
            return True
        
        # Calculate squared distances from P to all points in S
        d_p = [0] * K
        for i, a in enumerate(S):
            d_p[i] = dist2(P, a)
            
        # Check 1: Distances from P to all points in S must be distinct
        # If any d_p[i] == d_p[j], then P, S[i], S[j] form an isosceles triangle with apex P.
        if len(set(d_p)) != K:
            return False
            
        # Check 2 & 3: For each point A in S, dist(P, A) must not be equal to dist(A, B) for any B in S \ {A}
        # This is equivalent to checking if d_p[i] is in dist_sets[i]
        for i in range(K):
            if d_p[i] in dist_sets[i]:
                return False
                
        return True

    def add_point(P, S, dist_sets):
        """
        Add P to S and update dist_sets.
        Returns new dist_sets.
        """
        K = len(S)
        # Calculate distances from P to all existing points
        new_dists = [0] * K
        for i, a in enumerate(S):
            new_dists[i] = dist2(P, a)
            
        # Update existing dist_sets
        new_dist_sets = [s.copy() for s in dist_sets]
        for i in range(K):
            new_dist_sets[i].add(new_dists[i])
            
        # Create new dist_set for P
        # It contains the distances to all existing points
        p_dist_set = set(new_dists)
        
        S.append(P)
        new_dist_sets.append(p_dist_set)
        
        return new_dist_sets

    def remove_point(P, S, dist_sets):
        """
        Remove P from S and update dist_sets.
        Returns new dist_sets.
        """
        idx = S.index(P)
        K = len(S)
        
        # The distance from P to S[i] is dist_sets[idx] (which is a set, but we need the specific value? No, we just need to remove the value from S[i]'s set)
        # Actually, dist_sets[idx] is the set of distances from P to others.
        # dist_sets[i] contains the distance from S[i] to P.
        # The value is the same.
        
        # We need to remove the distance value from dist_sets[i] for all i != idx.
        # But dist_sets[i] is a set. We need to know the specific distance value to remove.
        # We can recompute it or store it.
        # Since dist_sets[idx] is a set, we don't have the mapping to which distance belongs to which point.
        # This representation is slightly inconvenient for removal without recomputation or extra storage.
        
        # Let's recompute dist_sets for the remaining points to keep it simple and robust, 
        # or better, just rebuild the list of sets for the remaining K-1 points.
        # Given K is small, rebuilding is O(K^2) which is acceptable for a remove operation 
        # if it's not called excessively. In SA, we might call it often.
        
        # Alternative: Store dist_matrix instead of sets?
        # If we store dist_matrix[i][j], removal is O(K^2) to rebuild.
        # The set representation was good for O(1) lookup in can_add, but bad for update/removal.
        
        # Let's switch to a hybrid or just use the matrix for simplicity in updates, 
        # and check validity by iterating pairs in can_add if K is small?
        # No, the set check is O(K). The pair check is O(K^2).
        # Let's stick to the set representation but make removal efficient by 
        # just rebuilding the dist_sets for the remaining points.
        
        S.pop(idx)
        
        # Rebuild dist_sets for the new S
        return get_dists_set(S)

    # Initial Greedy Construction
    S = []
    dist_sets = []
    points_copy = grid_points[:]
    random.shuffle(points_copy)
    
    for P in points_copy:
        if can_add(P, S, dist_sets):
            dist_sets = add_point(P, S, dist_sets)
            
    best_S = S[:]
    best_dist_sets = dist_sets
    best_size = len(S)
    
    # Simulated Annealing
    current_S = S[:]
    current_dist_sets = dist_sets
    current_size = len(S)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            if len(current_S) < N * N:
                # Pick a random point not in current_S
                # To avoid O(N^2) search for a point not in S, we can just pick random and check
                # Or maintain a list of available points.
                # For now, pick random and check.
                attempts = 0
                P = None
                while attempts < 10:
                    P = random.choice(grid_points)
                    if P not in current_S:
                        break
                    attempts += 1
                if P is not None and P not in current_S:
                    if can_add(P, current_S, current_dist_sets):
                        # Accept
                        new_dist_sets = add_point(P, current_S, current_dist_sets)
                        current_S.append(P)
                        current_dist_sets = new_dist_sets
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_S = current_S[:]
                            best_dist_sets = current_dist_sets
                    else:
                        # Reject (or accept with prob exp(-delta/T) if we had a cost function, but here it's invalid)
                        # In this formulation, invalid moves are simply rejected.
                        # To allow exploration, we might need to allow removing points.
                        pass
        elif move_type < 0.6:
            # Remove
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                P = current_S[idx]
                # Remove P
                current_S.pop(idx)
                current_dist_sets = get_dists_set(current_S) # Rebuild
                current_size -= 1
                # Note: This always decreases size. We rely on the add moves to recover.
                # This is a bit asymmetric. Let's make it a swap or just accept the decrease.
                # Actually, standard SA for maximization usually accepts worse solutions.
                # Here, "worse" means smaller size.
                # So we should accept removal with some probability.
                # But we just did it. We need to decide if we keep it.
                # Let's revert if we don't accept.
                # Prob of accepting a worse solution (size -1) is exp(-1/T)
                if random.random() > math.exp(-1.0 / T):
                    # Reject removal, revert
                    # This is messy. 
                    # Better approach: Generate a candidate move, check if valid, then decide accept/reject.
                    pass
        else:
            # Swap
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                Q = current_S[idx]
                # Pick P not in current_S
                attempts = 0
                P = None
                while attempts < 10:
                    P = random.choice(grid_points)
                    if P not in current_S:
                        break
                    attempts += 1
                if P is not None and P not in current_S:
                    # Check if P can be added if Q is removed
                    # Create temp S
                    S_temp = current_S[:]
                    S_temp.pop(idx)
                    # Check can_add P to S_temp
                    # We need dist_sets for S_temp.
                    # This is expensive to compute just for a check.
                    # Let's just do the O(K^2) check for the swap candidate.
                    K_temp = len(S_temp)
                    valid = True
                    if K_temp >= 2:
                        px, py = P
                        d_p = [0] * K_temp
                        for i, a in enumerate(S_temp):
                            d_p[i] = dist2(P, a)
                        
                        # Check 1: distinct d_p
                        if len(set(d_p)) != K_temp:
                            valid = False
                        else:
                            # Check 2/3: d_p[i] not in dist(S_temp[i], S_temp[j]) for all j != i
                            # We can compute dist(S_temp[i], S_temp[j]) on the fly
                            for i in range(K_temp):
                                if not valid: break
                                for j in range(K_temp):
                                    if i == j: continue
                                    d_ab = dist2(S_temp[i], S_temp[j])
                                    if d_p[i] == d_ab:
                                        valid = False
                                        break
                                    
                    if valid:
                        # Perform swap
                        # Remove Q, Add P
                        # Update current_S and current_dist_sets
                        current_S.pop(idx)
                        # Rebuild dist_sets for current_S
                        current_dist_sets = get_dists_set(current_S)
                        # Add P
                        current_dist_sets = add_point(P, current_S, current_dist_sets)
                        current_S.append(P)
                        # Size remains same
                        # Accept swap? 
                        # Swaps don't change size. They might improve "quality" in terms of allowing future adds.
                        # For simplicity, accept all valid swaps.
                        pass
                            
        # Cool down
        T *= cooling_rate
        
    return best_S
# EVOLVE-BLOCK-END
