import time
import random
import numpy as np

N = 16
NUM_POINTS = N * N

# Precompute points
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute squared distances
dist2 = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# Precompute conflict pairs
# conflict_pairs[p] is a list of (i, j) such that (p, i, j) is an isosceles triangle
# We assume i < j to avoid duplicates
conflict_pairs = [[] for _ in range(NUM_POINTS)]

for p in range(NUM_POINTS):
    # For each pair (i, j) with i < j, check if (p, i, j) is isosceles
    # This is O(N^3) which is 16^3 = 4096 * 256/2 ~ 500k operations. Fine.
    for i in range(NUM_POINTS):
        if i == p:
            continue
        for j in range(i + 1, NUM_POINTS):
            if j == p:
                continue
            d_pi = dist2[p, i]
            d_pj = dist2[p, j]
            d_ij = dist2[i, j]
            
            # Check for isosceles
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                # Ensure we store unique pairs (min, max)
                if i < j:
                    conflict_pairs[p].append((i, j))
                else:
                    conflict_pairs[p].append((j, i))

def is_maximal(in_S):
    """Check if the set represented by boolean array in_S is maximal."""
    for p in range(NUM_POINTS):
        if in_S[p]:
            continue
        # Check if p is blocked
        # p is blocked if there exists (i, j) in conflict_pairs[p] such that in_S[i] and in_S[j]
        blocked = False
        for i, j in conflict_pairs[p]:
            if in_S[i] and in_S[j]:
                blocked = True
                break
        if not blocked:
            return False
    return True

def can_add(in_S, p):
    """Check if point p can be added to set in_S without creating an isosceles triangle."""
    # p can be added if there are no two points i, j in S such that (p, i, j) is isosceles.
    # This is equivalent to: in_S does not contain any pair from conflict_pairs[p].
    for i, j in conflict_pairs[p]:
        if in_S[i] and in_S[j]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Initial greedy solution
    # Try multiple random orderings to get diverse starting points
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random.seed(1000 + restart)
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        in_S = np.zeros(NUM_POINTS, dtype=bool)
        current_set = []
        
        for p in order:
            if can_add(in_S, p):
                in_S[p] = True
                current_set.append(p)
        
        # Check if maximal
        if not is_maximal(in_S):
            # Greedily add to make maximal
            for p in order:
                if not in_S[p] and can_add(in_S, p):
                    in_S[p] = True
                    current_set.append(p)
            # It should be maximal now
            
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        # Local Search: Minimize
        current_in_S = in_S.copy()
        current_list = list(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to randomize removal order
            random.shuffle(current_list)
            
            for p in current_list:
                if time.time() > deadline:
                    break
                # Try removing p
                if current_in_S[p]:
                    current_in_S[p] = False
                    current_list.remove(p)
                    
                    if is_maximal(current_in_S):
                        # Accept removal
                        improved = True
                        if len(current_list) < best_size:
                            best_size = len(current_list)
                            best_set = list(current_list)
                        # Break to restart the loop with the new smaller set
                        break
                    else:
                        # Restore p
                        current_in_S[p] = True
                        current_list.append(p)
        
        # Perturbation if stuck or to explore further
        if time.time() < deadline - 5:
            # Remove a few random points
            num_remove = random.randint(2, 5)
            to_remove = random.sample(current_list, min(num_remove, len(current_list)))
            for p in to_remove:
                if current_in_S[p]:
                    current_in_S[p] = False
                    if p in current_list:
                        current_list.remove(p)
            
            # Greedily add to restore maximality
            # We need to add points until maximal.
            # Use a random order for adding
            candidates = [p for p in range(NUM_POINTS) if not current_in_S[p]]
            random.shuffle(candidates)
            
            for p in candidates:
                if is_maximal(current_in_S):
                    break
                if can_add(current_in_S, p):
                    current_in_S[p] = True
                    current_list.append(p)
            
            # Final check for maximality (in case greedy didn't cover all)
            if not is_maximal(current_in_S):
                # Force maximal
                for p in range(NUM_POINTS):
                    if not current_in_S[p]:
                        # Check if adding p keeps it valid? 
                        # Actually, if it's not maximal, some q is not blocked.
                        # We can just add q? No, q might conflict.
                        # The greedy loop above should have handled it if can_add was used.
                        # But is_maximal checks if ALL p not in S are blocked.
                        # If we add points until no more can be added (maximal independent set), it is maximal.
                        # The loop "while not maximal: add random valid point" ensures this.
                        pass
            
            # Minimize the perturbed set
            improved = True
            while improved and time.time() < deadline:
                improved = False
                random.shuffle(current_list)
                for p in current_list:
                    if time.time() > deadline:
                        break
                    if current_in_S[p]:
                        current_in_S[p] = False
                        if p in current_list:
                            current_list.remove(p)
                        
                        if is_maximal(current_in_S):
                            improved = True
                            if len(current_list) < best_size:
                                best_size = len(current_list)
                                best_set = list(current_list)
                            break
                        else:
                            current_in_S[p] = True
                            current_list.append(p)

    if best_set is None:
        # Fallback
        in_S = np.zeros(NUM_POINTS, dtype=bool)
        current_set = []
        for p in range(NUM_POINTS):
            if can_add(in_S, p):
                in_S[p] = True
                current_set.append(p)
        return [POINTS[i] for i in current_set]
        
    return [POINTS[i] for i in best_set]
