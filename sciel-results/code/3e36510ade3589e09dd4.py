import random
import time
import math

N = 64

def construct():
    all_points = [(x, y) for y in range(N) for x in range(N)]
    point_indices = {p: i for i, p in enumerate(all_points)}
    
    # Initialize with a greedy solution to have a good starting point
    # Reuse the logic from the current program but just get one good start
    # Or just start with a random set of size 20
    
    # Let's try to get a decent initial set using a fast greedy
    initial_set = []
    for p in all_points:
        # Check if p can be added
        # For speed in init, we can be less strict or just use a subset
        # Actually, let's just use a random set of size 50 to start SA
        pass
    
    # Better: Start with a small random set and grow it via SA
    random.seed(42)
    current_set = set(random.sample(all_points, 50))
    
    # Calculate initial energy
    # We need a function to get total isosceles count?
    # No, we only need delta. But we need to know if the final set is valid.
    # We will track the best valid set found.
    
    best_valid_set = set()
    best_size = 0
    
    # Check initial set validity
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Check isosceles
                    p1, p2, p3 = S_list[i], S_list[j], S_list[k]
                    d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                    d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                    d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                    if d12 == d23 or d12 == d13 or d23 == d13:
                        return False
        return True

    # This is_valid is O(N^3). Too slow for frequent checks.
    # We will rely on the energy function to guide us to 0 triangles.
    # At the end, we verify.
    
    def count_isosceles_involving(p, S_list):
        count = 0
        n = len(S_list)
        if n < 2:
            return 0
        dists = []
        for a in S_list:
            d = (p[0]-a[0])**2 + (p[1]-a[1])**2
            dists.append(d)
        
        # Case 1: P is apex
        freq = {}
        for d in dists:
            if d in freq:
                count += freq[d]
            else:
                freq[d] = 1
        
        # Case 2 & 3: A or B is apex
        # Optimize: only check if necessary?
        # For small n, just do it.
        for i in range(n):
            for j in range(i+1, n):
                a = S_list[i]
                b = S_list[j]
                d_ab = (a[0]-b[0])**2 + (a[1]-b[1])**2
                if dists[i] == d_ab:
                    count += 1
                if dists[j] == d_ab:
                    count += 1
        return count

    T = 10.0
    T_min = 0.01
    cooling = 0.999
    penalty = 100.0
    
    start_time = time.time()
    time_limit = 290.0 # Leave 10s for cleanup
    
    current_set_list = list(current_set)
    
    # We don't track total energy, just deltas.
    # But we need to know if we are at 0 energy to update best_valid_set.
    # We can periodically check if current_set is valid.
    # Or we can track the number of isosceles triangles exactly.
    # Let's track `current_isosceles_count`.
    
    # Calculate initial count
    current_isosceles_count = 0
    # O(N^3) calculation for init is fine for N=50
    S_list = current_set_list
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = S_list[i], S_list[j], S_list[k]
                d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                if d12 == d23 or d12 == d13 or d23 == d13:
                    current_isosceles_count += 1
                    
    if current_isosceles_count == 0:
        best_valid_set = set(current_set)
        best_size = len(current_set)

    while time.time() - start_time < time_limit:
        # Generate move
        if random.random() < 0.5:
            # Add point
            if len(current_set) < 4096:
                # Pick a random point not in set
                # For efficiency, pick from all_points and check
                # This is O(1) on average if set is sparse
                p = random.choice(all_points)
                if p in current_set:
                    continue
                # Calculate delta
                new_count = count_isosceles_involving(p, current_set_list)
                delta_e = penalty * new_count - 1.0
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_set.add(p)
                    current_set_list.append(p)
                    current_isosceles_count += new_count
                    if current_isosceles_count == 0 and len(current_set) > best_size:
                        best_valid_set = set(current_set)
                        best_size = len(current_set)
        else:
            # Remove point
            if len(current_set) > 0:
                p = random.choice(current_set_list)
                # Calculate delta (removing p)
                # The triangles involving p are removed
                removed_count = count_isosceles_involving(p, [x for x in current_set_list if x != p])
                # Note: count_isosceles_involving expects S not containing p.
                # My function above takes S and p. It counts triangles in S U {p} involving p.
                # So if I pass S without p, it counts triangles formed by adding p to S.
                # These are exactly the triangles that will be removed.
                delta_e = -penalty * removed_count + 1.0
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_set.remove(p)
                    # Remove from list efficiently?
                    # List removal is O(N).
                    try:
                        idx = current_set_list.index(p)
                        current_set_list.pop(idx)
                    except ValueError:
                        pass # Should not happen
                    current_isosceles_count -= removed_count
                    if current_isosceles_count == 0 and len(current_set) > best_size:
                        best_valid_set = set(current_set)
                        best_size = len(current_set)
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if random.random() < 0.01:
                T = 5.0

    return list(best_valid_set)
