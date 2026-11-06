import random
import time
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Leave 15s buffer for 300s limit
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(all_points)
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def is_valid_addition(p, current_set, dist_map):
        """
        Checks if point p can be added to current_set without forming an isosceles triangle.
        Returns True if valid, False otherwise.
        """
        # 1. Check if p forms isosceles triangle as apex
        # Distances from p to all other points must be distinct
        dists_from_p = []
        for q in current_set:
            d = sq_dist(p, q)
            dists_from_p.append(d)
            # 2. Check if p forms isosceles triangle with q as apex
            # Distance d(p, q) must not be in dist_map[q]
            if d in dist_map[q]:
                return False
        
        # Check for duplicates in dists_from_p
        if len(dists_from_p) != len(set(dists_from_p)):
            return False
            
        return True

    def add_point(p, current_set, dist_map):
        """Adds point p to the set and updates dist_map."""
        current_set.append(p)
        dist_map[p] = set()
        for i in range(len(current_set) - 1):
            q = current_set[i]
            d = sq_dist(p, q)
            dist_map[p].add(d)
            dist_map[q].add(d)

    def remove_point(idx, current_set, dist_map):
        """Removes point at index idx from the set and updates dist_map."""
        p = current_set[idx]
        # Remove distances from p to all others
        for i in range(len(current_set)):
            if i == idx:
                continue
            q = current_set[i]
            d = sq_dist(p, q)
            dist_map[q].discard(d)
        
        # Remove p from set
        current_set.pop(idx)
        # Clean up dist_map
        del dist_map[p]

    def greedy_construct(seed):
        """Generates an initial valid set using greedy random selection."""
        random.seed(seed)
        shuffled_points = all_points[:]
        random.shuffle(shuffled_points)
        
        current_set = []
        dist_map = {}
        
        for p in shuffled_points:
            if is_valid_addition(p, current_set, dist_map):
                add_point(p, current_set, dist_map)
                
        return current_set, dist_map

    def simulated_annealing(seed, max_iters=5000):
        """
        Uses Simulated Annealing to improve the set size.
        """
        random.seed(seed)
        current_set, dist_map = greedy_construct(seed)
        
        if not current_set:
            return current_set

        current_size = len(current_set)
        best_set = current_set[:]
        best_size = current_size
        
        # Temperature parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        complement = set(all_points) - set(current_set)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            if T < min_T:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a point
                if not complement:
                    continue
                p = random.choice(list(complement))
                if is_valid_addition(p, current_set, dist_map):
                    add_point(p, current_set, dist_map)
                    complement.remove(p)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
            elif move_type < 0.7:
                # Try to remove a point (to escape local optima)
                if current_size <= 1:
                    continue
                idx = random.randint(0, current_size - 1)
                p = current_set[idx]
                remove_point(idx, current_set, dist_map)
                complement.add(p)
                current_size -= 1
            else:
                # Try to swap a point
                if not complement or current_size == 0:
                    continue
                p_out_idx = random.randint(0, current_size - 1)
                p_out = current_set[p_out_idx]
                p_in = random.choice(list(complement))
                
                # Temporarily remove p_out
                remove_point(p_out_idx, current_set, dist_map)
                complement.add(p_out)
                current_size -= 1
                
                # Try to add p_in
                if is_valid_addition(p_in, current_set, dist_map):
                    add_point(p_in, current_set, dist_map)
                    complement.remove(p_in)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                else:
                    # Revert: add p_out back
                    # Note: p_out was removed, so we need to check if it's still valid to add back
                    # It should be, as the set is the same as before the removal, except p_in wasn't added
                    # Wait, the set is exactly the same as before we removed p_out?
                    # Yes, we removed p_out and didn't add p_in.
                    # So we just add p_out back.
                    add_point(p_out, current_set, dist_map)
                    complement.remove(p_out)
                    current_size += 1
            
            T *= cooling_rate
            
        return best_set

    best_overall_set = []
    best_overall_size = 0
    
    seed = 0
    while time.time() < deadline:
        # Run simulated annealing
        result = simulated_annealing(seed)
        
        if len(result) > best_overall_size:
            best_overall_size = len(result)
            best_overall_set = result
            # Optional: print progress for debugging, but keep silent for final
            # print(f"Seed {seed}: Found set of size {best_overall_size}")
            
        seed += 1
        
        # If we have a very large set, we might break early? No, keep searching.
        
    return best_overall_set
# EVOLVE-BLOCK-END
