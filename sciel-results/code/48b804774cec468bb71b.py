import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320  # Safety margin before 330s kill
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_solution = []
    best_size = 0

    def is_valid_addition(point, current_set):
        """
        Check if 'point' can be added to 'current_set' without forming an isosceles triangle.
        Assumes current_set is valid.
        """
        px, py = point
        pts = list(current_set)
        n = len(pts)
        
        if n == 0:
            return True
        
        # 1. Check if point has unique distance to all points in current_set
        dists = {}
        for p in pts:
            d2 = (px - p[0])**2 + (py - p[1])**2
            if d2 in dists:
                return False
            dists[d2] = True
            
        # 2. Check if d(point, A) == d(A, B) for any pair A, B in current_set
        # This covers d(point, A) == d(A, B) and d(point, B) == d(A, B)
        for i in range(n):
            ax, ay = pts[i]
            d_pa = (px - ax)**2 + (py - ay)**2
            for j in range(i + 1, n):
                bx, by = pts[j]
                d_ab = (ax - bx)**2 + (ay - by)**2
                if d_pa == d_ab:
                    return False
                # We also need to check d(point, B) == d(A, B)
                d_pb = (px - bx)**2 + (py - by)**2
                if d_pb == d_ab:
                    return False
        return True

    def greedy_fill(current_set):
        """Greedily add random points."""
        remaining = [p for p in all_points if p not in current_set]
        random.shuffle(remaining)
        for p in remaining:
            if is_valid_addition(p, current_set):
                current_set.add(p)
            if time.time() > deadline:
                break

    def local_search(current_set):
        """Local search with additions and swaps."""
        no_improvement = 0
        max_no_improvement = 200
        
        while no_improvement < max_no_improvement:
            if time.time() > deadline:
                break
            
            improved = False
            candidates = [p for p in all_points if p not in current_set]
            
            # 1. Try to add a random point
            if candidates:
                p = random.choice(candidates)
                if is_valid_addition(p, current_set):
                    current_set.add(p)
                    improved = True
                    no_improvement = 0
                    continue
            
            # 2. Try a swap
            if current_set and candidates:
                # Try a few random swaps
                for _ in range(5):
                    remove_pt = random.choice(list(current_set))
                    add_pt = random.choice(candidates)
                    # Check validity of the new set
                    new_set = current_set - {remove_pt}
                    if is_valid_addition(add_pt, new_set):
                        current_set.remove(remove_pt)
                        current_set.add(add_pt)
                        # Swap accepted. We don't reset no_improvement because size didn't change.
                        # But we break to try adding again.
                        break
                else:
                    # No swap worked
                    no_improvement += 1
            else:
                no_improvement += 1

            if not improved:
                # If we didn't add, we might have swapped. 
                # If we swapped, we loop again to try adding.
                # If we didn't add or swap, no_improvement was incremented.
                pass

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        current_set = set()
        
        # 1. Greedy Fill
        greedy_fill(current_set)
        
        # 2. Local Search
        local_search(current_set)
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = list(current_set)
            
        # If we've used too much time, break
        if time.time() > deadline - 10:
            break
            
    return best_solution
# EVOLVE-BLOCK-END
