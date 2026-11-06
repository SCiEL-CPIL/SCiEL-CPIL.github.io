import random
import time
import math
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 10s buffer for 300s limit
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(all_points)
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def count_violations_for_addition(p, selected, dist_counts):
        """
        Calculates the number of isosceles triangles formed if p is added to selected.
        Returns the count.
        """
        if not selected:
            return 0
        
        # 1. Count violations where p is the apex
        # Distances from p to all q in selected
        dists_from_p = []
        for q in selected:
            d = sq_dist(p, q)
            dists_from_p.append(d)
        
        # Count pairs with same distance
        dist_freq = defaultdict(int)
        for d in dists_from_p:
            dist_freq[d] += 1
        
        apex_violations = 0
        for d, count in dist_freq.items():
            if count > 1:
                apex_violations += count * (count - 1) // 2
                
        # 2. Count violations where q in selected is the apex
        # For each q, check if d(p, q) exists in dist_counts[q]
        # If it does, let k = dist_counts[q][d(p, q)]
        # Then there are k other points r such that d(q, r) = d(q, p)
        # Each such r forms an isosceles triangle (q, p, r) with q as apex
        base_violations = 0
        for q in selected:
            d = sq_dist(p, q)
            if d in dist_counts[q]:
                # dist_counts[q][d] includes the count of points at distance d from q
                # Since p is not yet in selected, dist_counts[q][d] is the count of existing points
                # Each of these existing points r forms a triangle (q, p, r)
                base_violations += dist_counts[q][d]
                
        return apex_violations + base_violations

    def count_violations_for_removal(p, selected, dist_counts):
        """
        Calculates the number of isosceles triangles removed if p is removed from selected.
        Returns the count.
        """
        if len(selected) <= 1:
            return 0
            
        # 1. Count violations where p was the apex
        # These are triangles (p, q1, q2) where d(p, q1) = d(p, q2)
        # The count is sum of C(k, 2) for each distance k in dist_counts[p]
        apex_violations = 0
        for d, count in dist_counts[p].items():
            if count > 1:
                apex_violations += count * (count - 1) // 2
                
        # 2. Count violations where q in selected\{p} was the apex
        # These are triangles (q, p, r) where d(q, p) = d(q, r)
        # For a fixed q, let d = d(q, p).
        # The number of such r is dist_counts[q][d] - 1 (excluding p itself)
        base_violations = 0
        for q in selected:
            if q == p:
                continue
            d = sq_dist(p, q)
            if d in dist_counts[q]:
                # dist_counts[q][d] includes p
                # So other points r are dist_counts[q][d] - 1
                base_violations += dist_counts[q][d] - 1
                
        return apex_violations + base_violations

    def add_point(p, selected, dist_counts):
        """Adds point p to selected and updates dist_counts."""
        selected.add(p)
        dist_counts[p] = defaultdict(int)
        for q in selected:
            if q == p:
                continue
            d = sq_dist(p, q)
            dist_counts[p][d] += 1
            dist_counts[q][d] += 1

    def remove_point(p, selected, dist_counts):
        """Removes point p from selected and updates dist_counts."""
        for q in selected:
            if q == p:
                continue
            d = sq_dist(p, q)
            dist_counts[p][d] -= 1
            dist_counts[q][d] -= 1
            if dist_counts[q][d] == 0:
                del dist_counts[q][d]
        
        selected.remove(p)
        del dist_counts[p]

    def run_sa(seed, time_limit=280.0):
        """
        Runs Simulated Annealing to minimize violations.
        Returns the best valid set found.
        """
        random.seed(seed)
        
        # Initial state: Empty set
        selected = set()
        dist_counts = defaultdict(lambda: defaultdict(int))
        current_violations = 0
        
        best_set = set()
        best_size = 0
        
        # Temperature
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.01
        
        # Precompute all points not in selected for random choice
        # We can just pick a random point from all_points and check if it's in selected
        # To make it faster, maintain a list of available points?
        # Given N=4096, picking from all_points is O(1) average if we use random.choice
        # but checking membership is O(1).
        
        while time.time() < start_time + time_limit:
            if T < min_T:
                # Restart with higher T? Or just break and start new seed
                break
                
            # Generate move: Add, Remove, or Swap
            move_type = random.random()
            
            if move_type < 0.33:
                # Add
                if len(selected) == N * N:
                    continue
                p = random.choice(all_points)
                if p in selected:
                    continue
                
                delta_v = count_violations_for_addition(p, selected, dist_counts)
                if delta_v < 0 or random.random() < math.exp(-delta_v / T):
                    add_point(p, selected, dist_counts)
                    current_violations += delta_v
                    if current_violations == 0 and len(selected) > best_size:
                        best_size = len(selected)
                        best_set = selected.copy()
                        
            elif move_type < 0.66:
                # Remove
                if not selected:
                    continue
                p = random.choice(list(selected))
                
                delta_v = -count_violations_for_removal(p, selected, dist_counts)
                # Note: Removing a point decreases violations.
                # So new_violations = current_violations - violations_removed
                # delta_v = new_violations - current_violations = -violations_removed
                if delta_v < 0 or random.random() < math.exp(-delta_v / T):
                    remove_point(p, selected, dist_counts)
                    current_violations += delta_v
                    if current_violations == 0 and len(selected) > best_size:
                        best_size = len(selected)
                        best_set = selected.copy()
                        
            else:
                # Swap
                if not selected or len(selected) == N * N:
                    continue
                p_out = random.choice(list(selected))
                p_in = random.choice(all_points)
                if p_in in selected:
                    continue
                    
                # Calculate delta for removing p_out and adding p_in
                # This is complex to do exactly in one step without modifying state
                # We can approximate by doing remove then add, but we need to revert if rejected
                # Or calculate exact delta.
                # Let's calculate delta_remove and delta_add separately?
                # No, the interactions matter.
                # Easier: Simulate the changes on copies? Too slow.
                # Let's just do Remove then Add sequentially with acceptance?
                # If we accept remove but reject add, we are in a different state.
                # It's safer to treat Swap as two steps: Remove p_out, then Add p_in.
                # But SA usually treats a move as atomic.
                # Given the complexity, let's just use Add and Remove moves.
                # Swaps are less frequent in simple SA implementations if Add/Remove are balanced.
                # However, swaps are good for maintaining size.
                # Let's implement a simple swap check:
                # 1. Calculate violations if we remove p_out.
                # 2. Calculate violations if we add p_in to (selected - {p_out}).
                # This requires temporary state modification or careful calculation.
                
                # For now, skip swaps to keep implementation robust and fast.
                # Add and Remove should be sufficient to explore the space.
                pass
            
            T *= cooling_rate
            
        return best_set

    best_overall_set = set()
    best_overall_size = 0
    
    seed = 0
    while time.time() < deadline:
        # Run SA for a short duration
        # Adjust time_limit per run to allow multiple restarts
        time_per_run = 5.0
        if time.time() + time_per_run > deadline:
            time_per_run = deadline - time.time()
            if time_per_run <= 0:
                break
                
        result_set = run_sa(seed, time_per_run)
        
        if len(result_set) > best_overall_size:
            best_overall_size = len(result_set)
            best_overall_set = result_set
            
        seed += 1
        
        # If we found a very large set, we might want to refine it
        # But for now, keep searching with new seeds
        
    return list(best_overall_set)
# EVOLVE-BLOCK-END
