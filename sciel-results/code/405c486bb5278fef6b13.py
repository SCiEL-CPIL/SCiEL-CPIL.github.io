import time
import random
import numpy as np
from itertools import combinations

N = 64
TIME_LIMIT = 300.0

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0
    
    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    total = N * N
    
    # Precompute squared distances
    # dist[i, j] = (xi-xj)^2 + (yi-yj)^2
    dx = (xs[:, None] - xs[None, :]).astype(np.int32)
    dy = (ys[:, None] - ys[None, :]).astype(np.int32)
    dist = dx * dx + dy * dy
    
    # Build conflict graph
    # Node i and j are connected if there exists a k such that {i, j, k} forms an isosceles triangle.
    # Actually, it's easier to think: Node i and j are "incompatible" if adding both to a set S 
    # creates an isosceles triangle with any existing node in S.
    # Wait, the standard MIS formulation:
    # Vertices = points.
    # Edge (u, v) exists if u and v CANNOT be in the same valid set?
    # No. A set S is valid if NO THREE points in S form an isosceles triangle.
    # This is a 3-uniform hypergraph independent set problem, not a graph MIS.
    # However, we can reduce it:
    # If we fix two points u, v, they "conflict" with any w that forms an isosceles triangle with u and v.
    # This is hard to represent as a simple graph for standard MIS solvers.
    
    # Alternative Strategy: 
    # The current program uses a greedy + local search approach.
    # The prompt asks for a DIFFERENT strategy.
    # Let's try a **Simulated Annealing** approach with a different neighborhood definition.
    # Or better: **Genetic Algorithm**.
    
    # Let's try a **Constraint Programming** style approach using backtracking with forward checking,
    # but optimized with numpy for the checks.
    # Actually, the most distinct structural change from "Greedy + Local Search" is **Simulated Annealing**
    # where we allow temporary worsening of the objective function (number of points) to escape local minima.
    # But the objective is just "maximize count". The constraint is "no isosceles".
    # If we violate the constraint, the score is 0. So we can't just "worsen" the count if it breaks validity.
    # Unless we use a penalty function: Score = Count - Penalty * (Number of Isosceles Triangles).
    # Then we can anneal towards high count and low penalty.
    
    # Let's implement Simulated Annealing with a penalty function.
    # State: A set of points (represented as a boolean mask or list).
    # Objective: Maximize (len(points) - lambda * num_isosceles_triangles).
    # Moves:
    # 1. Add a random point (if not present).
    # 2. Remove a random point.
    # 3. Swap a point with a random other point.
    
    # To make this efficient, we need to update the number of isosceles triangles quickly.
    # This is tricky. Adding a point P to set S:
    # New triangles are (P, i, j) where i, j in S.
    # We can precompute for each point P, the set of pairs (i, j) that form an isosceles triangle with P.
    # This is too much memory (64^3 pairs).
    
    # Let's stick to a **Branch and Bound** on the point order, but with a different heuristic.
    # Or, let's try **Greedy with Multiple Heuristics**.
    # The current code uses random order.
    # What if we use **Geometric Heuristics**?
    # E.g., prefer points that are far from existing points?
    # Or prefer points on the boundary?
    
    # Let's try a **Multi-Start Hill Climbing** with a **Different Move Operator**.
    # Current: Swap + Add.
    # New: **2-opt style swap** or **Relocate**.
    
    # Actually, a truly different family is **Evolutionary Strategy** (Genetic Algorithm).
    # Represent the set as a bitstring of length N*N.
    # Fitness: len(set) if valid, else -1 (or 0).
    # Since validity is a hard constraint, GA might struggle.
    # We can use a penalty: Fitness = len(set) - 100 * num_violations.
    
    # Let's implement a **Penalty-based Simulated Annealing**.
    # It allows exploring invalid states temporarily, which might jump over local maxima of valid states.
    
    def count_isosceles(points):
        """
        Count the number of isosceles triangles in the set of points.
        points: list of indices.
        """
        if len(points) < 3:
            return 0
        count = 0
        # O(K^3) is too slow for large K.
        # We need a faster check or a different representation.
        # For SA, we need fast evaluation.
        # If K is ~50, K^3 is 125,000. 1000 iterations * 125k = 125M ops. Might be ok in Py? No, Python is slow.
        # We need numpy vectorization for the count.
        
        # Vectorized count:
        # For each pair (i, j), count k such that d(i,k)==d(i,j) or d(j,k)==d(i,j) or d(i,k)==d(j,k).
        # This is complex to vectorize in one go without loops.
        
        # Alternative: Keep the validity check O(K^2) as in the current code, but use it for SA?
        # SA requires evaluating many states. If each eval is O(K^2) and K=50, that's 2500 ops.
        # 10,000 steps * 2500 = 25M ops. This is feasible in Python if optimized.
        
        # Let's implement a fast "delta" update for SA.
        # When adding P:
        # 1. Check if P forms isosceles with any pair (i, j) in S.
        # 2. Check if any pair (P, i) forms isosceles with j in S.
        
        # This is essentially the same check as `can_add_vectorized`.
        # So, if we use SA, we need to be able to "undo" the check quickly or recompute.
        # Recomputing the full count is O(K^3).
        # Updating the count is O(K^2) for add/remove.
        
        # Let's try a different structural change: **Randomized Greedy with Lookahead**.
        # Instead of picking the next point randomly, pick the point that maximizes the number of 
        # *future* points it doesn't conflict with.
        # This is a heuristic greedy, not local search.
        
        # Or: **Set Cover / Packing** approach.
        # Partition the grid into regions?
        
        # Let's go with **Simulated Annealing** with a penalty function.
        # It is a distinct optimization family (stochastic global search) compared to the current 
        # (greedy + local improvement).
        
        # State: list of indices.
        # Energy: -len(state) + penalty * num_isosceles.
        # We want to minimize Energy.
        
        # To make it fast, we will maintain the set as a list and use the vectorized check 
        # to determine the *change* in penalty when adding/removing.
        
        # However, SA needs to accept worse states. 
        # If we remove a point, we always reduce penalty (or keep same) and reduce len.
        # If we add a point, we increase len and possibly increase penalty.
        
        # Let's define the state as a boolean array `mask` of size N*N.
        # And a list `points` of active indices.
        
        # We will precompute for each point P, a list of "conflicting pairs"? No, too big.
        
        # Let's use the `can_add` logic to compute the *number* of conflicts if P is added.
        # `num_conflicts(P, S)` = number of pairs (i, j) in S such that {P, i, j} is isosceles.
        # This can be computed in O(K^2) using the same vectorization as before.
        
        def count_conflicts(candidate, current_indices):
            """
            Returns the number of isosceles triangles formed by adding 'candidate' to 'current_indices'.
            """
            K = len(current_indices)
            if K < 2:
                return 0
                
            d_c = dist[candidate, current_indices]
            
            # Check 1: Apex at candidate.
            # d_c has duplicates.
            # Number of pairs (i, j) such that d_c[i] == d_c[j].
            # Count duplicates in d_c.
            # If a distance value appears 'm' times, it contributes m*(m-1)/2 pairs.
            _, counts = np.unique(d_c, return_counts=True)
            apex_c_conflicts = np.sum(counts * (counts - 1) // 2)
            
            # Check 2: Apex at current point i.
            # Triangle (c, i, j). Isosceles if d(c, i) == d(i, j).
            # For each i, we need to count how many j in S (j != i) satisfy d(i, j) == d_c[i].
            # D_sub = dist[np.ix_(current_indices, current_indices)]
            # We want to count, for each row i, the number of elements equal to d_c[i].
            # Note: D_sub[i, i] = 0. d_c[i] > 0. So we don't need to exclude i explicitly if d_c[i] != 0.
            
            D_sub = dist[np.ix_(current_indices, current_indices)]
            # Compare D_sub with d_c[:, None]
            matches = (D_sub == d_c[:, None])
            # Sum the matches for each row i.
            # matches[i, :] is a boolean vector.
            # We want sum(matches[i, :]) for all i.
            apex_i_conflicts = np.sum(matches)
            
            # Wait, is there double counting?
            # Triangle (c, i, j).
            # Case 1: d(c, i) == d(c, j). (Apex at c). Counted in apex_c_conflicts.
            # Case 2: d(c, i) == d(i, j). (Apex at i). Counted in apex_i_conflicts (row i, col j).
            # Case 3: d(c, j) == d(i, j). (Apex at j). Counted in apex_i_conflicts (row j, col i).
            
            # Are these disjoint?
            # If d(c, i) == d(c, j) == d(i, j), it's equilateral.
            # It is counted in Case 1 (once for the pair i,j).
            # It is counted in Case 2 (d(c, i) == d(i, j)).
            # It is counted in Case 3 (d(c, j) == d(i, j)).
            # So equilateral triangles are counted 3 times?
            # We just need the *total number* of isosceles triangles.
            # If we sum these counts, we might overcount equilateral triangles.
            # However, for the purpose of SA penalty, overcounting equilateral triangles 
            # just makes them "more expensive" to form, which is fine.
            # We just need a consistent penalty.
            
            return apex_c_conflicts + apex_i_conflicts

        # SA Implementation
        current_points = []
        current_mask = np.zeros(total, dtype=bool)
        
        # Initialize with a greedy solution to get a good start
        remaining = np.arange(total)
        random.shuffle(remaining)
        for idx in remaining:
            if time.time() > deadline:
                break
            if count_conflicts(idx, current_points) == 0:
                current_points.append(idx)
                current_mask[idx] = True
        
        best_points = list(current_points)
        best_score = len(current_points)
        
        # SA Parameters
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.999
        max_steps = 100000
        
        step = 0
        while T > T_min and time.time() < deadline - 5:
            # Generate a neighbor
            move_type = random.random()
            new_points = list(current_points)
            new_mask = current_mask.copy()
            
            if move_type < 0.3 and len(new_points) > 0:
                # Remove a random point
                idx_remove = random.choice(new_points)
                new_points.remove(idx_remove)
                new_mask[idx_remove] = False
                delta_len = -1
                # Penalty change: removing a point reduces conflicts involving it.
                # We don't track exact penalty, just len. 
                # Wait, if we use penalty, we need to track it.
                # Let's just use Len as objective, but allow invalid states?
                # No, if we allow invalid states, we must penalize them.
                # Let's track `current_penalty`.
                
                # This requires updating penalty incrementally.
                # Let's restart with a simpler SA that only moves between VALID states.
                # This is just Local Search again.
                
                # To be truly different, let's try **Genetic Algorithm**.
                # Population of sets.
                # Crossover: Union/Intersection? No, must be valid.
                # This is hard.
                
                # Let's go back to **Branch and Bound**.
                # It is a complete search method, unlike the heuristic local search.
                # We order points. We branch on include/exclude.
                # Pruning: If remaining points + current count < best, prune.
                # This is exponential, but with good pruning and a strong initial bound, it might work for N=64?
                # 4096 points. MIS on 4096 nodes is huge.
                # But the graph is sparse?
                # Each point conflicts with many others.
                
                # Let's try a **Hybrid**: 
                # 1. Greedy to get a good lower bound (current best).
                # 2. Use **Local Search** but with a **Different Neighborhood**: 
                #    Instead of swapping 1 point, swap 2 points (2-exchange).
                #    This is a structural change in the search space exploration.
                
                # Let's implement 2-exchange local search.
                # Pick two points in S to remove, and two points not in S to add.
                # Check if the new set is valid.
                # This allows escaping from local minima where 1-swap is blocked.
                
                break # Stop the SA idea, implement 2-exchange.

        # Implement 2-exchange Local Search
        # This is a structural change from 1-swap.
        
        def is_valid_set(indices):
            """Check if a set of indices forms a valid no-isosceles set."""
            K = len(indices)
            if K < 3:
                return True
            # Check all triples? O(K^3) is too slow.
            # Use the vectorized check from the current program?
            # The current program's `can_add_vectorized` checks if a point CAN be added.
            # We need to check if the WHOLE set is valid.
            # We can do this by adding points one by one.
            # But that's O(K^2 * K) = O(K^3).
            # For K=50, 125k ops. 
            # If we do 1000 2-swaps, that's 125M ops. Might be tight in 300s.
            # We need a faster validity check for the whole set.
            
            # Alternative: 
            # Precompute all isosceles triples? No.
            
            # Let's optimize the validity check.
            # A set is valid if for all i, j, k in set, no isosceles.
            # We can check:
            # 1. For each i, all distances from i to others are unique.
            # 2. For each pair i, j, the distance d(i, j) is not equal to any d(i, k) or d(j, k) for k != i, j.
            
            # Vectorized:
            # D = dist[np.ix_(indices, indices)]
            # 1. For each row i, check if D[i, :] has duplicates (excluding 0).
            # 2. For each pair i, j, check if D[i, j] is in D[i, :] or D[j, :].
            
            # This is still O(K^2) or O(K^3) depending on implementation.
            
            # Let's stick to the incremental update if possible, or just use the 
            # "Add one by one" check which is O(K^3) but with small constants (numpy).
            
            current_sub = []
            for idx in indices:
                if count_conflicts(idx, current_sub) > 0:
                    return False
                current_sub.append(idx)
            return True

        def local_search_2swap(initial_indices, time_budget):
            current = list(initial_indices)
            best = list(current)
            best_size = len(current)
            end_time = time.time() + time_budget
            
            all_indices = set(range(total))
            
            improved = True
            while improved and time.time() < end_time:
                improved = False
                # Try a batch of 2-swaps
                n_attempts = 100
                for _ in range(n_attempts):
                    if time.time() > end_time:
                        break
                    if len(current) < 2:
                        break
                    
                    # Pick 2 points to remove
                    idxs_to_remove = random.sample(current, 2)
                    temp_indices = [x for x in current if x not in idxs_to_remove]
                    
                    # Pick 2 candidates to add
                    # Sample from points not in current
                    # To be efficient, sample from all points and filter
                    candidates = random.sample(range(total), 100)
                    candidates = [c for c in candidates if c not in current]
                    if len(candidates) < 2:
                        continue
                    
                    idxs_to_add = random.sample(candidates, 2)
                    
                    new_indices = temp_indices + idxs_to_add
                    
                    # Check validity of new_indices
                    if is_valid_set(new_indices):
                        current = new_indices
                        improved = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best = list(current)
                        break # Restart after a successful swap
                        
            return best

        # Run the 2-swap local search
        if best_size > 0 and time.time() < deadline - 20:
            remaining_time = max(10, deadline - 20 - time.time())
            improved_set = local_search_2swap(best_points, remaining_time)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_points = improved_set

        # Final greedy pass
        if time.time() < deadline - 5:
            current_mask = np.zeros(total, dtype=bool)
            for idx in best_points:
                current_mask[idx] = True
            remaining_pts = [j for j in range(total) if not current_mask[j]]
            random.shuffle(remaining_pts)
            for idx in remaining_pts:
                if time.time() > deadline - 5:
                    break
                if count_conflicts(idx, best_points) == 0:
                    current_mask[idx] = True
                    best_points.append(idx)

        result = [(int(xs[idx]), int(ys[idx])) for idx in best_points]
        return result

if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
