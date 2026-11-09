import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_TIME = 300

# Precompute coordinates
xs = np.arange(NUM_POINTS) // N
ys = np.arange(NUM_POINTS) % N

# Precompute squared distance matrix
# dist2[i, j] = (xi - xj)^2 + (yi - yj)^2
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2_mat = (dx * dx + dy * dy).astype(np.int32)

# For each point i, we need to know which distances it has to other points.
# To check if adding point p to set S is valid:
# 1. No two points in S can be at the same distance from p.
#    i.e., len(set(dist2_mat[p, S])) == len(S)
# 2. For every a in S, dist2_mat[p, a] must not be equal to dist2_mat[b, a] for any b in S.
#    i.e., for all a in S, dist2_mat[p, a] is not in {dist2_mat[b, a] for b in S}.

# To speed up local search, we can maintain a "conflict count" or similar,
# but for Simulated Annealing, we often need to evaluate moves.
# A move is typically: remove one point, add one point (swap), or just add/remove.
# Given the tight time limit and the need for a structural change, 
# we will implement a fast "Add/Remove" search with vectorized checks where possible,
# but primarily focus on a smarter heuristic: 
# "Maximally Compatible" Greedy with Randomized Restarts and Annealing-like Perturbation.

# Actually, a pure Simulated Annealing on 4096 points with O(N^2) validity checks is too slow in Python loops.
# Let's stick to the "Greedy + Local Search" family but change the *search operator* and *evaluation* strategy.
# New Strategy: 
# 1. Use a precomputed "distance frequency" approach.
# 2. Instead of random swaps, use "Conflict Resolution":
#    - Start with a large random set.
#    - Identify all isosceles triples.
#    - Remove points involved in the most conflicts.
#    - Try to re-add points that were removed, prioritizing those with fewer conflicts.
#    - This is a "Local Search via Conflict Minimization".

# Precompute for each point i, a list of (j, d2) sorted by d2? No, that's huge.
# Let's use the existing dist2_mat.

def count_conflicts(S_set, dist2_mat):
    """
    Returns a list of (point_index, conflict_count) for points in S_set.
    A conflict for point p is a pair (a, b) in S_set such that {p, a, b} is isosceles.
    We count how many isosceles triangles each point participates in.
    """
    S_list = list(S_set)
    n = len(S_list)
    if n < 3:
        return [(p, 0) for p in S_list]
    
    # conflicts[p] = number of isosceles triangles p is part of
    conflicts = np.zeros(NUM_POINTS, dtype=np.int32)
    
    # This O(N^3) check is too slow if done naively in Python.
    # We need a faster way to count conflicts or just find *one* conflict to resolve.
    # For the purpose of the "Conflict Resolution" strategy, we can just find *any* isosceles triangle,
    # remove one of its points, and repeat. This is effectively a local search.
    
    # To make it faster, we'll just scan for the first few conflicts.
    # But scanning is O(N^3).
    
    # Alternative: 
    # Maintain the set S.
    # In each step, pick a random point p in S.
    # Check if it forms an isosceles triangle with any two other points in S.
    # If yes, remove it.
    # If no, keep it.
    # This is similar to the current "remove if invalid" but more aggressive.
    
    # Let's try a different structural change:
    # **Genetic Algorithm / Crossover**
    # Maintain a population of sets.
    # Crossover: Take two sets A and B. 
    # Create child C by taking points from A that are compatible with B, and vice versa?
    # This is complex.
    
    # Let's go back to **Simulated Annealing** but with a **Fast Validity Check** using NumPy.
    # The key is to check if a point p can be added to S quickly.
    # Current check: O(|S|) in Python.
    # NumPy check: 
    #   d_to_S = dist2_mat[p, S_array]
    #   1. Are all d_to_S distinct? -> np.unique(d_to_S).size == len(d_to_S)
    #   2. For each a in S, is d_to_S[a] unique in the column a of dist2_mat restricted to S?
    #      This is harder to vectorize across all a simultaneously without advanced indexing.
    
    # Given the 300s limit, and the current score of 10.5, 
    # the bottleneck is likely the Python overhead in the local search loop.
    
    # New Strategy: **Batched Greedy with NumPy Vectorization**
    # 1. Generate a random ordering of points.
    # 2. Process points in batches? No, dependencies are sequential.
    
    # Let's try **Iterated Local Search (ILS)** with a **Strong Perturbation**.
    # 1. Run a fast greedy to get a solution S.
    # 2. Perturb S by removing k random points.
    # 3. Re-run greedy on the remaining points + new candidates.
    # 4. Keep the best.
    
    # To make the greedy faster, we will precompute for each point p, 
    # the "distance profile" to all other points? No.
    
    # Let's optimize the `can_add` function using NumPy.
    
    def can_add_numpy(p, S_array):
        """
        Returns True if point p can be added to the set represented by S_array (np.array of indices).
        """
        if len(S_array) == 0:
            return True
        
        # Distances from p to all points in S
        d_ps = dist2_mat[p, S_array]
        
        # Check 1: Distances from p to points in S must be distinct.
        # If there are duplicates, p is the apex of an isosceles triangle.
        if len(np.unique(d_ps)) != len(d_ps):
            return False
            
        # Check 2: For each a in S, the distance d_ps[a] must not be equal to any distance d_sb where b in S and b != a.
        # Actually, the condition is: for any a, b in S (a != b), d(p, a) != d(p, b) [Already checked]
        # AND for any a, b in S (a != b), d(a, p) != d(b, a) ? No.
        # The condition for no isosceles triangle with vertices {p, a, b} is:
        # d(p,a) != d(p,b)  (Apex p)
        # d(a,p) != d(a,b)  (Apex a)
        # d(b,p) != d(a,b)  (Apex b)
        
        # So for each a in S, we need d(p, a) != d(b, a) for all b in S, b != a.
        # This means d(p, a) must not be in the set of distances {d(b, a) for b in S, b != a}.
        
        # Let's compute the matrix of distances between points in S: D_S = dist2_mat[S_array[:, None], S_array[None, :]]
        # For a specific a at index i in S_array:
        # We need d_ps[i] (which is d(p, S_array[i])) to not be in D_S[i, :] excluding D_S[i, i] (which is 0).
        
        # This check is O(N^2) if done naively.
        # Can we vectorize?
        # We want to check if d_ps[i] appears in D_S[i, :] (excluding index i).
        
        # Let's create a boolean mask for D_S where values match d_ps broadcasted.
        # D_S is (n, n). d_ps is (n,).
        # We want to check for each row i: is d_ps[i] in D_S[i, :]?
        
        # Matches = (D_S == d_ps[:, None])
        # But we must exclude the diagonal (i,i) where distance is 0.
        # And we must ensure that if d_ps[i] == 0 (p is in S, which shouldn't happen), it's handled.
        
        # Actually, d_ps[i] is the distance from p to S[i].
        # D_S[i, j] is the distance from S[i] to S[j].
        # We need to check if d_ps[i] == D_S[i, j] for any j != i.
        
        # Let's compute Matches = (D_S == d_ps[:, None])
        # Then, for each row i, if there is any True in Matches[i, :] at index j != i, then it's a conflict.
        
        # We can set the diagonal of Matches to False to ignore the self-distance.
        # Then, if any row has a True, it's a conflict.
        
        # Memory: D_S is n x n. n can be up to ~100-200. 200x200 is small.
        
        D_S = dist2_mat[S_array[:, None], S_array[None, :]]
        
        # d_ps is shape (n,)
        # D_S is shape (n, n)
        
        # Check if d_ps[i] is in D_S[i, :] for j != i
        matches = (D_S == d_ps[:, None])
        
        # Exclude diagonal
        np.fill_diagonal(matches, False)
        
        # If any row has a True, it means for that point a=S[i], there exists b=S[j] such that d(a,p) == d(a,b).
        # This forms an isosceles triangle with apex a.
        
        if np.any(matches):
            return False
            
        return True

    # This NumPy check is O(N^2) but with low constant factor due to vectorization.
    # For N=100, it's very fast.
    
    # Strategy:
    # 1. Multiple Greedy Restarts using `can_add_numpy`.
    # 2. Local Search: 
    #    - Try to add a random point.
    #    - If fails, try to remove a random point and add a different one.
    #    - Use `can_add_numpy` for validation.
    
    def greedy_construct(order):
        S = []
        S_array = np.array([], dtype=np.int32)
        
        for p in order:
            if can_add_numpy(p, S_array):
                S.append(p)
                S_array = np.append(S_array, p)
        
        return S, S_array

    def local_search(S, S_array, deadline):
        # S is list, S_array is np.array
        best_S = S[:]
        best_array = S_array.copy()
        best_size = len(S)
        
        stagnation = 0
        
        while time.time() < deadline:
            moved = False
            # Randomly choose an operation
            r = random.random()
            
            if r < 0.6:
                # Try to add a random point
                p = random.randint(0, NUM_POINTS - 1)
                if p not in S_array: # Simple check, S_array is small
                    if can_add_numpy(p, S_array):
                        S.append(p)
                        S_array = np.append(S_array, p)
                        moved = True
            else:
                # Try to swap: remove one, add one
                if len(S) > 0:
                    # Remove a random point
                    idx = random.randint(0, len(S) - 1)
                    p_remove = S.pop(idx)
                    S_array = np.delete(S_array, idx)
                    
                    # Try to add a new point
                    added = False
                    for _ in range(20): # Try 20 random candidates
                        p_add = random.randint(0, NUM_POINTS - 1)
                        if p_add not in S_array and p_add != p_remove:
                            if can_add_numpy(p_add, S_array):
                                S.append(p_add)
                                S_array = np.append(S_array, p_add)
                                added = True
                                moved = True
                                break
                    
                    if not added:
                        # Revert removal
                        S.append(p_remove)
                        S_array = np.append(S_array, p_remove)
            
            if moved:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                    best_array = S_array.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
            
            # Perturbation if stuck
            if stagnation > 50:
                # Remove 10% of points
                num_remove = max(1, len(S) // 10)
                for _ in range(num_remove):
                    if S:
                        idx = random.randint(0, len(S) - 1)
                        S.pop(idx)
                        S_array = np.delete(S_array, idx)
                stagnation = 0
        
        return best_S, best_array

    def construct():
        start_time = time.time()
        deadline = start_time + 280
        
        best_S_global = []
        best_size_global = 0
        
        # Phase 1: Multiple Greedy Restarts
        # We will run several greedy passes.
        # To diversify, we can use different random seeds or orderings.
        
        num_restarts = 20 # Adjust based on speed
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            
            order = list(range(NUM_POINTS))
            random.shuffle(order)
            
            S, S_array = greedy_construct(order)
            
            if len(S) > best_size_global:
                best_size_global = len(S)
                best_S_global = S[:]
            
            # Local Search on the result of this greedy
            S_ls, S_array_ls = local_search(S, S_array, min(time.time() + 10, deadline))
            
            if len(S_ls) > best_size_global:
                best_size_global = len(S_ls)
                best_S_global = S_ls[:]
        
        # Convert indices back to coordinates
        return [(i // N, i % N) for i in best_S_global]

# Test/Run
if __name__ == "__main__":
    res = construct()
    print(f"Score: {len(res)}")
