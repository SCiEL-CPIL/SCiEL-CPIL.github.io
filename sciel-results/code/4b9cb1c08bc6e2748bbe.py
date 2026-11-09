import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Precompute all points
    # points[i] = (x, y) where i = x * N + y
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances between all pairs? 
    # M=4096. M^2 = ~16 million. 
    # Storing 16M int32s is 64MB. Acceptable.
    # dist[i, j] = squared distance between point i and point j
    # This allows O(1) distance lookup.
    
    # However, we need to check if adding point p to set S creates an isosceles triangle.
    # Condition: For any distinct a, b in S, dist(p, a) != dist(p, b) AND dist(p, a) != dist(a, b) AND dist(p, b) != dist(a, b).
    # Actually, the condition is that no three points form an isosceles triangle.
    # If we have set S and add p:
    # 1. Check if p is apex: dist(p, a) == dist(p, b) for any a, b in S.
    # 2. Check if a is apex: dist(a, p) == dist(a, b) for any b in S. (This is covered by maintaining distance sets for each point in S).
    # 3. Check if b is apex: dist(b, p) == dist(b, a) for any a in S. (Symmetric to 2).
    
    # So we need:
    # - For the new point p: all distances to existing points must be unique.
    # - For each existing point a: dist(a, p) must not be in the set of distances from a to other points in S.
    
    # Representation:
    # S: set of point indices
    # D: dict mapping point index -> set of squared distances to other points in S
    
    # Precomputing full distance matrix might be slow to build (16M ops) but fast to access.
    # Let's try to build it.
    # Actually, building it in Python loop is slow. Use NumPy broadcasting.
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.sum(diff**2, axis=2)
    # This creates a 4096x4096 array.
    
    # Let's check memory and time.
    # 4096^2 * 4 bytes = 67 MB.
    # Calculation time: vectorized, should be fast (<1s).
    
    # Let's do it.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.sum(diff * diff, axis=2).astype(np.int32)
    
    # Free memory
    del diff
    
    def try_add(S, D, p_idx):
        """
        Attempts to add point p_idx to set S.
        Returns True if successful, updating S and D.
        """
        if p_idx in S:
            return False
            
        if len(S) == 0:
            S.add(p_idx)
            D[p_idx] = set()
            return True
            
        # Get distances from p_idx to all points in S
        S_list = list(S)
        # dists_matrix[p_idx] gives distances to all M points.
        # We only need those in S.
        # But accessing a subset of a numpy array is fast.
        p_dists = dists_matrix[p_idx, S_list]
        
        # Check 1: p is apex. All distances must be unique.
        # Since p_dists is a numpy array of ints, we can check uniqueness.
        # Note: dists are squared distances, so they are integers.
        # If len(set(p_dists)) < len(p_dists), there's a duplicate.
        # Using set() on numpy array converts to Python ints, which is O(N).
        # For small S, this is fine.
        
        if len(set(p_dists.tolist())) != len(p_dists):
            return False
            
        # Check 2: Any existing point is apex.
        # For each s in S, check if dist(s, p) is already in D[s].
        # dist(s, p) is dists_matrix[s, p_idx].
        # We have p_dists[i] which is dists_matrix[p_idx, S_list[i]].
        # Since dist is symmetric, dists_matrix[p_idx, S_list[i]] == dists_matrix[S_list[i], p_idx].
        
        for i, s_idx in enumerate(S_list):
            d_val = int(p_dists[i])
            if d_val in D[s_idx]:
                return False
                
        # Validity confirmed. Update data structures.
        S.add(p_idx)
        D[p_idx] = set(p_dists.tolist())
        
        # Update distance sets for all existing points
        for i, s_idx in enumerate(S_list):
            D[s_idx].add(int(p_dists[i]))
            
        return True

    def remove_point(S, D, p_idx):
        """
        Removes point p_idx from set S and updates D.
        """
        if p_idx not in S:
            return
            
        S.discard(p_idx)
        D.pop(p_idx, None)
        
        # Update D for all remaining points
        for s in S:
            d_val = int(dists_matrix[p_idx, s])
            D[s].discard(d_val)

    def greedy_construct():
        """
        Constructs a set using a greedy strategy.
        """
        S = set()
        D = {}
        order = list(range(M))
        random.shuffle(order)
        
        for p_idx in order:
            try_add(S, D, p_idx)
            
        return S, D

    def iterated_greedy(S, D, time_limit):
        """
        Iterated Greedy: Remove k points, then greedily add back.
        """
        best_S = set(S)
        best_D = {k: set(v) for k, v in D.items()}
        best_size = len(best_S)
        
        while time.time() < time_limit:
            current_S = set(S)
            current_D = {k: set(v) for k, v in D.items()}
            
            # Remove some points
            # Adaptive k: start small, increase if stuck?
            # For now, random 2-5 points.
            if len(current_S) < 5:
                k = 1
            else:
                k = random.randint(2, 5)
                
            if k > len(current_S):
                k = len(current_S)
                
            to_remove = random.sample(list(current_S), k)
            for p in to_remove:
                remove_point(current_S, current_D, p)
                
            # Greedily add back
            # We can try to add the removed points first, or any points.
            # Let's try to add any points, prioritizing the removed ones?
            # Or just shuffle all candidates.
            
            candidates = list(range(M))
            random.shuffle(candidates)
            
            # Put removed points at the front to encourage restoration?
            # Or maybe not, to allow new configurations.
            # Let's just shuffle all.
            
            for p_idx in candidates:
                if time.time() > time_limit:
                    break
                try_add(current_S, current_D, p_idx)
                
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
                best_D = {k: set(v) for k, v in current_D.items()}
                S = set(current_S)
                D = {k: set(v) for k, v in current_D.items()}
                
        return best_S, best_D

    # Phase 1: Initial Greedy
    S, D = greedy_construct()
    
    # Phase 2: Local Search / Iterated Greedy
    # We will use the remaining time for this.
    # Split time: 10s for initial greedy (already done), rest for IG.
    
    # Actually, let's run multiple initial greeds to find a good start.
    best_S = set()
    best_D = {}
    best_size = 0
    
    t_phase1_end = start_time + 30
    
    while time.time() < t_phase1_end:
        S, D = greedy_construct()
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            best_D = {k: set(v) for k, v in D.items()}
            
    S = set(best_S)
    D = {k: set(v) for k, v in best_D.items()}
    
    # Phase 3: Iterated Greedy
    # Run until deadline - 5s
    S, D = iterated_greedy(S, D, deadline - 5)
    
    # Return the best solution found
    return [tuple(points[i]) for i in S]

# Quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
