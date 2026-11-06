import numpy as np
import time
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)

# Precompute squared distances
coords = np.array(POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

# Precompute distance sets for faster lookup?
# For each point i, we have D[i, :].
# To check if D[i, p] is in D[i, S \ {i}], we can just check the array.

def is_no_isosceles(S_indices):
    """Check if set of indices is no-isosceles."""
    if len(S_indices) < 3:
        return True
    S_arr = np.array(S_indices)
    # For each i in S, distances to others in S must be unique
    for i in S_arr:
        dists = D[i, S_arr]
        # Exclude self (index where D[i, S_arr] == 0)
        # Since i is in S_arr, there is exactly one 0.
        dists = dists[dists > 0]
        if len(np.unique(dists)) < len(dists):
            return False
    return True

def can_add_p(S_list, p):
    """
    Check if S U {p} is no-isosceles, given S is no-isosceles.
    S_list is a list of indices. p is an index.
    """
    if len(S_list) == 0:
        return True
    if len(S_list) == 1:
        return True # Any 2 points form no triangle
        
    S_arr = np.array(S_list)
    
    # 1. Distances from p to S must be unique
    d_p_s = D[p, S_arr]
    if len(np.unique(d_p_s)) < len(d_p_s):
        return False
        
    # 2. For each k in S, D[k, p] must not equal any D[k, j] for j in S, j != k
    # D[k, S_arr] contains D[k, j] for all j in S.
    # We need to check if D[k, p] is in D[k, S_arr] \ {0}
    # Note: D[k, p] is not in D[k, S_arr] because p is not in S.
    # So we just check if D[k, p] is in D[k, S_arr].
    # But D[k, S_arr] includes D[k, k]=0. D[k, p] > 0.
    # So we check if D[k, p] is in the set of distances from k to S.
    
    # Vectorized check:
    # D[S_arr, p] is a vector of distances from each k in S to p.
    # D[S_arr, S_arr] is the matrix.
    # We want to know if for any row k, D[k, p] == D[k, j] for some j in S, j != k.
    # This is equivalent to: (D[S_arr, p][:, None] == D[S_arr, S_arr]).any(axis=1)
    # But we must exclude the case j=k where D[k,k]=0.
    # Since D[k, p] > 0, it will never match D[k, k]=0.
    # So we can just check the whole matrix.
    
    matches = (D[S_arr, p][:, None] == D[S_arr, S_arr]).any(axis=1)
    if np.any(matches):
        return False
        
    return True

def is_maximal(S_list):
    """Check if S is maximal. S is maximal if for all p not in S, S U {p} is NOT no-isosceles."""
    S_set = set(S_list)
    for p in range(N_POINTS):
        if p in S_set:
            continue
        if can_add_p(S_list, p):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 280  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # ILS: Iterated Local Search
    # 1. Greedy construction of a random maximal set
    # 2. Local search (removal)
    # 3. Perturbation (swap)
    
    def greedy_random_maximal():
        S = []
        S_set = set()
        perm = np.random.permutation(N_POINTS)
        for p in perm:
            if len(S) == 0:
                S.append(p)
                S_set.add(p)
                continue
            if can_add_p(S, p):
                S.append(p)
                S_set.add(p)
        # S is now maximal?
        # Greedy adds until no more can be added in the given order.
        # This ensures maximality with respect to the order, but is it globally maximal?
        # Yes, if we iterate through ALL points and add if possible, the resulting set is maximal.
        # Because if there was a point p that could be added, we would have added it when we reached it.
        # Wait, if we skip p because it couldn't be added with the CURRENT S,
        # could it be added later if S changed? No, S only grows.
        # If S grows, it becomes harder to add points (more constraints).
        # So if p couldn't be added to S_old, it can't be added to S_new (S_new superset S_old).
        # So the final set is maximal.
        return S

    def local_search_removal(S):
        improved = True
        while improved:
            improved = False
            # Try removing each point
            for i in range(len(S)):
                new_S = S[:i] + S[i+1:]
                if len(new_S) >= best_size:
                    continue
                # Check if new_S is maximal
                if is_maximal(new_S):
                    S = new_S
                    best_size = len(S)
                    improved = True
                    break
        return S

    def perturb_swap(S):
        """
        Swap a random point in S with a random point not in S.
        Try a few times to find a valid swap.
        """
        if len(S) == 0:
            return S
        S_set = set(S)
        not_in_S = [p for p in range(N_POINTS) if p not in S_set]
        if not not_in_S:
            return S
            
        for _ in range(10): # Try 10 random swaps
            i = np.random.randint(len(S))
            u = S[i]
            v = np.random.choice(not_in_S)
            
            new_S = S[:i] + S[i+1:]
            # Check if new_S U {v} is no-isosceles
            # Since new_S is subset of S, it is no-isosceles.
            # We just need to check if v can be added to new_S.
            if can_add_p(new_S, v):
                new_S.append(v)
                # Check if new_S is maximal?
                # Perturbation usually doesn't require maximality, just validity (no-isosceles).
                # The next local search step will ensure maximality by removing points?
                # No, local search removal only shrinks. It doesn't add.
                # So if we swap, we might break maximality.
                # We should check maximality or rely on the next phase to fix it?
                # If we break maximality, the set is no longer a valid solution for the task.
                # The task requires the set to be maximal.
                # So we must ensure the perturbed set is maximal, or we must go back to a maximal state.
                
                # Let's check if new_S is maximal.
                if is_maximal(new_S):
                    return new_S
        return S # No valid swap found

    # Main Loop
    while time.time() - start_time < deadline:
        S = greedy_random_maximal()
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
        
        # Local Search
        S = local_search_removal(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
            
        # Perturb and Local Search again
        S_pert = perturb_swap(S)
        if S_pert != S:
            S_pert = local_search_removal(S_pert)
            if len(S_pert) < best_size:
                best_size = len(S_pert)
                best_set = [POINTS[i] for i in S_pert]

    if best_set is None:
        # Fallback
        S = greedy_random_maximal()
        best_set = [POINTS[i] for i in S]
        
    return best_set
