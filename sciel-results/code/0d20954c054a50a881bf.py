import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit

    N = 64
    M = N * N
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)
    
    # Precompute squared distance matrix
    # dist[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sum(diff**2, axis=-1).astype(np.int32)
    del diff # Free memory

    def is_valid_candidate(c, S_indices):
        """
        Check if point c can be added to set S_indices.
        S_indices is a numpy array of indices in S.
        Returns True if valid.
        """
        if len(S_indices) == 0:
            return True
            
        # 1. Distances from c to all points in S
        dists_c = dist_matrix[c, S_indices]
        
        # Check uniqueness of distances from c
        # If any distance repeats, c forms isosceles with apex c
        if len(np.unique(dists_c)) < len(dists_c):
            return False
        
        # 2. Check if any dist(c, p) equals dist(p, q) for some q in S
        # For each p in S, we check if dist(c, p) is in dist_matrix[p, S]
        # dist_matrix[S_indices, S_indices] is a |S|x|S| matrix
        # We want to check if dists_c[i] is in row i of dist_matrix[S_indices, S_indices]
        
        # Optimization: We only need to check if the value exists in the row.
        # dist_matrix[S_indices, S_indices] == dists_c[:, None] creates a |S|x|S| boolean matrix.
        # This might be memory heavy if |S| is large, but |S| is likely < 200.
        # 200x200 = 40k booleans. Very small.
        
        # However, we must exclude the case where q=p.
        # dist(p, p) = 0. dist(c, p) > 0. So if dist(c, p) == 0, it's invalid anyway (c in S).
        # Since c is not in S, dist(c, p) > 0.
        # So we just check if dists_c[i] appears in dist_matrix[S_indices[i], S_indices].
        
        # Vectorized check:
        # sub_matrix = dist_matrix[S_indices, S_indices]
        # matches = (sub_matrix == dists_c[:, None]).any(axis=1)
        # if matches.any(): return False
        
        # To save memory/time, we can do this in chunks or just let numpy handle it.
        # Given |S| is small, direct comparison is fine.
        
        sub_matrix = dist_matrix[S_indices, S_indices]
        # dists_c is shape (K,). We compare each dists_c[i] against sub_matrix[i, :]
        # This is equivalent to checking if dists_c[i] is in sub_matrix[i, :]
        
        # We can use np.isin? No, np.isin is for 1D arrays.
        # Let's use the broadcasting approach.
        # Create a mask of shape (K, K)
        # mask[i, j] is True if dist(S[i], S[j]) == dist(c, S[i])
        
        # Actually, we can avoid the full KxK matrix if we are clever, 
        # but K is small.
        
        # Check: does row i of sub_matrix contain dists_c[i]?
        # We can do:
        # for i in range(len(S_indices)):
        #     if dists_c[i] in sub_matrix[i, :]: return False
        # This is a Python loop over K, but each step is a fast C-level search in a small array.
        # Or we stick to full vectorization.
        
        # Full vectorization:
        # This creates a KxK temporary array.
        matches = (sub_matrix == dists_c[:, None]).any(axis=1)
        if matches.any():
            return False
            
        return True

    def greedy_construct():
        S = []
        S_indices = np.array([], dtype=np.int32)
        candidates = list(range(M))
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if is_valid_candidate(c, S_indices):
                S.append(c)
                S_indices = np.append(S_indices, c)
                # Optional: break if S is large enough? No, try to fill.
        
        return S

    def local_search(S):
        S = list(S)
        S_indices = np.array(S, dtype=np.int32)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 5:
            improved = False
            no_improve_count += 1
            
            # 1. Try to add a point
            candidates = [i for i in range(M) if i not in S]
            random.shuffle(candidates)
            
            # Limit candidates checked to save time if S is large
            check_limit = min(len(candidates), 500) 
            for c in candidates[:check_limit]:
                if time.time() > deadline - 5:
                    break
                if is_valid_candidate(c, S_indices):
                    S.append(c)
                    S_indices = np.append(S_indices, c)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue

            # 2. If no addition, try a swap
            if len(S) > 0:
                # Remove a random point
                idx_to_remove = random.randint(0, len(S) - 1)
                p_removed = S.pop(idx_to_remove)
                S_indices = np.delete(S_indices, idx_to_remove)
                
                # Try to add a different point
                candidates = [i for i in range(M) if i not in S and i != p_removed]
                random.shuffle(candidates)
                
                added = False
                check_limit = min(len(candidates), 500)
                for c in candidates[:check_limit]:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_candidate(c, S_indices):
                        S.append(c)
                        S_indices = np.append(S_indices, c)
                        added = True
                        improved = True
                        no_improve_count = 0
                        break
                
                if not added:
                    # Revert
                    S.append(p_removed)
                    S_indices = np.append(S_indices, p_removed)
                    # If we couldn't add anything, we might be stuck.
                    # Try a 2-swap? Or just break and restart.
                    if no_improve_count > 5:
                        break
            else:
                break
                
        return S

    best_S = []
    
    # Multiple restarts
    # We have ~280s. Greedy is fast. Local search is slower.
    # Let's do 50 restarts.
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        S = greedy_construct()
        S = local_search(S)
        
        if len(S) > len(best_S):
            best_S = S.copy()
            
    # Final polish
    if time.time() < deadline - 10:
        best_S = local_search(best_S)

    return [points[i] for i in best_S]

if __name__ == "__main__":
    res = construct()
    print(len(res))
