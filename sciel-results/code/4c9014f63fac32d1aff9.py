import math
import time

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Build candidate set: all x in [1,255] with gcd(x,255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {val: i for i, val in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # 2. Build adjacency bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 3. Bron-Kerbosch with pivot and bitmasks
    best_clique_indices = [idx_255]
    best_size = 1

    # Initial lower bound: all multiples of 3 in candidates
    init_clique = [i for i, val in enumerate(candidates) if val % 3 == 0]
    if idx_255 in init_clique:
        best_clique_indices = init_clique
        best_size = len(best_clique_indices)

    def bron_kerbosch(R, P, X):
        nonlocal best_clique_indices, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique_indices = R.copy()
            return
        
        if len(R) + bin(P).count('1') <= best_size:
            return
        
        # Pivot selection
        U = P | X
        pivot = -1
        max_neighbors = -1
        
        temp_U = U
        while temp_U:
            lsb = temp_U & -temp_U
            u = lsb.bit_length() - 1
            temp_U ^= lsb
            
            count = bin(adj[u] & P).count('1')
            if count > max_neighbors:
                max_neighbors = count
                pivot = u
                
        if pivot != -1:
            candidates_mask = P & ~(adj[pivot])
        else:
            candidates_mask = P
            
        temp_cand = candidates_mask
        while temp_cand:
            if time.time() > deadline:
                return
            
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            temp_cand ^= lsb
            
            new_R = R + [v]
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
    # Start with 255 forced in the clique
    R = [idx_255]
    P = adj[idx_255]
    X = 0
    
    bron_kerbosch(R, P, X)
    
    result_set = set(candidates[i] for i in best_clique_indices)
    
    if 255 not in result_set:
        result_set.add(255)
        
    res_list = list(result_set)
    if not res_list:
        return [255]
        
    valid = True
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid or 255 not in res_list:
        return list(range(3, 256, 3))
        
    return res_list

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
