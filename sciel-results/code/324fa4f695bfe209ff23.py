import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin within 90s limit
    
    # 255 = 3 * 5 * 17
    # Candidates must share a factor with 255 to have gcd > 1 with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n = len(candidates)
    
    # Build adjacency using bitsets
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Find index of 255 in candidates
    idx_255 = val_to_idx[255]
    
    # Bron-Kerbosch with pivot to find maximum clique containing 255
    best_clique = [255]
    
    def bronk(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pivot selection: choose vertex from P ∪ X with maximum degree in P
        U = P | X
        if U == 0:
            return
        
        # Find pivot with max neighbors in P
        max_deg = -1
        pivot = 0
        u = U
        while u:
            lsb = u & (-u)
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            u ^= lsb
        
        # Candidates: P minus neighbors of pivot
        P_cand = P & ~(adj[pivot] | (1 << pivot))
        
        u = P_cand
        while u:
            lsb = u & (-u)
            v = lsb.bit_length() - 1
            u ^= lsb
            
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bronk(new_R, new_P, new_X)
            
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Initial P: all candidates adjacent to 255 (which is all candidates since they share factor with 255)
    # Actually, P should be all candidates that can be in a clique with 255
    # Since all candidates have gcd > 1 with 255, all are potential members
    # But we need to ensure the final clique contains 255, so we start with R = {255}
    # and P = all other candidates that are adjacent to 255
    
    initial_P = 0
    for i in range(n):
        if i != idx_255 and (adj[idx_255] & (1 << i)):
            initial_P |= (1 << i)
    
    bronk(1 << idx_255, initial_P, 0)
    
    # Convert bitset back to values
    result = []
    for i in range(n):
        if best_clique[i] if isinstance(best_clique, list) and len(best_clique) == n else False:
            pass
    
    # Wait, best_clique is stored as a list of values when updated
    # Let me fix the bronk function to store the bitset and convert at the end
    
    # Actually, let me redo this more carefully
    best_clique_bits = 1 << idx_255
    
    def bronk2(R, P, X):
        nonlocal best_clique_bits
        if time.time() > deadline:
            return
        
        if not P and not X:
            if R.bit_count() > best_clique_bits.bit_count():
                best_clique_bits = R
            return
        
        U = P | X
        if U == 0:
            return
        
        max_deg = -1
        pivot = 0
        u = U
        while u:
            lsb = u & (-u)
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            u ^= lsb
        
        P_cand = P & ~(adj[pivot] | (1 << pivot))
        
        u = P_cand
        while u:
            lsb = u & (-u)
            v = lsb.bit_length() - 1
            u ^= lsb
            
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bronk2(new_R, new_P, new_X)
            
            P &= ~(1 << v)
            X |= (1 << v)
    
    bronk2(1 << idx_255, initial_P, 0)
    
    # Convert bitset to list of values
    result = []
    for i in range(n):
        if best_clique_bits & (1 << i):
            result.append(candidates[i])
    
    return sorted(result)
# EVOLVE-BLOCK-END
