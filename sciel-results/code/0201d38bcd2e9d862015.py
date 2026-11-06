import math
import time

def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        
        # 1. Generate Candidates
        # x in [1, 255] such that gcd(x, 255) > 1
        candidates = []
        for i in range(1, 256):
            if math.gcd(i, 255) > 1:
                candidates.append(i)
        
        n = len(candidates)
        val_to_idx = {v: i for i, v in enumerate(candidates)}
        
        # 2. Build Adjacency Bitmasks
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * n
        for i in range(n):
            vi = candidates[i]
            mask = 0
            for j in range(n):
                if i == j: continue
                vj = candidates[j]
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
            adj[i] = mask
            
        # 3. Bron-Kerbosch with Pivoting
        best_clique = []
        best_size = 0
        
        def popcount(x):
            return bin(x).count('1')
            
        def bronk(R, P, X):
            nonlocal best_clique, best_size
            
            # Pruning
            if len(R) + popcount(P) <= best_size:
                return
                
            if P == 0 and X == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return
            
            # Check time
            if time.time() > deadline:
                return

            # Pivot selection
            # We want u in P U X that maximizes popcount(P & adj[u])
            # If P is not empty, we can restrict pivot to P? No, X is allowed.
            # Standard: u in P U X.
            
            candidates_union = P | X
            if candidates_union == 0:
                return # Should be caught by P==0 and X==0? No, if P!=0 but no candidates? Impossible.
                
            # Find best pivot
            best_pivot = -1
            max_inter = -1
            temp = candidates_union
            while temp:
                b = temp & -temp
                u = b.bit_length() - 1
                temp ^= b
                inter = P & adj[u]
                cnt = popcount(inter)
                if cnt > max_inter:
                    max_inter = cnt
                    best_pivot = u
                    if cnt == popcount(P):
                        break # Can't get better than |P|
                        
            # Iterate over P \ N(best_pivot)
            # Nodes to process: P & ~adj[best_pivot]
            to_process = P & ~adj[best_pivot]
            
            temp = to_process
            while temp:
                b = temp & -temp
                v = b.bit_length() - 1
                temp ^= b
                
                # Recurse
                bronk(R + [v], P & adj[v], X & adj[v])
                
                P ^= b # Remove v from P
                X |= b # Add v to X
                
                # Early stop if we found a clique of size n? Unlikely.
                
        # Initial call
        # R = []
        # P = all nodes (mask with n bits set)
        # X = 0
        initial_P = (1 << n) - 1
        bronk([], initial_P, 0)
        
        # 4. Verify 255 is in best_clique
        # As proven, max clique MUST contain 255.
        # But just in case, if 255 is not in best_clique, we might have an issue.
        # However, the proof holds.
        
        # Convert indices back to values
        result = [candidates[i] for i in best_clique]
        
        # Ensure 255 is in result (sanity check)
        if 255 not in result:
             # This shouldn't happen, but if it does, fallback to multiples of 3
             result = [i for i in range(3, 256, 3)]
             
        return sorted(result)
