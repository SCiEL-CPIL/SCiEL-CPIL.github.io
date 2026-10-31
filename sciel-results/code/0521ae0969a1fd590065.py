import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130

    # 1. Generate weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    N_supports = len(supports)
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    N = N_supports * 16
    
    vectors = np.zeros((N, D), dtype=np.int64)
    
    for s_idx, support in enumerate(supports):
        for sp_idx, signs in enumerate(sign_patterns):
            idx = s_idx * 16 + sp_idx
            for pos, s in zip(support, signs):
                vectors[idx, pos] = s
                
    # 2. Build conflict masks
    # conflict_mask[i] is a bitset of vertices conflicting with i
    
    # Precompute support sharing 3
    support_sharing_3 = {s_idx: [] for s_idx in range(N_supports)}
    for i_s in range(N_supports):
        s1 = set(supports[i_s])
        for j_s in range(i_s + 1, N_supports):
            s2 = set(supports[j_s])
            if len(s1.intersection(s2)) == 3:
                support_sharing_3[i_s].append(j_s)
                support_sharing_3[j_s].append(i_s)
                
    conflict_masks = [0] * N
    
    for i in range(N):
        s_idx = i // 16
        sp_idx = i % 16
        support = supports[s_idx]
        signs = sign_patterns[sp_idx]
        vi = vectors[i]
        
        mask = 0
        for j_s in support_sharing_3[s_idx]:
            j_support = supports[j_s]
            inter = set(support).intersection(set(j_support))
            inter_list = list(inter)
            
            for j_sp in range(16):
                j_idx = j_s * 16 + j_sp
                vj = vectors[j_idx]
                
                match = True
                for pos in inter_list:
                    if vi[pos] != vj[pos]:
                        match = False
                        break
                if match:
                    mask |= (1 << j_idx)
                    
        conflict_masks[i] = mask

    # 3. Tomita MIS with Bitsets
    
    best_mis = []
    
    def greedy_color(P):
        # P is a bitset (int)
        # Returns list of color classes (lists of vertex indices)
        # We need to iterate vertices in P
        # And color them
        # To make it fast, we can extract vertices
        
        # Extract vertices from bitset
        # This is O(popcount(P))
        # We can use a helper to get set bits
        
        colors = {}
        color_classes = []
        
        # Iterate through set bits in P
        # Python int bit iteration:
        # while P:
        #   lsb = P & -P
        #   idx = lsb.bit_length() - 1
        #   P ^= lsb
        
        # But we need to color based on neighbors in P.
        # For a vertex v, its neighbors in P are `conflict_masks[v] & P`.
        
        # We process vertices in some order.
        # Standard Tomita sorts by degree in P? Or just arbitrary?
        # Arbitrary is fine for now.
        
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            temp_P ^= lsb
            
            # Find used colors
            used = set()
            neigh = conflict_masks[v] & P
            # Iterate neighbors in P
            t_neigh = neigh
            while t_neigh:
                lsb_n = t_neigh & -t_neigh
                u = lsb_n.bit_length() - 1
                t_neigh ^= lsb_n
                if u in colors:
                    used.add(colors[u])
            
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            while len(color_classes) <= c:
                color_classes.append([])
            color_classes[c].append(v)
            
        return color_classes

    def tomita(P, S, color_classes):
        nonlocal best_mis
        
        if time.time() > deadline - 5:
            return
            
        if P == 0:
            if len(S) > len(best_mis):
                best_mis = S.copy()
            return
            
        # Process color classes in reverse
        for i in range(len(color_classes) - 1, -1, -1):
            C = color_classes[i]
            if not C:
                continue
            
            # Bound
            # Sum of sizes of classes 0..i
            bound_size = sum(len(color_classes[k]) for k in range(i + 1))
            if len(S) + bound_size <= len(best_mis):
                return
            
            for v in C:
                # Check if v is still in P
                if not (P & (1 << v)):
                    continue
                
                # Include v
                # New P = P & ~(conflict_masks[v] | (1 << v))
                mask = conflict_masks[v] | (1 << v)
                new_P = P & ~mask
                
                # Recurse
                new_colors = greedy_color(new_P)
                tomita(new_P, S + [v], new_colors)
                
                # Exclude v
                P &= ~(1 << v)
                
                if P == 0:
                    break

    P = (1 << N) - 1
    S = []
    initial_colors = greedy_color(P)
    
    tomita(P, S, initial_colors)
    
    # 4. Add weight-1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # 5. Combine
    if best_mis:
        vectors_v4 = vectors[best_mis]
        final_vectors = np.vstack([vectors_v4, vectors_v1])
    else:
        final_vectors = vectors_v1
        
    return final_vectors

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start}, Count: {len(res)}")
