import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    N = 512

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i * j // g
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Build compatibility graph using bitmasks
    compat = [0] * N  # compat[i] is a bitmask of neighbors of node i (0-indexed)
    
    for i in range(1, N + 1):
        idx = i - 1
        mask = 0
        for j in range(1, N + 1):
            if i != j and lcm_table[i][j] <= N:
                jidx = j - 1
                mask |= (1 << jidx)
        compat[idx] = mask

    best_clique = [0]
    best_size = 0
    
    def get_color_bound(P):
        """
        Returns the maximum color number in a greedy coloring of the subgraph induced by P.
        This is an upper bound on the clique size of the subgraph.
        """
        if P == 0:
            return 0
        
        # Extract vertices
        verts = []
        tmp = P
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            tmp ^= lsb
        
        colors = {}
        # Order vertices by degree descending to improve coloring tightness?
        # Optional, but might help.
        # For now, simple order.
        
        for v in verts:
            used_colors = set()
            # Check neighbors of v in P
            neigh = compat[v] & P
            n_tmp = neigh
            while n_tmp:
                lsb = n_tmp & (-n_tmp)
                u_idx = lsb.bit_length() - 1
                if u_idx in colors:
                    used_colors.add(colors[u_idx])
                n_tmp ^= lsb
            
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
        
        return max(colors.values()) if colors else 0

    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        if not P:
            return
        
        # Basic bound
        if len(R) + P.bit_count() <= best_size:
            return
            
        # Color bound
        # Only compute if basic bound didn't prune and P is not too large?
        # Actually, for N=512, color bound is cheap enough.
        max_c = get_color_bound(P)
        if len(R) + max_c <= best_size:
            return
            
        # Pivot selection
        U = P | X
        if U == 0:
            return
            
        max_count = -1
        pivot = -1
        u_tmp = U
        while u_tmp:
            lsb = u_tmp & (-u_tmp)
            idx = lsb.bit_length() - 1
            u_tmp ^= lsb
            count = (P & compat[idx]).bit_count()
            if count > max_count:
                max_count = count
                pivot = idx
            if count == 0:
                break
        
        if pivot == -1:
            lsb = U & (-U)
            pivot = lsb.bit_length() - 1

        candidates = P & ~(compat[pivot])
        
        # To improve search, process candidates with higher degree in P first?
        # Let's sort candidates by degree in P descending.
        cand_list = []
        c_tmp = candidates
        while c_tmp:
            lsb = c_tmp & (-c_tmp)
            idx = lsb.bit_length() - 1
            deg = (P & compat[idx]).bit_count()
            cand_list.append((deg, idx))
            c_tmp ^= lsb
            
        # Sort by degree descending
        cand_list.sort(reverse=True)
        
        for deg, v in cand_list:
            R.append(v + 1)
            
            P_new = P & compat[v]
            X_new = X & compat[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    initial_P = (1 << N) - 1
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    if best_clique:
        return best_clique
    
    divs = []
    for i in range(1, 361):
        if 360 % i == 0:
            divs.append(i)
    return divs

if __name__ == "__main__":
    print(construct())
