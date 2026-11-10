import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Precompute compatibility bitsets
    # compat_bits[i] has bit j set if lcm(i, j) <= 512
    compat_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                bits |= (1 << (j - 1))
        compat_bits[i] = bits

    # Greedy initialization to get a good lower bound
    def greedy_seed():
        S = set()
        # Sort nodes by degree (number of compatible neighbors) descending
        # deg(i) = popcount(compat_bits[i])
        nodes_sorted = sorted(range(1, n + 1), key=lambda x: bin(compat_bits[x]).count('1'), reverse=True)
        for x in nodes_sorted:
            # Check if x is compatible with all in S
            # We can check if x conflicts with any node in S
            # Using bitsets: 
            # Let S_bits be the bitset of S.
            # If (compat_bits[x] & S_bits) == S_bits, then x is compatible with all in S?
            # No. compat_bits[x] contains neighbors.
            # We need x to be a neighbor of ALL nodes in S.
            # So (compat_bits[x] & S_bits) must equal S_bits.
            S_bits = 0
            for s in S:
                S_bits |= (1 << (s - 1))
            if (compat_bits[x] & S_bits) == S_bits:
                S.add(x)
        return list(S)

    best_clique = greedy_seed()
    best_size = len(best_clique)
    
    # Bron-Kerbosch with Pivot and Bitsets
    # R: list of nodes in current clique
    # P_bits: bitset of candidates
    # X_bits: bitset of excluded candidates
    
    def bron_kerbosch(R, P_bits, X_bits):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        # Pruning
        P_count = P_bits.bit_count()
        if len(R) + P_count <= best_size:
            return
            
        if P_bits == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Pivot selection
        # u in P ∪ X maximizing |P ∩ N(u)|
        PX_bits = P_bits | X_bits
        best_u = -1
        max_inter = -1
        
        temp_px = PX_bits
        while temp_px:
            lsb = temp_px & (-temp_px)
            u = lsb.bit_length() - 1 + 1
            temp_px ^= lsb
            
            inter = (P_bits & compat_bits[u]).bit_count()
            if inter > max_inter:
                max_inter = inter
                best_u = u
                
        if best_u == -1:
            return
            
        # Branch on P \ N(best_u)
        branch_bits = P_bits & ~compat_bits[best_u]
        
        # Get nodes to branch on, sorted by degree in P
        branch_nodes = []
        temp_b = branch_bits
        while temp_b:
            lsb = temp_b & (-temp_b)
            v = lsb.bit_length() - 1 + 1
            temp_b ^= lsb
            # Degree of v in P
            deg = (P_bits & compat_bits[v]).bit_count()
            branch_nodes.append((deg, v))
            
        # Sort by degree descending
        branch_nodes.sort(reverse=True)
        
        for deg, v in branch_nodes:
            if time.time() > deadline:
                return
                
            # Add v to R
            R.append(v)
            
            # New P = P ∩ N(v)
            new_P_bits = P_bits & compat_bits[v]
            new_P_bits &= ~(1 << (v - 1)) # Remove v
            
            # New X = X ∩ N(v)
            new_X_bits = X_bits & compat_bits[v]
            
            bron_kerbosch(R, new_P_bits, new_X_bits)
            
            R.pop()
            
            # Move v from P to X
            P_bits &= ~(1 << (v - 1))
            X_bits |= (1 << (v - 1))
            
    # Initial call
    initial_P_bits = 0
    for i in range(1, n + 1):
        initial_P_bits |= (1 << (i - 1))
    initial_X_bits = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P_bits, initial_X_bits)
    
    return best_clique
