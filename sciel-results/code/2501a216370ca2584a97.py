import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Precompute conflict matrix using bitsets
    # conflict_bits[i] is a bitmask where bit j is 1 if lcm(i, j) > 512
    conflict_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l > n:
                bits |= (1 << (j - 1))
        conflict_bits[i] = bits
    
    # Precompute compatible sets (complement of conflict)
    # compatible_bits[i] has bit j set if j is compatible with i (or j==i)
    compatible_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        # Start with all bits set for 1..n
        all_bits = (1 << n) - 1
        compatible_bits[i] = all_bits & ~conflict_bits[i]
        # Ensure i itself is compatible
        compatible_bits[i] |= (1 << (i - 1))
    
    def is_compatible(x, S_mask):
        """Check if x is compatible with all elements in S_mask (bitset)."""
        return (conflict_bits[x] & S_mask) == 0
    
    def get_compatible_with(S_mask, x):
        """Return bitset of elements in S_mask compatible with x."""
        return S_mask & compatible_bits[x]
    
    def popcount(x):
        """Count bits in integer x."""
        return bin(x).count('1')
    
    def bits_to_list(mask):
        """Convert bitset to list of integers."""
        result = []
        i = 0
        while mask:
            if mask & 1:
                result.append(i + 1)
            mask >>= 1
            i += 1
        return result
    
    def list_to_bits(lst):
        """Convert list of integers to bitset."""
        bits = 0
        for x in lst:
            bits |= (1 << (x - 1))
        return bits
    
    # Initial best solution: divisors of highly composite numbers
    # 504 = 2^3 * 3^2 * 7 has many divisors
    # 480 = 2^5 * 3 * 5
    # 420 = 2^2 * 3 * 5 * 7
    # 360 = 2^3 * 3^2 * 5
    
    def divisors_of(m):
        return [k for k in range(1, n + 1) if m % k == 0]
    
    # Try multiple highly composite numbers
    candidates = [504, 480, 420, 360, 336, 240, 210, 180, 168, 144]
    best_seed = []
    for m in candidates:
        d = divisors_of(m)
        if len(d) > len(best_seed):
            best_seed = d
    
    # Also try greedy from random orders
    for _ in range(50):
        random.shuffle(nodes)
        S_mask = 0
        S_size = 0
        for x in nodes:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_size += 1
        if S_size > len(best_seed):
            best_seed = bits_to_list(S_mask)
    
    best_size = len(best_seed)
    best_clique = best_seed
    best_mask = list_to_bits(best_seed)
    
    # Bron-Kerbosch with bitsets for maximum speed
    # R: current clique (list for easy access)
    # P_mask: candidates that can be added (bitset)
    # X_mask: excluded candidates (bitset)
    
    def bron_kerbosch(R, P_mask, X_mask):
        """
        Bron-Kerbosch with pivot for maximum clique, using bitsets.
        R: current clique (list)
        P_mask: candidates that can be added (bitset)
        X_mask: excluded candidates (bitset)
        """
        nonlocal best_size, best_clique, best_mask
        
        if time.time() > deadline:
            return
        
        # Base case
        if P_mask == 0 and X_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
                best_mask = list_to_bits(R)
            return
        
        # If P is empty, we can't extend, but check if X is also empty
        if P_mask == 0:
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        # u should maximize |P ∩ N(u)| = |P & compatible_bits[u]|
        # We want to branch on P \ N(u), so we want u that maximizes this intersection
        union_mask = P_mask | X_mask
        best_u = -1
        best_count = -1
        
        # Iterate through bits in union_mask
        temp = union_mask
        u_idx = 0
        while temp:
            if temp & 1:
                u = u_idx + 1
                # Count |P ∩ N(u)| = popcount(P_mask & compatible_bits[u])
                count = popcount(P_mask & compatible_bits[u])
                if count > best_count:
                    best_count = count
                    best_u = u
            temp >>= 1
            u_idx += 1
        
        if best_u == -1:
            return
        
        # Branch on P \ N(best_u)
        # These are elements in P that are NOT compatible with best_u
        branch_mask = P_mask & ~compatible_bits[best_u]
        
        # Iterate through elements in branch_mask
        temp = branch_mask
        v_idx = 0
        while temp:
            if time.time() > deadline:
                return
            if temp & 1:
                v = v_idx + 1
                
                # Add v to R
                R.append(v)
                
                # New P = P ∩ N(v) \ {v}
                new_P_mask = (P_mask & compatible_bits[v]) & ~(1 << (v - 1))
                # New X = X ∩ N(v)
                new_X_mask = X_mask & compatible_bits[v]
                
                bron_kerbosch(R, new_P_mask, new_X_mask)
                
                R.pop()
                
                # Remove v from P, add to X
                P_mask &= ~(1 << (v - 1))
                X_mask |= (1 << (v - 1))
            temp >>= 1
            v_idx += 1
    
    # Run Bron-Kerbosch
    initial_R = []
    initial_P_mask = (1 << n) - 1  # All nodes
    initial_X_mask = 0
    
    bk_start = time.time()
    bron_kerbosch(initial_R, initial_P_mask, initial_X_mask)
    
    # Final greedy polish: try to add any remaining compatible elements
    if best_clique:
        S = set(best_clique)
        S_mask = list_to_bits(S)
        for x in range(1, n + 1):
            if x not in S and is_compatible(x, S_mask):
                S.add(x)
                S_mask |= (1 << (x - 1))
        best_clique = list(S)
        best_size = len(S)
    
    return best_clique if best_clique else [1]
