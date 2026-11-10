import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Precompute conflict matrix using bitsets
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

    def is_compatible(x, S):
        """Check if x can be added to set S (S is a set of ints)"""
        bits_x = conflict_bits[x]
        for c in S:
            if bits_x & (1 << (c - 1)):
                return False
        return True

    def greedy_polish(S):
        """Greedily add all compatible elements to S, return new set"""
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in S and is_compatible(x, S):
                    S.add(x)
                    changed = True
        return S

    # Initial seed: divisors of 504
    seed0 = [k for k in range(1, n + 1) if 504 % k == 0]
    best_size = [len(seed0)]
    best_clique = [list(seed0)]
    
    # Run Bron-Kerbosch with time limit
    bk_start = time.time()
    bk_deadline = bk_start + 30.0  # Give 30 seconds to Bron-Kerbosch
    
    # Precompute for faster operations
    # For each node, precompute its non-neighbors (compatible nodes) as a bitset
    # compat_bits[i] has bit j set if lcm(i,j) <= 512 (i != j)
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

    def bron_kerbosch_timed(R, P_bits, X_bits, R_set):
        """
        Bron-Kerbosch with pivot for maximum clique, using bitsets.
        R: list of nodes in current clique
        P_bits: bitset of candidates that can be added
        X_bits: bitset of excluded candidates
        R_set: set of nodes in R for quick lookup
        """
        if time.time() > bk_deadline:
            return
        
        # Pruning: if |R| + popcount(P_bits) <= best_size, no improvement possible
        P_count = bin(P_bits).count('1')
        if len(R) + P_count <= best_size[0]:
            return
        
        if P_bits == 0 and X_bits == 0:
            if len(R) > best_size[0]:
                best_size[0] = len(R)
                best_clique[0] = list(R)
            return
        
        if P_bits == 0:
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        # N(u) in compatibility graph = non-conflict with u
        # We want to minimize branching: branch on P \ N(u)
        # So pick u that maximizes |P ∩ compat_bits[u]|
        
        # Get all nodes in P ∪ X
        PX_bits = P_bits | X_bits
        # Extract nodes from bitset
        PX_nodes = []
        temp = PX_bits
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            PX_nodes.append(idx + 1)
            temp ^= lsb
        
        best_u = None
        best_count = -1
        for u in PX_nodes:
            # Count |P ∩ compat_bits[u]|
            count = bin(P_bits & compat_bits[u]).count('1')
            if count > best_count:
                best_count = count
                best_u = u
        
        if best_u is None:
            return
        
        # Branch on P \ N(best_u) = P & ~compat_bits[best_u]
        branch_bits = P_bits & ~compat_bits[best_u]
        
        # Extract nodes to branch on
        branch_nodes = []
        temp = branch_bits
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            branch_nodes.append(idx + 1)
            temp ^= lsb
        
        for v in branch_nodes:
            if time.time() > bk_deadline:
                return
            
            # Add v to R
            R.append(v)
            R_set.add(v)
            
            # New P = P ∩ compat_bits[v] (excluding v itself)
            new_P_bits = P_bits & compat_bits[v]
            new_P_bits &= ~(1 << (v - 1))  # Remove v
            
            # New X = X ∩ compat_bits[v]
            new_X_bits = X_bits & compat_bits[v]
            
            bron_kerbosch_timed(R, new_P_bits, new_X_bits, R_set)
            
            R.pop()
            R_set.remove(v)
            
            # Move v from P to X
            P_bits &= ~(1 << (v - 1))
            X_bits |= (1 << (v - 1))
        
        # Restore P_bits and X_bits for caller (they were modified in loop)
        # Actually, we modified them in-place in the loop, so we need to restore
        # But since we're using local copies, this is fine for the recursive call
        # The caller's P_bits and X_bits are not affected

    # Initialize for BK
    initial_P_bits = 0
    for i in range(1, n + 1):
        initial_P_bits |= (1 << (i - 1))
    initial_X_bits = 0
    initial_R = []
    initial_R_set = set()
    
    bron_kerbosch_timed(initial_R, initial_P_bits, initial_X_bits, initial_R_set)
    
    # Greedy polish on the best result
    if best_clique[0]:
        polished = greedy_polish(best_clique[0])
        if len(polished) > best_size[0]:
            best_size[0] = len(polished)
            best_clique[0] = list(polished)
    
    # If BK didn't find anything good, try multiple seeds with greedy
    if best_size[0] < 15:
        seeds_to_try = []
        for base in [504, 480, 420, 360, 336, 288, 240, 210, 180, 168]:
            seeds_to_try.append([k for k in range(1, n + 1) if base % k == 0])
        
        for seed in seeds_to_try:
            if time.time() > deadline - 5.0:
                break
            polished = greedy_polish(seed)
            if len(polished) > best_size[0]:
                best_size[0] = len(polished)
                best_clique[0] = list(polished)
    
    # Final greedy polish
    if best_clique[0]:
        final = greedy_polish(best_clique[0])
        if len(final) > best_size[0]:
            best_size[0] = len(final)
            best_clique[0] = list(final)
    
    return best_clique[0] if best_clique[0] else [1]
