import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    n = 512

    # Precompute lcm matrix and compatibility
    # compat[i] is a bitset where bit j-1 is set if lcm(i,j) <= 512
    # We only care about i != j for conflicts, but including self doesn't hurt for the algorithm
    # Actually, for maximum clique on compatibility graph, we want edges between i and j if lcm(i,j) <= 512
    # Self-loops don't matter.

    # Precompute gcd and lcm for all pairs to build compat bits
    # 512 nodes, so 512x512 matrix of lcm.
    # lcm(i,j) = i * j // gcd(i,j)
    
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

    # We are looking for a maximum clique in the graph defined by compat_bits
    # Vertices: 1..512
    # Edge (i,j) exists if i is in compat_bits[j] (and vice versa, symmetric)
    
    # Bron-Kerbosch with pivot, using bitsets
    # R: list of nodes in current clique
    # P: bitset of candidates
    # X: bitset of excluded candidates
    
    best_clique = []
    best_size = 0

    def popcount(x):
        return bin(x).count('1')

    def bron_kerbosch(R, P, X, depth=0):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return

        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return

        if P == 0:
            return

        # Pruning: if |R| + |P| <= best_size, no improvement possible
        if len(R) + popcount(P) <= best_size:
            return

        # Choose pivot u from P ∪ X to minimize branching
        # We want to branch on P \ N(u)
        # N(u) in this graph is compat_bits[u]
        # So we want to maximize |P ∩ N(u)| to minimize |P \ N(u)|
        
        PX = P | X
        # To find pivot, we can iterate through nodes in PX
        # But iterating through 512 nodes every time is expensive.
        # Let's try a few heuristics or just pick the first one?
        # Or compute max properly.
        
        # Optimization: Instead of iterating all PX nodes, we can just pick one
        # but that is less optimal. Let's iterate.
        # 512 is small enough?
        
        best_u = None
        best_count = -1
        
        # Extract nodes from PX
        temp = PX
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            u = idx + 1
            # Count |P ∩ N(u)|
            # N(u) is compat_bits[u]
            count = popcount(P & compat_bits[u])
            if count > best_count:
                best_count = count
                best_u = u
            temp ^= lsb

        if best_u is None:
            return

        # Branch on P \ N(best_u)
        # Branch candidates are in P but not in N(best_u)
        # i.e., P & ~compat_bits[best_u]
        
        branch_candidates = P & ~compat_bits[best_u]
        
        temp = branch_candidates
        while temp:
            if time.time() > deadline:
                return
            
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            v = idx + 1
            
            # Add v to R
            R.append(v)
            
            # New P = P ∩ N(v)
            # N(v) is compat_bits[v]
            new_P = P & compat_bits[v]
            
            # New X = X ∩ N(v)
            new_X = X & compat_bits[v]
            
            bron_kerbosch(R, new_P, new_X, depth + 1)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << (v - 1))
            X |= (1 << (v - 1))
            
            temp ^= lsb

    # Initial call
    # P = all nodes 1..512
    initial_P = 0
    for i in range(1, n + 1):
        initial_P |= (1 << (i - 1))
    initial_X = 0
    initial_R = []

    bron_kerbosch(initial_R, initial_P, initial_X)

    # If BK didn't finish or found nothing (unlikely), fallback to greedy
    if best_size == 0:
        # Greedy fallback
        S = set()
        # Pick node with max degree first?
        # Or just iterate
        for i in range(1, n + 1):
            if all((i in compat_bits[j] or j in compat_bits[i]) for j in S):
                S.add(i)
        best_clique = list(S)
        best_size = len(S)

    # Final polish: try to add any remaining compatible nodes to the found clique
    # This handles cases where BK might have stopped early or if the clique found
    # is not maximal (though BK with pivot and full expansion should find maximal cliques)
    # However, if we return early due to time limit, the current R might not be maximal.
    # But `best_clique` is only updated when P=0 and X=0, so it is a maximal clique.
    # Wait, if time limit is hit, we return without updating best_clique if it's not a leaf.
    # So best_clique is always a valid maximal clique from a completed branch.
    
    # Let's do a greedy polish on best_clique just in case, to ensure it's as large as possible locally.
    # Actually, a maximal clique is already locally maximal.
    # But maybe we can swap nodes? That's local search, too complex.
    # Greedy polish: check if any node outside can be added.
    S = set(best_clique)
    changed = True
    while changed:
        changed = False
        for i in range(1, n + 1):
            if i not in S:
                # Check if i is compatible with all in S
                # i in compat_bits[j] for all j in S
                ok = True
                for j in S:
                    if not (compat_bits[i] & (1 << (j - 1))):
                        ok = False
                        break
                if ok:
                    S.add(i)
                    changed = True
    if len(S) > best_size:
        best_clique = list(S)
        best_size = len(S)

    return best_clique
