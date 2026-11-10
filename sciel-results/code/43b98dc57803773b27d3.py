import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 512

    # Precompute lcm matrix and compatibility
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

    # Precompute popcount for all possible bit patterns we might encounter
    # Actually, Python's int.bit_count() is available and fast
    # Let's use a helper that uses bit_count if available, else bin count
    try:
        # Test if bit_count is available (Python 3.10+)
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    # Compute degree of each node for greedy initialization
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = popcount(compat_bits[i])

    # Greedy initialization to get a good lower bound
    # Sort nodes by degree descending
    sorted_nodes = sorted(range(1, n + 1), key=lambda x: -degrees[x])
    
    greedy_clique = []
    greedy_set = set()
    for v in sorted_nodes:
        if all((1 << (u - 1)) & compat_bits[v] for u in greedy_set):
            greedy_clique.append(v)
            greedy_set.add(v)
    
    best_clique = list(greedy_clique)
    best_size = len(greedy_clique)

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
        # We want to maximize |P ∩ N(u)|
        PX = P | X
        
        # Optimization: Use a heuristic to avoid iterating all PX nodes
        # Try a few candidates first, then if needed, do full scan
        # For now, let's optimize the full scan by breaking early if we find a perfect pivot
        
        best_u = None
        best_count = -1
        
        # Fast path: if P is small, just pick the first node in P
        if popcount(P) <= 3:
            lsb = P & (-P)
            idx = lsb.bit_length() - 1
            best_u = idx + 1
            best_count = popcount(P & compat_bits[best_u])
        else:
            # Sample some nodes from PX to find a good pivot quickly
            # Try up to 10 random samples, then do full scan if not good enough
            temp = PX
            # We'll do a limited scan first
            count_checked = 0
            while temp and count_checked < 20:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                u = idx + 1
                count = popcount(P & compat_bits[u])
                if count > best_count:
                    best_count = count
                    best_u = u
                # If we found a perfect pivot, we can stop
                if best_count == popcount(P):
                    break
                temp ^= lsb
                count_checked += 1
            
            # If we didn't find a perfect pivot and still have time, do full scan
            if best_count < popcount(P) and time.time() < deadline:
                temp = PX
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    u = idx + 1
                    count = popcount(P & compat_bits[u])
                    if count > best_count:
                        best_count = count
                        best_u = u
                        if best_count == popcount(P):
                            break
                    temp ^= lsb

        if best_u is None:
            return

        # Branch on P \ N(best_u)
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
    initial_P = 0
    for i in range(1, n + 1):
        initial_P |= (1 << (i - 1))
    initial_X = 0
    initial_R = []

    bron_kerbosch(initial_R, initial_P, initial_X)

    # If BK didn't finish or found nothing (unlikely), fallback to greedy
    if best_size == 0:
        S = set()
        for i in range(1, n + 1):
            if all((i in compat_bits[j] or j in compat_bits[i]) for j in S):
                S.add(i)
        best_clique = list(S)
        best_size = len(S)

    # Final polish: try to add any remaining compatible nodes to the found clique
    S = set(best_clique)
    changed = True
    while changed:
        changed = False
        for i in range(1, n + 1):
            if i not in S:
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
