import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    
    # Precompute compatibility bitmask for each vertex
    # compat_mask[i] for vertex i (1-indexed), bit (j-1) set if lcm(i,j) <= N
    compat_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                mask |= (1 << (j - 1))
        compat_mask[i] = mask

    def time_up():
        return time.time() > deadline

    # Strategy 1: Divisors of a highly composite number M <= 512
    # If S is a set of divisors of M, then lcm(a,b) divides M, so lcm(a,b) <= M <= 512.
    # We want to maximize |S|. This is equivalent to finding M <= 512 with max number of divisors.
    best_divisor_set = []
    max_divs = 0
    for M in range(1, N + 1):
        # Count divisors of M
        divs = 0
        for d in range(1, int(M**0.5) + 1):
            if M % d == 0:
                divs += 2
        if M == int(M**0.5) ** 2:
            divs -= 1
        if divs > max_divs:
            max_divs = divs
            # Reconstruct divisors
            divisors = []
            for d in range(1, int(M**0.5) + 1):
                if M % d == 0:
                    divisors.append(d)
                    if d != M // d:
                        divisors.append(M // d)
            best_divisor_set = divisors
            
    best_clique = best_divisor_set

    # Strategy 2: Greedy + Local Search
    # Compute degrees
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(compat_mask[i]).count('1')
    
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(1, N + 1), key=lambda v: -degrees[v])
    
    def greedy_clique(order):
        clique = []
        for v in order:
            if time_up():
                break
            compatible = True
            for u in clique:
                if not (compat_mask[v] & (1 << (u - 1))):
                    compatible = False
                    break
            if compatible:
                clique.append(v)
        return clique

    # Try a few greedy orders
    orders = [
        sorted_vertices,
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda v: -degrees[v] / v)
    ]
    
    for order in orders:
        if time_up():
            break
        g_clique = greedy_clique(order)
        if len(g_clique) > len(best_clique):
            best_clique = g_clique

    # Local Search: Try to improve best_clique
    # We represent the current set as a list.
    # We try to add a vertex v not in the set.
    # If it conflicts with existing vertices, we try to remove one conflicting vertex and see if the size increases (it won't if we remove 1 and add 1, but maybe we can add 1 and remove 0? No, if it conflicts, we must remove).
    # Actually, standard local search for Max Clique:
    # 1. Try to add any vertex that is compatible with all. (Already done in greedy if we process all).
    # 2. Try swapping: remove u from clique, add v. If v is compatible with all others (clique - {u}), we have a new clique of same size.
    # 3. Try adding v and removing multiple u's?
    
    # A simpler improvement: 
    # Given current clique C, check all v not in C.
    # If v is compatible with all C, add it.
    # If not, let Conflicts = {u in C | v not compatible with u}.
    # If we can remove a subset R of Conflicts such that |R| < 1 (impossible) or we just do single swaps to escape local minima?
    # Single swaps don't increase size.
    
    # Let's try a "Grow and Shrink" heuristic.
    # Actually, for N=512, the divisor set (size ~48) is likely very close to optimal or optimal.
    # Let's verify if we can beat 48.
    
    # Let's do a simple local search: 
    # Iterate through all pairs (v_out, v_in) where v_out in C, v_in not in C.
    # If v_in is compatible with C \ {v_out}, then C' = (C \ {v_out}) U {v_in} is a valid clique of same size.
    # This doesn't help maximize size directly, but can move us to a better local minimum.
    # To increase size, we need to find a v_in compatible with C.
    
    # Since Greedy already tries to add compatible vertices, the resulting clique is maximal (no single vertex can be added).
    # So we are stuck at a local maximum for size.
    # To find a larger clique, we need to move to a different local maximum.
    # We can do this by random restarts of Greedy with random perturbations.
    
    import random
    random.seed(42)
    
    for restart in range(50):
        if time_up():
            break
        
        # Perturb the order
        perturbed_order = list(sorted_vertices)
        # Swap a few elements
        for _ in range(10):
            i = random.randint(0, N-2)
            j = random.randint(i+1, N-1)
            perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
        g_clique = greedy_clique(perturbed_order)
        if len(g_clique) > len(best_clique):
            best_clique = g_clique

    # Final check: ensure best_clique is valid
    # (It should be, but just in case)
    # And return sorted
    
    return sorted(best_clique)
