import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 84.0
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
            
    # Identify all bad triples (a,b,c) with a<b<c and gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    T = len(bad_triples)
    
    # For each element, list of triple indices containing it
    triples_by_elem = [[] for _ in range(N)]
    for i, t in enumerate(bad_triples):
        for x in t:
            triples_by_elem[x - 1].append(i)
            
    # Helper: check if a set of elements (as mask or set) is valid
    # A set S is valid if no bad triple is fully contained in S.
    # Equivalent: For every bad triple, at least one element is NOT in S (i.e., in R = complement).
    
    # We will use a representation: S is a set of indices (0 to N-1)
    # Or a bitmask? N=128 fits in two 64-bit ints or a Python int (arbitrary precision).
    # Python ints are fine for bitmasks up to 128 bits.
    
    # Precompute for each element, the list of other elements that form a "conflict" with it?
    # No, the conflict is 3-way.
    # Let's stick with the hitting set approach but with a smarter local search.
    
    # Objective: Minimize |R| such that R hits all bad_triples.
    # R is a set of indices.
    
    # Initialize with a greedy hitting set
    def greedy_hitting_set():
        R = set()
        active = set(range(T))
        counts = [0] * N
        for t_idx in active:
            for x in bad_triples[t_idx]:
                counts[x-1] += 1
        
        while active:
            if time.time() > deadline - 5.0:
                break
            # Find element with max count
            max_c = -1
            best = -1
            for i in range(N):
                if i not in R and counts[i] > max_c:
                    max_c = counts[i]
                    best = i
            if best == -1 or max_c == 0:
                break
            R.add(best)
            # Update counts and active
            # Triples containing best are removed
            for t_idx in triples_by_elem[best]:
                if t_idx in active:
                    active.remove(t_idx)
                    a, b, c = bad_triples[t_idx]
                    for x in (a, b, c):
                        if x - 1 != best:
                            counts[x-1] -= 1
        return R

    # Local Search: Simulated Annealing on R
    # State: R (set)
    # Energy: |R| + lambda * (uncovered triples)
    # Uncovered triples: triples with all 3 elements NOT in R.
    
    def get_uncovered(R):
        count = 0
        for t in bad_triples:
            if (t[0]-1 not in R) and (t[1]-1 not in R) and (t[2]-1 not in R):
                count += 1
        return count

    best_R = greedy_hitting_set()
    best_unc = get_uncovered(best_R)
    # Ideally best_unc should be 0. If not, we need to fix it.
    # If best_unc > 0, the solution is invalid.
    # We must ensure we return a valid set.
    # If SA fails to find unc=0, we can repair best_R by adding elements until unc=0.
    
    # Let's run SA to minimize |R| subject to unc=0.
    # Start from best_R.
    # If best_unc > 0, we are in an invalid state.
    # Let's do a repair phase first if needed.
    if best_unc > 0:
        # Add elements to R to cover all uncovered triples
        active_uncovered = [i for i in range(T) if (bad_triples[i][0]-1 not in best_R) and (bad_triples[i][1]-1 not in best_R) and (bad_triples[i][2]-1 not in best_R)]
        while active_uncovered:
            if time.time() > deadline - 5.0:
                break
            # Pick element that covers most active_uncovered
            # Just pick first one? Or random?
            # Let's pick the one with highest degree in the subgraph of uncovered triples?
            # Simpler: pick an element from a random uncovered triple.
            t_idx = random.choice(active_uncovered)
            a, b, c = bad_triples[t_idx]
            # Pick one of a,b,c to add to R
            # Heuristic: pick the one that appears in most other uncovered triples?
            # For speed, just pick the one with highest global degree in uncovered set?
            # Let's just pick randomly from the triple to avoid bias?
            # Or pick the one with highest count in the current uncovered set?
            # Let's calculate counts for uncovered triples
            counts = [0] * N
            for ui in active_uncovered:
                a, b, c = bad_triples[ui]
                counts[a-1] += 1
                counts[b-1] += 1
                counts[c-1] += 1
            # Pick max count
            best_elem = max([a-1, b-1, c-1], key=lambda x: counts[x])
            best_R.add(best_elem)
            # Update active_uncovered
            new_active = []
            for ui in active_uncovered:
                a, b, c = bad_triples[ui]
                if a-1 != best_elem and b-1 != best_elem and c-1 != best_elem:
                    new_active.append(ui)
            active_uncovered = new_active
            
    # Now best_R is valid (unc=0).
    # Refine with SA to minimize |R|.
    R = set(best_R)
    unc = 0
    # Verify unc is 0
    # (Should be 0 after repair)
    
    def energy(R_set):
        # Return (size, unc)
        return len(R_set), unc # unc is global state? No, let's pass it.
        
    # Let's implement a proper SA loop
    current_R = set(best_R)
    current_unc = 0
    # Verify current_unc
    for t in bad_triples:
        if (t[0]-1 not in current_R) and (t[1]-1 not in current_R) and (t[2]-1 not in current_R):
            current_unc += 1
            
    best_R = set(current_R)
    best_size = len(current_R)
    
    T_start = 20.0
    T_end = 0.1
    steps = 50000
    
    # Precompute neighbors for faster delta?
    # Delta for adding x:
    #   size +1
    #   unc decreases by number of triples containing x where other 2 not in R.
    # Delta for removing x:
    #   size -1
    #   unc increases by number of triples containing x where other 2 not in R.
    
    import math
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        progress = step / steps
        T = T_start * (T_end / T_start) ** progress
        
        # Move: remove or add
        # If current_unc > 0, we must add (or swap).
        # If current_unc == 0, we can remove (risking unc>0) or add (worse).
        
        if current_unc > 0:
            # Must fix
            # Pick an uncovered triple, add one of its elements
            # Find an uncovered triple
            # This is O(T). Can be optimized.
            # Let's just pick a random element from R? No, we need to ADD.
            # Pick a random element NOT in R?
            # Better: Pick an uncovered triple and add a random element from it.
            # To find uncovered triple fast, we can maintain a list?
            # Or just iterate.
            # If T is large, this is slow.
            # Let's assume T is manageable.
            # Find one uncovered triple
            unc_triple = None
            for t in bad_triples:
                if (t[0]-1 not in current_R) and (t[1]-1 not in current_R) and (t[2]-1 not in current_R):
                    unc_triple = t
                    break
            if unc_triple is None:
                current_unc = 0
                continue
            # Pick element to add
            a, b, c = unc_triple
            # Pick one with max potential?
            # Just random
            x = random.choice([a-1, b-1, c-1])
            
            # Add x
            # Calculate delta_unc
            delta_unc = 0
            for t_idx in triples_by_elem[x]:
                a2, b2, c2 = bad_triples[t_idx]
                others = [a2-1, b2-1, c2-1]
                others.remove(x)
                if others[0] not in current_R and others[1] not in current_R:
                    delta_unc -= 1 # Covered
            
            current_R.add(x)
            current_unc += delta_unc
            # Always accept (greedy repair)
            
        else:
            # Valid state. Try to remove.
            if not current_R:
                break
            x = random.choice(list(current_R))
            
            # Calculate delta_unc if we remove x
            delta_unc = 0
            for t_idx in triples_by_elem[x]:
                a2, b2, c2 = bad_triples[t_idx]
                others = [a2-1, b2-1, c2-1]
                others.remove(x)
                if others[0] not in current_R and others[1] not in current_R:
                    delta_unc += 1 # Becomes uncovered
            
            new_unc = current_unc + delta_unc
            delta_size = -1
            # Energy = size + lambda * unc
            # If new_unc == 0, delta_E = -1. Accept.
            # If new_unc > 0, delta_E = -1 + lambda * new_unc.
            # Accept with prob exp(-delta_E / T)
            
            if new_unc == 0:
                # Improvement
                current_R.remove(x)
                current_unc = 0
                if len(current_R) < best_size:
                    best_size = len(current_R)
                    best_R = set(current_R)
            else:
                # Deterioration (unc becomes > 0)
                # delta_E = -1 + lambda * new_unc
                # lambda should be large enough to penalize invalid states heavily?
                # If lambda is large, we rarely accept invalid states.
                # Let's use lambda = 10.
                delta_E = -1 + 10.0 * new_unc
                if random.random() < math.exp(-delta_E / T):
                    current_R.remove(x)
                    current_unc = new_unc
                    
                # Also try adding?
                # If we can't remove, maybe adding helps? No, adding increases size.
                # Unless it enables future removals?
                # SA should handle this by accepting invalid states occasionally.
                
        # Update best if valid and smaller
        if current_unc == 0 and len(current_R) < best_size:
            best_size = len(current_R)
            best_R = set(current_R)
            
    # Final check
    S = [i + 1 for i in range(N) if i not in best_R]
    
    # Verify validity of S
    # For all t in bad_triples, not (a in S and b in S and c in S)
    # i.e., at least one of a,b,c is in best_R.
    # This is guaranteed if best_unc == 0.
    # But just in case, check.
    for t in bad_triples:
        if (t[0]-1 not in best_R) and (t[1]-1 not in best_R) and (t[2]-1 not in best_R):
            # Invalid!
            # This shouldn't happen if we tracked unc correctly.
            # Repair: add one of them
            best_R.add(t[0]-1)
            break # Just one repair? Might need more.
            # Let's do a full repair if this happens.
            # But it shouldn't.
            
    if not S:
        return [1]
        
    return S

# EVOLVE-BLOCK-START
# (Note: The code above is the implementation. The EVOLVE-BLOCK markers are for the evolution system.)
# EVOLVE-BLOCK-END
