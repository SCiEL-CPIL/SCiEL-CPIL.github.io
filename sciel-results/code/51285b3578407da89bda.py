"""Combined program: maximum-clique search on the gcd-compatibility graph."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s wall-clock kill

    # Candidates: numbers in [1,255] sharing a factor with 255 = 3*5*17
    candidates = [i for i in range(1, 256)
                  if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_255 = candidates.index(255)

    # Compatibility: compat[i][j] = True iff gcd(candidates[i], candidates[j]) > 1
    # We will store neighbors as bitmasks for fast set operations
    neighbors = [0] * n
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            if math.gcd(ci, candidates[j]) > 1:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
        # Self-loop not needed for clique logic usually, but let's be safe or just rely on P/X logic
        # Actually, standard BK uses P = P \ N(v). If v is in R, it's removed from P.
        # The condition is gcd(a,b)>1 for distinct pairs.
        # So we don't need self-loops in neighbors for the intersection logic, 
        # because we remove v from P before recursing, and we only intersect with neighbors of v.
        # Wait, the standard algorithm:
        # bronk(R, P, X):
        #   if P and X are empty: update best
        #   for v in P \ N(u):
        #       bronk(R U {v}, P \ N(v), X \ N(v))
        #       P = P \ {v}
        #       X = X U {v}
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    # Greedy initialization to find a good lower bound quickly
    def greedy_clique():
        # Start with 255
        R = {idx_255}
        P = {i for i in range(n) if i != idx_255 and (neighbors[idx_255] >> i) & 1}
        
        while P:
            # Pick a node in P with max degree in P
            best_v = -1
            max_deg = -1
            for v in P:
                deg = popcount(neighbors[v] & (sum(1 << u for u in P) if P else 0))
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v == -1:
                break
            R.add(best_v)
            # Update P to be neighbors of best_v
            P = {u for u in P if u != best_v and (neighbors[best_v] >> u) & 1}
        return R

    initial_clique = greedy_clique()
    best_set = set(initial_clique)
    best_size = len(best_set)

    # Bron-Kerbosch with bitmasks
    def bronk(R, P_mask, X_mask):
        nonlocal best_set, best_size
        
        # Pruning: even if we add all of P, we can't beat best
        if len(R) + popcount(P_mask) <= best_size:
            return
            
        if not P_mask and not X_mask:
            if len(R) > best_size:
                best_size = len(R)
                # Reconstruct set from R (which is a set of indices)
                best_set = set(R)
            return

        # Pivot selection: pick u in P U X maximizing |P ∩ N(u)|
        # We can iterate over bits in P_mask | X_mask
        P_U_X = P_mask | X_mask
        u = -1
        maxc = -1
        temp = P_U_X
        while temp:
            # Extract lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            
            c = popcount(P_mask & neighbors[idx])
            if c > maxc:
                maxc = c
                u = idx
        
        # Iterate over v in P \ N(u)
        # If u is -1, P \ N(u) is P. Otherwise P & ~N(u)
        if u == -1:
            candidates_for_v = P_mask
        else:
            candidates_for_v = P_mask & ~neighbors[u]
            
        temp = candidates_for_v
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # Recurse
            new_R = R | {v}
            new_P = P_mask & neighbors[v]
            new_X = X_mask & neighbors[v]
            
            bronk(new_R, new_P, new_X)
            
            # Update P and X for the next iteration
            P_mask &= ~lsb
            X_mask |= lsb

    # Initial call: R={255}, P=Neighbors(255), X=Empty
    P_init = 0
    for i in range(n):
        if i != idx_255 and (neighbors[idx_255] >> i) & 1:
            P_init |= (1 << i)
    
    bronk({idx_255}, P_init, 0)

    # Convert best_set (indices) to values
    # best_set is already a set of indices.
    # We need to return sorted values.
    
    # Polish pass: try to add any compatible element, then swaps
    # We work with indices for speed, then convert at the end.
    # best_set is currently indices.
    
    def try_add(S):
        # S is a set of indices
        for i in range(n):
            if i in S:
                continue
            # Check if i is compatible with all in S
            # We can use bitmask for S? S is a set, let's convert to mask for speed if S is large
            # Or just loop. n is small (~100).
            ok = True
            for y in S:
                if not ((neighbors[i] >> y) & 1):
                    ok = False
                    break
            if ok:
                S.add(i)
                return True
        return False

    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False
        if try_add(best_set):
            improved = True
            continue
        # remove-one-add-many
        for rem in list(best_set):
            if rem == idx_255:
                continue
            S_t = best_set - {rem}
            base = len(S_t)
            for add in range(n):
                if add in S_t:
                    continue
                ok = True
                for y in S_t:
                    if not ((neighbors[add] >> y) & 1):
                        ok = False
                        break
                if ok:
                    S_t.add(add)
            if len(S_t) > len(best_set):
                best_set = S_t
                improved = True
                break

    # Random-restart local search to explore mixed structures
    rng = random.Random(12345)
    for _ in range(2000):
        if time.time() - start_time > deadline - 3:
            break
        core = {idx_255}
        others = [i for i in range(n) if i != idx_255]
        rng.shuffle(others)
        for i in others[:rng.randint(0, 4)]:
            # Check compatibility
            ok = True
            for y in core:
                if not ((neighbors[i] >> y) & 1):
                    ok = False
                    break
            if ok:
                core.add(i)
        S = set(core)
        # greedy expand
        rem = [i for i in range(n) if i not in S]
        rng.shuffle(rem)
        for i in rem:
            ok = True
            for y in S:
                if not ((neighbors[i] >> y) & 1):
                    ok = False
                    break
            if ok:
                S.add(i)
        # local improvement
        imp = True
        while imp and (time.time() - start_time < deadline - 3):
            imp = False
            if try_add(S):
                imp = True
                continue
            for rem_el in list(S):
                if rem_el == idx_255:
                    continue
                S_t = S - {rem_el}
                for add in range(n):
                    if add in S_t:
                        continue
                    ok = True
                    for y in S_t:
                        if not ((neighbors[add] >> y) & 1):
                            ok = False
                            break
                    if ok:
                        S_t.add(add)
                if len(S_t) > len(S):
                    S = S_t
                    imp = True
                    break
        if len(S) > len(best_set):
            best_set = S

    return sorted(candidates[i] for i in best_set)

# EVOLVE-BLOCK-END
