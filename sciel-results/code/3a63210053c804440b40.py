import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    
    # 1. Precompute GCD matrix and Forbidden Triples
    # We need to know for any pair (a, b), which c's are forbidden.
    # Forbidden condition: gcd(a,b) == gcd(a,c) == gcd(b,c)
    
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # forbidden[a][b] is a bitmask of c such that {a,b,c} is a forbidden triple.
    # Using integers as bitmasks for speed.
    forbidden = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Precompute for each pair (a, b) the mask of c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    mask |= (1 << c)
            forbidden[a][b] = mask
            forbidden[b][a] = mask

    # 2. Initial Solution Construction
    # Strategy: Start with all numbers, then greedily remove elements that cause the most conflicts.
    # A "conflict" is a forbidden triple.
    # Alternatively, build up a set.
    # Let's try a greedy "maximize independent set" heuristic.
    # Order elements by "centrality" in the conflict hypergraph?
    # Or simply: Start with S = [], add elements that are compatible.
    
    # Better initial: 
    # The condition is local to the gcd structure.
    # Numbers with distinct prime factors (square-free) are often safer?
    # Actually, let's just run a fast greedy on the full set.
    
    def greedy_initial():
        S = []
        S_mask = 0
        # Sort by some heuristic? Maybe just 1..N
        # To be robust, let's try multiple random orderings and pick the best
        best_S = []
        best_mask = 0
        
        # One deterministic pass
        for x in range(1, N + 1):
            # Check if x can be added to S
            # x conflicts if there exist a, b in S such that (a, b, x) is forbidden.
            # This is equivalent to: for all a in S, is x in forbidden[a][b] for some b in S?
            # Wait, forbidden[a][b] is a mask of c.
            # So if we have S, adding x is valid iff:
            # For all a, b in S, x is NOT in forbidden[a][b].
            # This check is O(|S|^2) if done naively.
            # Optimization: 
            # Maintain a "danger" mask for each potential x?
            # Or simply check pairs in S. Since S grows, this is O(N^3) total which is 128^3 ~ 2M ops. Fast enough.
            
            valid = True
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if (forbidden[a][b] >> x) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
                S_mask |= (1 << x)
        
        return S, S_mask

    S_init, mask_init = greedy_initial()
    
    # Let's try a few random restarts for the greedy to get a better starting point
    best_S = S_init
    best_mask = mask_init
    
    for _ in range(10):
        if time.time() > DEADLINE - 20: break
        random.shuffle(UNIVERSE := list(range(1, N + 1)))
        S_curr = []
        mask_curr = 0
        for x in UNIVERSE:
            valid = True
            for i in range(len(S_curr)):
                a = S_curr[i]
                for j in range(i + 1, len(S_curr)):
                    b = S_curr[j]
                    if (forbidden[a][b] >> x) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_curr.append(x)
                mask_curr |= (1 << x)
        
        if len(S_curr) > len(best_S):
            best_S = S_curr
            best_mask = mask_curr

    # 3. Simulated Annealing / Local Search
    # State: mask (integer representing subset)
    # Objective: Maximize popcount(mask) subject to no forbidden triples.
    # We will perform moves:
    # 1. Add x (if valid)
    # 2. Remove x
    # 3. Swap x (in) and y (out)
    
    # To make this fast, we need efficient validity checks.
    # Given a mask M, is it valid?
    # We can precompute for each pair (a, b) the forbidden mask.
    # A set M is invalid if there exist a, b in M such that (forbidden[a][b] & M) != 0 (excluding a, b themselves? 
    # Actually forbidden[a][b] contains c != a, b. So if M contains a, b and any c in forbidden[a][b], it's invalid.)
    
    # Efficient Move Evaluation:
    # When adding x to M:
    # x is valid if for all a, b in M, x is not in forbidden[a][b].
    # This is still O(|M|^2).
    
    # Alternative: Precompute "conflict masks" for each element x?
    # Let Conflict[x] be the set of pairs (a, b) such that x is in forbidden[a][b].
    # This is equivalent to: a, b are such that gcd(a,b)=gcd(a,x)=gcd(b,x).
    # Let's precompute for each x, a list of pairs (a, b) that conflict with x.
    # Actually, we can precompute for each x, a bitmask of "forbidden partners"? No, it's pairs.
    
    # Let's stick to the O(|M|^2) check but optimize it.
    # |M| is around 50-80. |M|^2 is ~6000.
    # 100,000 iterations * 6000 ops = 600M ops. Might be too slow in Python.
    
    # Optimization:
    # Precompute for each x, a list of "conflict pairs" (a, b).
    # conflict_pairs[x] = list of (a, b) such that x is forbidden with a and b.
    # To check if x can be added to M:
    # Check if any (a, b) in conflict_pairs[x] has both a and b in M.
    # This is O(len(conflict_pairs[x])).
    # How large is conflict_pairs[x]?
    # For x, the pairs (a, b) must satisfy gcd(a,b)=gcd(a,x)=gcd(b,x)=g.
    # This implies a, b are multiples of g, and a/g, b/g, x/g are pairwise coprime?
    # No, gcd(a/g, b/g) = 1, gcd(a/g, x/g) = 1, gcd(b/g, x/g) = 1.
    # So a/g, b/g, x/g are pairwise coprime.
    # The number of such pairs is bounded.
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            for b in range(a + 1, N + 1):
                if b == x: continue
                if (forbidden[a][b] >> x) & 1:
                    conflict_pairs[x].append((a, b))
                    
    # Now, check_add(M_mask, x):
    # for a, b in conflict_pairs[x]:
    #    if (M_mask >> a) & 1 and (M_mask >> b) & 1: return False
    # return True

    # This is much faster if conflict_pairs[x] is small.
    
    # Let's refine the SA loop.
    current_mask = best_mask
    current_S = [i for i in range(1, N+1) if (current_mask >> i) & 1]
    best_mask_sa = current_mask
    best_S_sa = current_S
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    # We need a fast popcount. Python 3.8+ has int.bit_count()
    
    while time.time() < DEADLINE - 5:
        if T < 0.01:
            # Local search phase: only accept improvements
            pass 
            
        # Generate a random move
        # 1. Try adding a random element not in M
        # 2. Try removing a random element in M
        # 3. Try swapping
        
        move_type = random.random()
        new_mask = current_mask
        
        if move_type < 0.4:
            # Add
            # Pick random x not in current_mask
            # We can precompute list of all elements and filter
            # Or pick random index
            while True:
                x = random.randint(1, N)
                if not (current_mask >> x) & 1:
                    break
            
            # Check validity
            valid = True
            for a, b in conflict_pairs[x]:
                if (current_mask >> a) & 1 and (current_mask >> b) & 1:
                    valid = False
                    break
            
            if valid:
                new_mask = current_mask | (1 << x)
            else:
                # Invalid move, skip or penalize? 
                # In SA, we usually only consider valid states for this constraint satisfaction problem,
                # or we use a penalty function.
                # Let's stick to valid-only moves (local search) for robustness, 
                # or use a soft penalty.
                # Given the "hard" constraint, valid-only is safer to guarantee output validity.
                # But valid-only can get stuck.
                # Let's try "Add if valid, else do nothing" for this iteration.
                continue
                
        elif move_type < 0.7:
            # Remove
            # Pick random x in current_mask
            # Find set bits
            # Efficient way: pick random position, check if set
            while True:
                x = random.randint(1, N)
                if (current_mask >> x) & 1:
                    break
            new_mask = current_mask & ~(1 << x)
            
        else:
            # Swap
            # Pick x not in M, y in M
            while True:
                x = random.randint(1, N)
                if not (current_mask >> x) & 1:
                    break
            while True:
                y = random.randint(1, N)
                if (current_mask >> y) & 1:
                    break
            
            # Check if adding x to (M - {y}) is valid
            temp_mask = current_mask & ~(1 << y)
            valid = True
            for a, b in conflict_pairs[x]:
                if (temp_mask >> a) & 1 and (temp_mask >> b) & 1:
                    valid = False
                    break
            
            if valid:
                new_mask = temp_mask | (1 << x)
            else:
                continue

        # Evaluate
        curr_val = current_mask.bit_count()
        new_val = new_mask.bit_count()
        
        # Since we only generate valid states (mostly), we accept if new_val >= curr_val
        # Or use SA logic if we allowed invalid states. 
        # Let's accept if new_val > curr_val, or (new_val == curr_val and random() < 0.5) to escape plateaus?
        # Actually, if new_val < curr_val, it's a "bad" move.
        
        if new_val > curr_val:
            current_mask = new_mask
            if new_val > best_mask_sa.bit_count():
                best_mask_sa = new_mask
                best_S_sa = [i for i in range(1, N+1) if (new_mask >> i) & 1]
        elif new_val == curr_val:
            # Accept sideways moves with some probability to explore
            if random.random() < 0.5:
                current_mask = new_mask
        else:
            # Reject worse moves (or accept with prob exp(-delta/T) if doing full SA on penalty)
            # Since we are constrained to valid sets, "worse" means smaller size.
            # We generally want to maximize size.
            pass
        
        T *= cooling
        
    # Final polish: Greedy add
    S_final = [i for i in range(1, N+1) if (best_mask_sa >> i) & 1]
    mask_final = best_mask_sa
    
    # Try to add any remaining elements
    improved = True
    while improved and time.time() < DEADLINE - 1:
        improved = False
        for x in range(1, N + 1):
            if (mask_final >> x) & 1:
                continue
            valid = True
            for a, b in conflict_pairs[x]:
                if (mask_final >> a) & 1 and (mask_final >> b) & 1:
                    valid = False
                    break
            if valid:
                mask_final |= (1 << x)
                improved = True
                # Restart loop to see if more can be added
                break
                
    # Verify final result
    S_result = [i for i in range(1, N+1) if (mask_final >> i) & 1]
    
    # Double check validity
    valid = True
    for i in range(len(S_result)):
        a = S_result[i]
        for j in range(i+1, len(S_result)):
            b = S_result[j]
            for k in range(j+1, len(S_result)):
                c = S_result[k]
                if (forbidden[a][b] >> c) & 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to initial if something went wrong
        return best_S
        
    return S_result

# Ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
