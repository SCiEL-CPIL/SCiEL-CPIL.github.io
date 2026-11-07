"""Optimized solver for packing distinct multiples of a fixed integer set."""
import time
from math import gcd
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // gcd(a, b)

# Precompute LCM of all divisors
L = DIVISORS[0]
for d in DIVISORS[1:]:
    L = _lcm(L, d)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Compute LCM of all divisors for periodicity
    lcm_all = L
    
    # We need to find a window [L, L+W] of minimum width W
    # such that for each divisor d_i, there exists a multiple of d_i
    # in [L, L+W], and all chosen multiples are distinct.
    
    # Strategy: Use bipartite matching for feasibility check at each window position.
    # For a given window start S and width W, each divisor d_i can take values
    # {m * d_i : m * d_i in [S, S+W]}. We need a system of distinct representatives.
    
    # To exploit periodicity: the pattern of available multiples repeats every lcm_all.
    # So we only need to check S in [0, lcm_all) and for each S, find min W.
    
    # But lcm_all might be huge. Let's compute it.
    # 6,10,14,15,21,22,26,33,35,39
    # LCM(6,10) = 30
    # LCM(30,14) = 210
    # LCM(210,15) = 210
    # LCM(210,21) = 210
    # LCM(210,22) = 2310
    # LCM(2310,26) = 2310*26/gcd(2310,26) = 2310*26/26 = 2310... wait
    # gcd(2310, 26): 2310 = 2*3*5*7*11, 26 = 2*13. gcd = 2. LCM = 2310*26/2 = 30030
    # LCM(30030, 33): 33=3*11. 30030=2*3*5*7*11*13. gcd=33. LCM=30030
    # LCM(30030, 35): 35=5*7. gcd=35. LCM=30030
    # LCM(30030, 39): 39=3*13. gcd=39. LCM=30030
    
    # So LCM = 30030. That's manageable!
    
    # For each residue class S in [0, 30030), find the minimum W such that
    # a valid assignment exists in window [S, S+W].
    
    # For a fixed S and divisor d_i, the available multiples in [S, S+W] are:
    # floor(S/d_i) * d_i, (floor(S/d_i)+1)*d_i, ... up to S+W
    # i.e., multiples k*d_i where k*d_i >= S and k*d_i <= S+W
    
    # For matching: we have N divisors on left, and for each, a set of candidate values.
    # Two divisors conflict if they can only use the same value (or we need distinct values).
    
    # Actually, for minimum W, we can think of it as: for each divisor d_i, 
    # the "next" multiple after S is ceil(S/d_i)*d_i. Let's call this next_i(S).
    # If we set W_i = next_i(S) - S, then all next_i(S) are in [S, S + max(W_i)].
    # But they might not be distinct!
    
    # So the basic W is max_i (next_i(S) - S), but we need distinctness.
    # If two divisors have the same next multiple, one must take the next one.
    
    # This is a classic problem. Let me use a greedy/matching approach.
    
    # For a given S, let's compute for each divisor d_i the sorted list of multiples
    # starting from the first >= S. We want to find min W such that there's a matching.
    
    # Since N=10 is small, we can do backtracking or use Hopcroft-Karp on a small graph.
    
    # For a given S and W, build bipartite graph: left = divisors, right = candidate values.
    # Edge (i, v) if v is a multiple of d_i and S <= v <= S+W.
    # Check if perfect matching exists.
    
    # To find min W for a given S: binary search on W? Or incrementally increase W?
    # Since W is at most ~max(d_i) in the worst case per period, and we have 30030 starts,
    # we need this to be fast.
    
    # Alternative: for a given S, the minimum W can be found by considering the 
    # "conflict" structure. Let me try a different approach.
    
    # For each S in [0, 30030):
    #   Compute next_i = ceil(S / d_i) * d_i for each i
    #   Sort divisors by next_i
    #   Try to assign distinct values greedily, and track the max assigned value
    #   This gives a feasible W, but is it optimal?
    
    # Actually, let me just do a clean binary search + matching for each S.
    
    # But 30030 * log(10000) * matching_cost might be too slow in Python.
    
    # Let me think about the matching cost. N=10 divisors. For a given W, each divisor
    # has at most W/d_i + 1 candidates. For small W, this is small.
    
    # Let me try: for each S, do a DFS/backtracking to find if a valid assignment exists
    # with max value <= S + W. Start with W = max(next_i - S) and increase if needed.
    
    # Actually, let me use a simpler approach: for each S, compute the minimal W via
    # a backtracking search that tries to assign values in order of increasing next_i.
    
    best_span = MAX_VAL
    best_solution = None
    
    for S in range(lcm_all):
        if time.time() > deadline:
            break
        
        # Compute next multiple for each divisor >= S
        nexts = []
        for i, d in enumerate(DIVISORS):
            if S % d == 0:
                nxt = S
            else:
                nxt = ((S // d) + 1) * d
            nexts.append(nxt)
        
        # Sort by next value, with index
        order = sorted(range(N), key=lambda i: nexts[i])
        
        # Backtracking: assign values to divisors in order
        # We want to minimize max assigned value
        used = set()
        best_local = nexts[order[-1]]  # worst case upper bound
        
        def backtrack(pos, current_max):
            nonlocal best_local, best_solution
            if current_max >= best_local:
                return False
            if pos == N:
                best_local = current_max
                # Store solution
                sol = [0] * N
                # We need to reconstruct, so let's pass assignments
                return True
            
            i = order[pos]
            d = DIVISORS[i]
            start_val = nexts[i]
            # Try values: start_val, start_val + d, start_val + 2*d, ...
            # But we can limit to start_val + d * (some small number)
            # Since we're minimizing, try in order
            
            # Upper bound on how far we need to go
            # We need current_max to stay < best_local
            # So max value we can try is best_local - 1
            max_try = best_local - 1
            
            k = 0
            while True:
                val = start_val + k * d
                if val > max_try:
                    break
                if val not in used:
                    used.add(val)
                    new_max = max(current_max, val)
                    if backtrack(pos + 1, new_max):
                        return True
                    used.remove(val)
                k += 1
                if time.time() > deadline:
                    return False
            
            return False
        
        # Need to track the actual assignment
        assignment = [0] * N
        
        def backtrack2(pos, current_max):
            nonlocal best_local
            if current_max >= best_local:
                return False
            if pos == N:
                best_local = current_max
                return True
            
            i = order[pos]
            d = DIVISORS[i]
            start_val = nexts[i]
            max_try = best_local - 1
            
            k = 0
            while True:
                val = start_val + k * d
                if val > max_try:
                    break
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    new_max = max(current_max, val)
                    if backtrack2(pos + 1, new_max):
                        return True
                    used.remove(val)
                    assignment[i] = 0
                k += 1
                if time.time() > deadline:
                    return False
            
            return False
        
        # Initialize
        best_local = max(nexts)  # initial upper bound
        used = set()
        
        if backtrack2(0, 0):
            span = best_local - S
            if span < best_span:
                best_span = span
                best_solution = [int(assignment[i]) for i in range(N)]
    
    # Fallback: if no solution found or best is still MAX_VAL
    if best_solution is None or best_span >= MAX_VAL:
        # Use simple multiples
        best_solution = [d * 1 for d in DIVISORS]
        # Ensure distinct
        seen = set()
        for i in range(N):
            while best_solution[i] in seen:
                best_solution[i] += DIVISORS[i]
            seen.add(best_solution[i])
        best_span = max(best_solution) - min(best_solution)
    
    # Ensure all values are positive and <= 1000000
    for i in range(N):
        if best_solution[i] <= 0:
            best_solution[i] = DIVISORS[i]
        if best_solution[i] > MAX_VAL:
            best_solution[i] = MAX_VAL - (MAX_VAL % DIVISORS[i])
    
    # Ensure distinctness as final check
    seen = set()
    for i in range(N):
        v = best_solution[i]
        while v in seen:
            v += DIVISORS[i]
        best_solution[i] = v
        seen.add(v)
    
    return [int(x) for x in best_solution]
