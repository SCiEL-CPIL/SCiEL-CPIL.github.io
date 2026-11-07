import time
from math import gcd
from functools import reduce
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Pre-compute multiples for each divisor
    # multiples[i] is a sorted list of multiples of divisors[i] up to max_val
    multiples = []
    for d in divisors:
        m = []
        x = d
        while x <= max_val:
            m.append(x)
            x += d
        multiples.append(m)
    
    # Helper: Check if a valid distinct assignment exists for candidates
    # candidates is a list of lists: candidates[i] contains multiples of divisors[i] in the window
    def check_assignment(candidates):
        # If any divisor has no candidates, fail
        for c in candidates:
            if not c:
                return None
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # If any has 0, already handled, but just in case
        if len(candidates[order[0]]) == 0:
            return None
            
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Event-driven sliding window for a fixed W
    # We iterate L by jumping to the next point where the set of multiples in [L, L+W] changes
    def find_solution_for_W(W):
        if W >= max_val:
            return None
            
        # We need to track which multiples are in the window for each divisor.
        # Instead of maintaining a dynamic set, we can just re-calculate candidates
        # for the new L using binary search on the pre-computed multiples lists.
        # This is O(N * log M) per step.
        # Number of steps is number of events.
        
        # To jump L, we need to know the next multiple that enters or leaves.
        # A multiple m is in [L, L+W] if L <= m <= L+W.
        # It leaves when L > m. Next L change: m+1.
        # It enters when L <= m and L+W >= m => L >= m-W. Next L change: m-W.
        
        # We can pre-compute all "events" for a given W? No, W changes.
        # But we can just iterate L by 1? No, that's 10^6 steps.
        # 10^6 steps * 10 divisors * log(10^6/d) binary searches.
        # 10^6 * 10 * 20 = 2 * 10^8 operations. Might be too slow in Python.
        
        # Better: Jump L.
        # For current L, find the next L' > L such that the set of candidates changes.
        # The set changes if:
        # 1. Some multiple m_k of d_k is such that L < m_k <= L+W (enters as L decreases? No, L increases).
        #    As L increases, m_k leaves the window when L > m_k.
        #    So the next leaving event is min(m_k + 1) for all m_k in current window?
        #    Actually, if m_k is in [L, L+W], it stays until L becomes m_k + 1.
        #    So next leave is min(m_k + 1) for all m_k in window.
        # 2. Some multiple m_k of d_k is such that L+W < m_k <= L+W + delta.
        #    It enters when L+W >= m_k => L >= m_k - W.
        #    So the next entering event is min(m_k - W) for all m_k > L+W.
        
        # Let's just iterate L from 1 to max_val - W, but skip if no change?
        # Simpler: Just iterate L from 1 to max_val - W.
        # If it's too slow, we optimize.
        
        # Optimization: Only check L where at least one divisor has a candidate change?
        # Actually, for small W, the number of candidates is very small.
        
        # Let's try iterating L from 1 to max_val - W.
        # To speed up, we can pre-compute for each L the candidates? No.
        
        # Let's try the jump method.
        L = 1
        # We need to know the current candidates to find the next jump.
        # This requires maintaining the state.
        
        # Alternative: Just iterate L.
        # 10^6 iterations.
        # Inside: 10 binary searches.
        # 10^7 binary searches.
        # Python can do ~10^7 simple ops/sec? Maybe 10^6.
        # 10^6 * 10 * log(10^6) ~ 10^6 * 10 * 20 = 2*10^8. Too slow.
        
        # We MUST jump.
        
        # Let's maintain pointers for each divisor into the multiples list.
        # ptr[i] is the index in multiples[i] of the first multiple >= L.
        # As L increases, ptr[i] only increases.
        
        ptrs = [0] * n
        # Initialize ptrs for L=1
        for i in range(n):
            # First multiple >= 1 is multiples[i][0]
            ptrs[i] = 0
            
        while L <= max_val - W:
            # Build candidates for current L
            candidates = [[] for _ in range(n)]
            for i in range(n):
                d = divisors[i]
                # Find first index in multiples[i] >= L
                # We can maintain ptrs[i]
                # While multiples[i][ptrs[i]] < L: ptrs[i] += 1
                # But L jumps, so we might skip many.
                # Use bisect.
                left_idx = bisect_left(multiples[i], L)
                right_idx = bisect_right(multiples[i], L + W)
                candidates[i] = multiples[i][left_idx:right_idx]
                
            # Check assignment
            sol = check_assignment(candidates)
            if sol is not None:
                return sol
            
            # Determine next L
            # Next leave: min(candidates[i][0] + 1) for all i where candidates[i] is not empty
            # Next enter: min(m - W) for all i, where m is the first multiple in multiples[i] > L+W
            next_L_leave = max_val + 1
            next_L_enter = max_val + 1
            
            for i in range(n):
                # Leave
                if candidates[i]:
                    first_in_window = candidates[i][0]
                    next_L_leave = min(next_L_leave, first_in_window + 1)
                
                # Enter
                # Find first multiple > L+W
                idx = bisect_right(multiples[i], L + W)
                if idx < len(multiples[i]):
                    next_m = multiples[i][idx]
                    # Enters when L >= next_m - W
                    next_L_enter = min(next_L_enter, next_m - W)
            
            next_L = min(next_L_leave, next_L_enter)
            if next_L <= L:
                next_L = L + 1 # Safety
            
            L = next_L
            
        return None

    # Try W from 1 upwards
    # We can limit W to a reasonable bound, e.g., 100
    # If no solution found for W <= 100, we might need to go higher,
    # but the problem implies a small solution exists.
    
    best_solution = None
    for W in range(1, 101):
        if time.time() > deadline:
            break
            
        sol = find_solution_for_W(W)
        if sol is not None:
            best_solution = sol
            break
        
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # Fallback: If no small W found, try a random search or larger W with fewer L checks
    # Or just return the divisors themselves (valid but high score)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
