import time
import sys
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of all divisors to determine the period of the problem
    # The pattern of which numbers are multiples of which divisors repeats every L
    L = reduce(_lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Theoretical minimum span for 10 distinct integers is 9
    # We search for the minimal span W
    for W in range(9, 1000):  # Upper bound on W is likely small
        if time.time() > deadline:
            break
            
        # We only need to check A in [1, L] because the problem is periodic with period L
        # However, we can restrict the range of A further if needed, but L=30030 is small.
        # Actually, we can just scan A from 1 to L.
        # If a solution exists for A, it also exists for A + k*L.
        # Since we just need *one* solution, scanning [1, L] is sufficient.
        
        found = False
        for A in range(1, L + 1):
            if time.time() > deadline:
                break
                
            # Build bipartite graph
            # Left: indices 0..n-1 (divisors)
            # Right: integers in [A, A+W]
            # We need to assign each divisor a distinct integer in the window that is a multiple of it.
            
            # Let's use a simple DFS based bipartite matching (Hopcroft-Karp is overkill for n=10)
            # match[j] = i means integer (A+j) is matched to divisor i
            # We iterate j from 0 to W
            # Actually, it's easier to iterate divisors and try to find an available multiple.
            # But standard bipartite matching is cleaner.
            
            # Let's use the standard "try to augment" approach.
            # match_div[i] = value (integer) assigned to divisor i, or -1
            # match_val[v] = index i of divisor assigned to value v, or -1
            # v ranges from A to A+W
            
            # To make indexing easier, let's map values to 0..W
            # val = A + j
            # match_val[j] = i
            
            match_div = [-1] * n
            match_val = [-1] * (W + 1)
            
            # We will try to match each divisor i to some value j in 0..W
            # such that (A+j) is divisible by divisors[i]
            
            # Standard DFS for bipartite matching
            # visited[j] is used to avoid cycles in a single DFS call
            
            def dfs(i, visited):
                for j in range(W + 1):
                    if visited[j]:
                        continue
                    val = A + j
                    if val % divisors[i] == 0:
                        visited[j] = True
                        # If value j is not matched, or the divisor currently matched to j can find another value
                        if match_val[j] == -1 or dfs(match_val[j], visited):
                            match_val[j] = i
                            match_div[i] = val
                            return True
                return False
            
            # We need to find a perfect matching for all n divisors
            # We can iterate i from 0 to n-1 and try to augment
            # But the order might matter? No, bipartite matching finds max matching.
            # If max matching size is n, we are good.
            
            # Reset match arrays for each A
            # We do this inside the loop
            
            # Optimization: Pre-check if every divisor has at least one multiple in [A, A+W]
            # This is a necessary condition.
            possible = True
            for i in range(n):
                d = divisors[i]
                # Smallest multiple >= A
                m = ((A + d - 1) // d) * d
                if m > A + W:
                    possible = False
                    break
            if not possible:
                continue
                
            # Now run matching
            # We need to be careful with recursion depth, but n=10 and W<1000, so it's fine.
            # However, python recursion limit is 1000, so we are safe.
            
            # We'll use a list for visited to avoid creating a new set every time if we optimize, 
            # but here creating a set/list is fine.
            
            # We need to reset match_val for each A
            # Actually, we can just re-initialize match_div and match_val inside the A loop.
            
            # Wait, the `dfs` function closes over `match_val` and `match_div`.
            # We need to ensure they are reset for each A.
            
            match_div = [-1] * n
            match_val = [-1] * (W + 1)
            
            # We want to match all divisors.
            # Standard algorithm:
            # For each left node u, try to find an augmenting path.
            # But we need to match ALL left nodes.
            
            # Let's iterate through divisors.
            # If we can match all, we found a solution.
            
            # Note: The order of processing divisors doesn't change the existence of a perfect matching,
            # but it might affect the specific assignment. We just need *one* assignment.
            
            # However, a simple greedy or DFS matching might fail to find a perfect matching if it gets stuck in a local optimum?
            # No, the standard augmenting path algorithm (BFS/DFS) guarantees finding the maximum matching.
            # If the maximum matching size is n, we have a perfect matching.
            
            # Implementation of Hopcroft-Karp is more complex but faster.
            # Given n=10, simple DFS is sufficient.
            
            # Let's implement the standard DFS-based bipartite matching.
            
            def try_augment(i, visited):
                for j in range(W + 1):
                    if visited[j]:
                        continue
                    val = A + j
                    if val % divisors[i] == 0:
                        visited[j] = True
                        if match_val[j] == -1 or try_augment(match_val[j], visited):
                            match_val[j] = i
                            match_div[i] = val
                            return True
                return False
            
            # We need to run try_augment for each i?
            # Actually, the standard algorithm iterates over all left nodes and tries to find an augmenting path for each.
            # If we successfully augment for all left nodes, we have a perfect matching.
            
            # But wait, `match_div` is updated inside `try_augment`.
            # We need to call `try_augment` for each i from 0 to n-1?
            # No, the standard algorithm is:
            # matching = 0
            # for u in left_nodes:
            #   visited = set()
            #   if dfs(u, visited):
            #      matching += 1
            #   if matching == n: break
            
            # Let's do that.
            
            matching_size = 0
            for i in range(n):
                visited = [False] * (W + 1)
                if try_augment(i, visited):
                    matching_size += 1
                    if matching_size == n:
                        break
                # If we can't augment for i, we continue. The final matching_size will tell us if it's perfect.
                # Actually, if we fail to augment for one, we might still succeed for others, but the total size won't be n.
                
            if matching_size == n:
                # We found a valid assignment
                # match_div[i] contains the value for divisor i
                # Check distinctness (guaranteed by matching)
                # Check bounds (A >= 1, A+W <= 1000000)
                # A is in [1, L] and L=30030, so A+W is small.
                # We must return the list in the order of divisors.
                return match_div
        
    # Fallback if no solution found within time or W limit
    # This should not happen for W < 100
    # But just in case, return a valid (though not optimal) set
    # e.g., just the divisors themselves? No, they must be distinct.
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are distinct.
    # Span = 39 - 6 = 33.
    # This is a valid solution.
    return divisors[:]
