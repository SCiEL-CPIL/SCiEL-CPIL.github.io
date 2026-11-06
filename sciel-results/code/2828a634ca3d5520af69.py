# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    best_set = []
    
    def is_valid_set(S):
        """Check if a set S is valid."""
        S_list = list(S)
        n = len(S_list)
        # For each a in S, check if a divides sum of any 3 distinct others
        for i, a in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True

    def greedy_construct(order):
        """Build a set using the given order of candidates."""
        S = []
        for x in order:
            # Check if x can be added
            # 1. x does not divide sum of any 3 in S
            valid = True
            if len(S) >= 3:
                for combo in combinations(S, 3):
                    if sum(combo) % x == 0:
                        valid = False
                        break
            
            # 2. No a in S divides sum of any 3 in S U {x} involving x
            if valid:
                for a in S:
                    # Triples involving x: (x, u, v) where u, v in S, u != v
                    # Sum = x + u + v
                    # Must not be divisible by a
                    # Note: a is in S. u, v are in S.
                    # u, v must be distinct from a? 
                    # "3 distinct OTHER selected integers"
                    # If the triple is (x, u, v), the "other" integers for a are x, u, v.
                    # They must be distinct from a.
                    # Since x is new, x != a.
                    # u, v are from S. If u == a or v == a, then a is in the triple?
                    # The condition is: a does not divide sum of 3 distinct OTHERS.
                    # If the triple contains a, it's not "other".
                    # So we only consider triples from (S U {x}) \ {a}.
                    # So u, v must be in S \ {a}.
                    
                    others_in_S = [y for y in S if y != a]
                    # We need to pick 2 from others_in_S, say u, v.
                    # The triple is (x, u, v).
                    # Sum = x + u + v.
                    # Check if a divides this sum.
                    for u_idx in range(len(others_in_S)):
                        u = others_in_S[u_idx]
                        for v_idx in range(u_idx + 1, len(others_in_S)):
                            v = others_in_S[v_idx]
                            if (x + u + v) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                S.append(x)
                
            if time.time() > deadline:
                break
                
        return S

    # Try different orders
    orders = []
    orders.append(list(range(192, 0, -1))) # Descending
    orders.append(list(range(1, 193)))     # Ascending
    
    # Random restarts
    random.seed(42)
    for _ in range(20):
        perm = list(range(1, 193))
        random.shuffle(perm)
        orders.append(perm)
        
    best_size = 0
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return best_set
# EVOLVE-BLOCK-END
