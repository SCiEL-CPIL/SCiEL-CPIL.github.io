import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 192
    max_elements = 48
    
    def check_divides_sum_of_three(S, x):
        """Check if x divides the sum of any 3 distinct elements in S."""
        L = len(S)
        if L < 3:
            return False
        # For each pair (i,j), check if there exists k such that (S[i]+S[j]+S[k]) % x == 0
        # i.e., S[k] % x == -(S[i]+S[j]) % x
        residues = [s % x for s in S]
        for i in range(L):
            for j in range(i+1, L):
                target = -(residues[i] + residues[j]) % x
                # Check if any element has this residue (and is distinct from i,j)
                for k in range(L):
                    if k != i and k != j and residues[k] == target:
                        return True
        return False
    
    def check_existing_divide_with_x(S, x):
        """Check if any y in S divides sum of any triple in S∪{x} that includes x."""
        for y in S:
            others = [z for z in S if z != y]
            L = len(others)
            if L < 2:
                continue
            residues = [z % y for z in others]
            rx = x % y
            for i in range(L):
                for j in range(i+1, L):
                    target = -(rx + residues[i] + residues[j]) % y
                    if target == 0:
                        # Need an element with residue 0, but that would be y itself which is excluded
                        # Actually we need a third element from others with residue target
                        # But we're checking triples that include x, so the triple is {x, others[i], others[j]}
                        # Wait, the triple must be 3 DISTINCT OTHER elements to y
                        # So the triple is {x, a, b} where a,b are from S\{y}
                        # We need (x + a + b) % y == 0
                        # Which is what we're checking
                        pass
                    # Check if (x + others[i] + others[j]) % y == 0
                    if (x + others[i] + others[j]) % y == 0:
                        return True
        return False
    
    def can_add(S, x):
        """Check if x can be added to S."""
        # Check 1: x doesn't divide sum of any 3 in S
        if check_divides_sum_of_three(S, x):
            return False
        # Check 2: No existing y divides sum of triple including x
        if check_existing_divide_with_x(S, x):
            return False
        return True
    
    def is_valid(S):
        """Full validation of set S."""
        S_list = list(S)
        K = len(S_list)
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            # Check if d divides sum of any 3 in others
            residues = [x % d for x in others]
            for a in range(L):
                for b in range(a+1, L):
                    target = -(residues[a] + residues[b]) % d
                    for c in range(L):
                        if c != a and c != b and residues[c] == target:
                            return False
        return True
    
    def greedy_construct(order):
        """Greedy construction following given order."""
        S = set()
        for x in order:
            if len(S) >= max_elements:
                break
            if can_add(S, x):
                S.add(x)
        return S
    
    # Phase 1: Try multiple greedy strategies
    best_S = set()
    
    # Strategy 1: Descending order
    S1 = greedy_construct(range(n, 0, -1))
    if len(S1) > len(best_S):
        best_S = S1
    
    # Strategy 2: Ascending order  
    S2 = greedy_construct(range(1, n + 1))
    if len(S2) > len(best_S):
        best_S = S2
    
    # Strategy 3: Random orders
    for _ in range(20):
        if time.time() - start_time > time_limit * 0.15:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy_construct(order)
        if len(S_rand) > len(best_S):
            best_S = S_rand
    
    S = set(best_S)
    
    # Phase 2: Local search - try to add elements
    while time.time() - start_time < time_limit * 0.5:
        improved = False
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        if not_in_S:
            random.shuffle(not_in_S)
            for x in not_in_S:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        if not improved:
            break
    
    # Phase 3: Local search - swaps
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try single swaps
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        if not_in_S and S:
            random.shuffle(not_in_S)
            S_list = list(S)
            random.shuffle(S_list)
            
            for r in S_list:
                if time.time() - start_time > time_limit:
                    break
                for x in not_in_S:
                    if x == r:
                        continue
                    new_S = (S - {r}) | {x}
                    if is_valid(new_S) and len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try 2-swap: remove two, add two
            if S and not_in_S and len(S) >= 2:
                S_list = list(S)
                random.shuffle(S_list)
                if len(S_list) >= 2:
                    r1, r2 = S_list[0], S_list[1]
                    if len(not_in_S) >= 2:
                        random.shuffle(not_in_S)
                        x1, x2 = not_in_S[0], not_in_S[1]
                        new_S = (S - {r1, r2}) | {x1, x2}
                        if is_valid(new_S) and len(new_S) >= len(S):
                            S = new_S
                            improved = True
        
        if not improved and time.time() - start_time > time_limit * 0.8:
            break
    
    return list(S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
