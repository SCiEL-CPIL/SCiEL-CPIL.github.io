import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    
    # Precompute: for each pair (i,j) with i<j, compute i+j
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c are ALL in the set
    # We want to maximize |S| such that no forbidden triple exists
    
    # Strategy: Start with a good initial solution, then use local search
    # to try to add elements or swap elements to increase size
    
    # Initial solution: upper half
    # Let's try a smarter initial solution
    # We can include some small numbers if we exclude their pair sums from the upper set
    
    # Let's think about it differently:
    # If S contains elements from [1..K] and [K+1..128], 
    # then for a,b,c all in [K+1..128], a+b > 2K+2. If 2K+2 > 128, i.e., K > 63, 
    # then no pair sums of upper elements are in range.
    # For mixed triples, we need to be careful.
    
    # Let's use a local search approach:
    # 1. Start with the upper half [65..128] (size 64)
    # 2. Try to add elements from [1..64] one at a time
    # 3. If adding an element creates a forbidden triple, remove elements to fix it
    
    # Better: use simulated annealing / local search on the full problem
    
    # Represent the set as a bitmask (128 bits)
    # Precompute conflict information
    
    # A triple (a,b,c) is "bad" if a+b in S, a+c in S, b+c in S
    # where a,b,c are distinct elements of S
    
    # Let's precompute for each value v in 1..128, which pairs (i,j) have i+j = v
    # pairs_summing_to[v] = list of (i,j) with i<j, i+j=v, i,j in 1..128
    
    pairs_summing_to = [[] for _ in range(129)]
    for v in range(2, 129):
        for i in range(1, min(v-1, 128)+1):
            j = v - i
            if j > i and j <= 128:
                pairs_summing_to[v].append((i, j))
    
    def is_valid(S):
        """Check if set S (as a set of ints) has no forbidden triple."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a >= b:
                    continue
                for c in S:
                    if c <= b:
                        continue
                    # Check if a+b, a+c, b+c are all in S
                    if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                        return False
        return True
    
    # For efficiency, let's use a different approach
    # Precompute for each pair of elements (i,j) in the universe, 
    # which third elements k would create a forbidden triple with i,j
    # Actually, let's just do local search with efficient checking
    
    # Local search: maintain a set, try to add/remove/swap elements
    
    def count_forbidden_triples(S_set, S_list):
        """Count forbidden triples in S_set."""
        count = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j+1, n):
                    c = S_list[k]
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        return count
    
    # Start with upper half
    current_set = set(range(65, 129))
    current_list = sorted(current_set)
    
    best_size = len(current_set)
    best_set = set(current_set)
    
    # Try to add elements from 1..64
    # For each candidate x in 1..64, check if adding x creates any forbidden triple
    # A forbidden triple involving x and two existing elements a,b requires:
    #   x+a in S, x+b in S, a+b in S
    # Or x is one of the pair sums... wait, x is an element, not a sum.
    # The triple is (a,b,c) where a+b, a+c, b+c are all in S.
    # If we add x, new forbidden triples involve x as one of a,b,c.
    # Case 1: x, a, b where a,b in current_set
    #   Need: x+a in S', x+b in S', a+b in S'
    #   S' = S ∪ {x}
    #   So need: x+a in S or x+a==x (impossible since a>=1), 
    #            x+b in S or x+b==x, a+b in S or a+b==x
    #   Since x <= 64 and a,b >= 65, x+a >= 66, x+b >= 66, a+b >= 131
    #   a+b >= 131 > 128, so a+b is never in S (since S ⊆ {1..128})
    #   Wait, that means if a,b are both >= 65, a+b >= 131 > 128, so a+b is never in S!
    #   Therefore, no forbidden triple can have two elements both >= 65!
    
    # This is a key insight: any two elements from [65,128] have sum >= 131 > 128,
    # so they can never both be part of a forbidden triple (since their sum would need to be in S).
    
    # So forbidden triples can have at most ONE element from [65,128].
    # A forbidden triple (a,b,c) with exactly one element >= 65, say c >= 65, and a,b <= 64:
    #   Need: a+b in S, a+c in S, b+c in S
    #   a+b <= 128, a+c in [66, 192], b+c in [66, 192]
    #   For a+c and b+c to be in S, they must be <= 128, so a+c <= 128 and b+c <= 128
    #   Since c >= 65, a <= 63 and b <= 63.
    
    # A forbidden triple with all three from [1,64]:
    #   a+b, a+c, b+c all in [2, 128], need to be in S.
    
    # So the constraint is:
    # 1. No three elements from [1,64] in S have all three pair sums in S
    # 2. No two elements from [1,64] and one from [65,128] in S have all three pair sums in S
    #    (which requires the two small elements' sum to be in S, and each small+large to be in S)
    
    # Let's reconsider. Let U = S ∩ [65,128] and L = S ∩ [1,64].
    # Forbidden triples:
    # Type A: three from L. (a,b,c) in L, a+b, a+c, b+c all in S.
    # Type B: two from L, one from U. (a,b) in L, c in U. a+b in S, a+c in S, b+c in S.
    #   a+c in S means a+c in U (since a+c >= 66 if c>=65 and a>=1, so a+c in [66,128])
    #   Similarly b+c in U.
    #   And a+b in S means a+b in L or a+b in U. Since a,b <= 64, a+b <= 128.
    #   If a+b <= 64, a+b in L. If a+b > 64, a+b in U.
    # Type C: one from L, two from U. (a) in L, (b,c) in U. a+b in S, a+c in S, b+c in S.
    #   b+c >= 131 > 128, so b+c not in S. Impossible!
    # Type D: three from U. a+b >= 131 > 128. Impossible!
    
    # So only Type A and Type B are possible forbidden triples.
    
    # Type B constraint: for any a,b in L and c in U, NOT(a+b in S AND a+c in S AND b+c in S)
    #   Equivalently: for any c in U, the set {a+b : a,b in L, a<b, a+c in S, b+c in S} must not be in S.
    #   Or: for any a,b in L with a+b in S, the set {c in U : a+c in S and b+c in S} must be empty.
    
    # Let's use local search. Start with U = [65,128], L = empty. Size = 64.
    # Try to add elements to L.
    
    # When we add x to L, we need to check:
    # 1. Type A: for any a,b in L (including x), a+b, a+c, b+c all in S
    # 2. Type B: for any a,b in L (including x) and c in U, a+b in S, a+c in S, b+c in S
    
    # Let's implement a greedy + local search approach
    
    def check_add(x, L_set, U_set, S_set):
        """Check if adding x to L creates any forbidden triple."""
        new_L = L_set | {x}
        new_S = S_set | {x}
        
        # Check Type A: three from new_L
        L_list = sorted(new_L)
        for i in range(len(L_list)):
            a = L_list[i]
            for j in range(i+1, len(L_list)):
                b = L_list[j]
                ab = a + b
                if ab not in new_S:
                    continue
                for k in range(j+1, len(L_list)):
                    c = L_list[k]
                    if (a + c) in new_S and (b + c) in new_S:
                        return False
        
        # Check Type B: two from new_L, one from U_set
        for i in range(len(L_list)):
            a = L_list[i]
            for j in range(i+1, len(L_list)):
                b = L_list[j]
                ab = a + b
                if ab not in new_S:
                    continue
                for c in U_set:
                    if (a + c) in new_S and (b + c) in new_S:
                        return False
        
        return True
    
    # Greedy: try to add elements from L candidates in some order
    # Then local search to improve
    
    # Let's try multiple random orderings and greedy adds
    
    best_result = set(range(65, 129))
    best_size = 64
    
    candidates = list(range(1, 65))
    
    # Try several random orderings
    for trial in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        random.shuffle(candidates)
        L_set = set()
        U_set = set(range(65, 129))
        S_set = set(range(65, 129))
        
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if check_add(x, L_set, U_set, S_set):
                L_set.add(x)
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
        
        if trial % 50 == 0:
            pass
    
    # Local search on best_result: try swapping elements
    if time.time() - start_time < DEADLINE:
        current = set(best_result)
        L_set = current & set(range(1, 65))
        U_set = current & set(range(65, 129))
        
        # Try to remove one element and add two
        # Or just try adding more elements with different strategies
        # Try removing a U element and adding two L elements
        
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            # Try each L candidate not in current
            for x in range(1, 65):
                if x in current:
                    continue
                if check_add(x, L_set, U_set, current):
                    current.add(x)
                    L_set.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one U element and adding one or more L elements
                for u in sorted(U_set):
                    if time.time() - start_time > DEADLINE:
                        break
                    new_U = U_set - {u}
                    new_L = L_set
                    new_S = current - {u}
                    # Try to add L elements
                    added = True
                    while added and time.time() - start_time < DEADLINE:
                        added = False
                        for x in range(1, 65):
                            if x in new_S:
                                continue
                            if check_add(x, new_L, new_U, new_S):
                                new_L.add(x)
                                new_S.add(x)
                                added = True
                                break
                    if len(new_S) > len(current):
                        current = new_S
                        L_set = new_L
                        U_set = new_U
                        improved = True
                        break
    
    result = sorted(best_result)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result[:20]}...{result[-20:]}")
    
    # Verify
    S = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if c <= b:
                    continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    print(f"Forbidden triple: {a},{b},{c}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
