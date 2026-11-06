import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    # conflicts[x] = list of (u, v) such that (u, v, x) is a bad triple
    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # For faster validity checking in VNS, we might use a different representation
    # But for N=128, set operations are fast enough.

    def is_valid_add(S_set, x):
        for (u, v) in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    # Initial Solution: Greedy
    random.seed(42)
    best_set = set()
    best_size = 0
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(conflicts[x])),
    ]
    
    for _ in range(10):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S = set()
        for x in order:
            if is_valid_add(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    S = best_set.copy()
    current_size = len(S)

    # Variable Neighborhood Search (VNS)
    # Neighborhoods:
    # N1: 1-for-1 swap
    # N2: 2-for-2 swap
    # N3: 3-for-3 swap (might be too slow, maybe skip or limit)

    def get_neighbors_1(S):
        """Yield 1-for-1 swaps that improve or maintain size (we only care about improving size, 
           but in VNS we usually look for ANY neighbor, then pick the best. 
           Here, size is the objective. We want to increase size. 
           1-for-1 does not change size. 
           So we need to look for moves that INCREASE size.
           1-for-0 (add) increases size.
           2-for-1 (remove 1, add 2) increases size.
           3-for-2 (remove 2, add 3) increases size.
           
           The current program tried:
           1. Add single (1-for-0)
           2. 1-for-1 swap (size same) - used to escape local minima for adding later?
           3. 1-for-2 exchange (size +1)
           
           In VNS, we should define neighborhoods that potentially increase size.
           N0: Add 1 element (if possible).
           N1: Remove 1, Add 2 (size +1).
           N2: Remove 2, Add 3 (size +2).
           
           If no move in N0, try N1. If no move in N1, try N2.
           """
        pass

    # Let's implement a specific VNS for maximization.
    
    def try_add_one(S):
        """Try to add one element. Return new S if successful, else None."""
        for x in range(1, N + 1):
            if x not in S:
                if is_valid_add(S, x):
                    new_S = S.copy()
                    new_S.add(x)
                    return new_S
        return None

    def try_swap_1_for_2(S):
        """Try to remove 1 element and add 2. Return new S if successful, else None."""
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        # To be efficient, we can limit the search if time is tight
        # But N=128 is small.
        for u in S_list:
            temp_S = S - {u}
            # Find two candidates that can be added
            # We can iterate candidates
            added = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    added.append(x)
                    temp_S_with_x = temp_S | {x}
                    for y in candidates:
                        if y != x and is_valid_add(temp_S_with_x, y):
                            new_S = temp_S | {x, y}
                            return new_S
            # Optimization: If we found one x, we checked all y. 
            # We can break early if we found a valid pair.
        return None

    def try_swap_2_for_3(S):
        """Try to remove 2 elements and add 3. Return new S if successful, else None."""
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        # This is O(|S|^2 * |candidates|^3) which is 128^2 * 128^3 ~ 10^9. Too slow.
        # We need a smarter way or limit the search.
        # Given time limits, we might skip N2 or make it very restricted.
        # Let's try a limited version: only check first few S elements and candidates.
        
        # Actually, 1-for-2 is often sufficient to escape most local minima in these problems.
        # Let's stick to N0 and N1, and maybe a randomized 2-for-2 (size same) to escape?
        # No, we want to increase size.
        
        return None

    # VNS Loop
    stuck_count = 0
    while time.time() < deadline:
        # N0: Add 1
        new_S = try_add_one(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            stuck_count = 0
            continue
        
        # N1: 1-for-2
        new_S = try_swap_1_for_2(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            stuck_count = 0
            continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        # This allows us to escape the local optimum where no +1 move is possible.
        # We need a 1-for-1 swap that is "valid" (resulting set is valid).
        # We don't need it to be "better" (size is same), just different.
        # Then we restart N0.
        
        improved_structure = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_S = S - {u}
            for v in candidates:
                if is_valid_add(temp_S, v):
                    # This swap is valid
                    S = temp_S | {v}
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > 20:
                # Random restart from best
                S = best_set.copy()
                stuck_count = 0
                # Maybe try a random 1-for-1 swap to perturb
                S_list = list(S)
                if S_list:
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S = S - {u}
                        if is_valid_add(temp_S, v):
                            S = temp_S | {v}

    return list(best_set)
