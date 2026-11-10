from math import gcd
import random

N = 128

# Precompute GCD matrix
gcd_mat = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i, N+1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

# Precompute neighbors for each gcd value g
# neighbors[g][a] = set of b such that gcd(a, b) == g
neighbors = [set() for _ in range(N+1)]
for g in range(1, N+1):
    for a in range(1, N+1):
        for b in range(a, N+1):
            if gcd_mat[a][b] == g:
                neighbors[g].add((a, b)) # Store as frozenset or just use list?
                # Actually, for fast lookup, we want: given g and a, which b have gcd(a,b)=g?
                # Let's make neighbors[g][a] a set of b.

neighbors = [dict() for _ in range(N+1)]
for g in range(1, N+1):
    for a in range(1, N+1):
        neighbors[g][a] = set()
for a in range(1, N+1):
    for b in range(a, N+1):
        g = gcd_mat[a][b]
        neighbors[g][a].add(b)
        if a != b:
            neighbors[g][b].add(a)
        # If a == b, gcd(a,a)=a. So neighbors[a][a] contains a. But we need distinct elements.

def can_add(x, S, S_set):
    """
    Check if x can be added to S (list) and S_set (set).
    """
    # Group elements in S by their gcd with x
    groups = {}
    for a in S:
        g = gcd_mat[a][x]
        if g not in groups:
            groups[g] = []
        groups[g].append(a)
    
    for g, S_g in groups.items():
        if len(S_g) < 2:
            continue
        # Check if any pair in S_g has gcd exactly g
        # We can iterate through S_g and check neighbors
        for a in S_g:
            # Check if any other element b in S_g is in neighbors[g][a]
            # neighbors[g][a] contains all b such that gcd(a,b)=g
            # We need b in S_g and b != a
            # Since S_g is a list, we can convert to set for fast lookup? 
            # Or just check membership.
            S_g_set = set(S_g)
            for b in neighbors[g][a]:
                if b in S_g_set and b != a:
                    return False
    return True

def local_improve(S, S_set, gcd_mat):
    S_list = list(S)
    S_set = set(S_list)
    improved = True
    iterations = 0
    max_iterations = 200
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # 1. Try to add any single element
        for x in range(1, N+1):
            if x not in S_set and can_add(x, S_list, S_set):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try 1-for-1 swap
        if len(S_list) > 0:
            candidates_to_remove = random.sample(S_list, min(20, len(S_list)))
            for x in candidates_to_remove:
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                # Try to add any y
                for y in range(1, N+1):
                    if y in S_temp_set:
                        continue
                    if can_add(y, S_temp_list, S_temp_set):
                        S_list = S_temp_list + [y]
                        S_set = S_temp_set | {y}
                        improved = True
                        break
                if improved:
                    break

        if improved:
            continue

        # 3. Try 1-for-2 swap (Remove 1, Add 2) - Net gain +1
        if len(S_list) > 0:
            candidates_to_remove = random.sample(S_list, min(20, len(S_list)))
            for x in candidates_to_remove:
                S_temp_list = [u for u in S_list if u != x]
                S_temp_set = S_set - {x}
                
                added = 0
                new_S_list = list(S_temp_list)
                new_S_set = set(S_temp_set)
                
                # Try to add 2 elements
                candidates = list(range(1, N+1))
                random.shuffle(candidates)
                for y in candidates:
                    if y in new_S_set:
                        continue
                    if can_add(y, new_S_list, new_S_set):
                        new_S_list.append(y)
                        new_S_set.add(y)
                        added += 1
                        if added == 2:
                            break
                            
                if added == 2:
                    S_list = new_S_list
                    S_set = new_S_set
                    improved = True
                    break

        if improved:
            continue

        # 4. Try 2-for-3 swap (Remove 2, Add 3) - Net gain +1
        if len(S_list) > 1:
            candidates_to_remove = random.sample(S_list, min(10, len(S_list)))
            for x1 in candidates_to_remove:
                remaining = [u for u in S_list if u != x1]
                if not remaining: continue
                candidates_to_remove2 = random.sample(remaining, min(10, len(remaining)))
                for x2 in candidates_to_remove2:
                    S_temp_list = [u for u in S_list if u != x1 and u != x2]
                    S_temp_set = S_set - {x1, x2}
                    
                    added = 0
                    new_S_list = list(S_temp_list)
                    new_S_set = set(S_temp_set)
                    
                    candidates = list(range(1, N+1))
                    random.shuffle(candidates)
                    for y in candidates:
                        if y in new_S_set:
                            continue
                        if can_add(y, new_S_list, new_S_set):
                            new_S_list.append(y)
                            new_S_set.add(y)
                            added += 1
                            if added == 3:
                                break
                                
                    if added == 3:
                        S_list = new_S_list
                        S_set = new_S_set
                        improved = True
                        break
                if improved:
                    break

    return S_list

def construct():
    best_set = []
    
    # Heuristic orders
    def count_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = [count_divisors(i) for i in range(N+1)]
    order_by_div = list(range(1, N+1))
    order_by_div.sort(key=lambda x: div_counts[x])
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        order_by_div,
        list(reversed(order_by_div)),
    ]
    
    # Add random orders
    for _ in range(50):
        cand = list(range(1, 129))
        random.shuffle(cand)
        orders.append(cand)
        
    for order in orders:
        S = []
        S_set = set()
        for x in order:
            if can_add(x, S, S_set):
                S.append(x)
                S_set.add(x)
                
        # Local improvement
        S = local_improve(S, S_set, gcd_mat)
        
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
