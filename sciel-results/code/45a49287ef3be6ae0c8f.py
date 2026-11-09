import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute divisors for each number
    divisors = [[] for _ in range(n + 1)]
    for d in range(1, n + 1):
        for multiple in range(d, n + 1, d):
            divisors[multiple].append(d)
    
    # For each x and each d|x, precompute which elements a have gcd(x,a) = d
    # i.e., gcd(x/d, a/d) = 1 and d|a
    # We'll use this for fast conflict checking
    
    # Precompute: for each x, for each d|x, the set of a such that gcd(x,a) = d
    gcd_groups = {}  # gcd_groups[x][d] = set of a where gcd(x,a) = d
    for x in range(1, n + 1):
        gcd_groups[x] = {}
        for d in divisors[x]:
            # Find all a where gcd(x, a) = d
            # This means d|a and gcd(x/d, a/d) = 1
            group = set()
            xd = x // d
            for a in range(d, n + 1, d):
                ad = a // d
                if gcd(xd, ad) == 1:
                    group.add(a)
            gcd_groups[x][d] = group
    
    # Precompute bad_triples_conflict: for each x, and for each pair (a,b) that would form
    # a bad triple with x, we need to detect this quickly.
    # A triple (x, a, b) is bad if gcd(x,a) = gcd(x,b) = gcd(a,b) = d for some d.
    # This means d|x, d|a, d|b, and gcd(x/d, a/d) = gcd(x/d, b/d) = gcd(a/d, b/d) = 1.
    
    # For fast checking: when adding x to a set S, check if there exist a,b in S
    # such that gcd(x,a) = gcd(x,b) = gcd(a,b).
    # Group elements in S by their gcd with x. For each group with gcd value d,
    # check if any pair in the group has gcd equal to d.
    
    def can_add(x, selected_set):
        """Check if x can be added without creating a bad triple."""
        # For each d|x, get elements in selected_set that have gcd(x, a) = d
        # For each such group, check if any pair has gcd equal to d
        for d in divisors[x]:
            group_elements = [a for a in selected_set if a in gcd_groups[x][d]]
            if len(group_elements) >= 2:
                # Check if any pair in group_elements has gcd == d
                for i in range(len(group_elements)):
                    for j in range(i + 1, len(group_elements)):
                        if gcd_table[group_elements[i]][group_elements[j]] == d:
                            return False
        return True
    
    def is_valid(selected_set):
        """Verify the entire set is valid."""
        lst = list(selected_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                ga = gcd_table[lst[i]][lst[j]]
                for k in range(j + 1, len(lst)):
                    if ga == gcd_table[lst[i]][lst[k]] == gcd_table[lst[j]][lst[k]]:
                        return False
        return True
    
    import random
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Greedy with multiple orderings - use structural insight for ordering
    # Elements with fewer divisors are "safer" to include
    # Sort by number of divisors (fewer divisors = less likely to create conflicts)
    max_restarts = 300
    for r in range(max_restarts):
        if time.time() > deadline - 35:
            break
        
        if r < 50:
            # Deterministic orderings based on structural properties
            if r == 0:
                ordering = sorted(range(1, 129), key=lambda x: len(divisors[x]))
            elif r == 1:
                ordering = sorted(range(1, 129), key=lambda x: (-len(divisors[x]), x))
            elif r == 2:
                # Primes first
                primes = [x for x in range(1, 129) if len(divisors[x]) == 2]
                others = [x for x in range(1, 129) if x not in primes]
                ordering = primes + others
            elif r == 3:
                # Powers of 2 first
                p2 = [x for x in range(1, 129) if x & (x-1) == 0]
                others = [x for x in range(1, 129) if x not in p2]
                ordering = p2 + others
            else:
                # Various structural orderings
                ordering = list(range(1, 129))
                rng.shuffle(ordering)
                # Sort by some structural criterion
                if r % 2 == 0:
                    ordering.sort(key=lambda x: len(divisors[x]))
                else:
                    ordering.sort(key=lambda x: x)
        else:
            ordering = list(range(1, 129))
            rng.shuffle(ordering)
        
        selected = set()
        for x in ordering:
            if can_add(x, selected):
                selected.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Phase 2: Local search - greedy fill
    if time.time() < deadline - 15:
        current = set(best_set)
        for _ in range(200):
            if time.time() > deadline - 15:
                break
            remaining = [x for x in range(1, 129) if x not in current]
            rng.shuffle(remaining)
            improved = False
            for x in remaining:
                if can_add(x, current):
                    current.add(x)
                    improved = True
            if not improved:
                break
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Phase 3: 1-for-1 swap local search
    if time.time() < deadline - 10:
        current = set(best_set)
        max_swap_iters = 1000
        for it in range(max_swap_iters):
            if time.time() > deadline - 10:
                break
            
            found_swap = False
            cur_list = list(current)
            rng.shuffle(cur_list)
            for remove in cur_list:
                if time.time() > deadline - 10:
                    break
                
                temp = set(current)
                temp.discard(remove)
                
                remaining = [x for x in range(1, 129) if x not in temp]
                rng.shuffle(remaining)
                
                for x in remaining:
                    if can_add(x, temp):
                        temp.add(x)
                        current = temp
                        found_swap = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    # Phase 4: 1-for-2 swap local search
    if time.time() < deadline - 5:
        current = set(best_set)
        max_swap_iters = 500
        for it in range(max_swap_iters):
            if time.time() > deadline - 5:
                break
            
            found_swap = False
            cur_list = list(current)
            rng.shuffle(cur_list)
            for remove in cur_list:
                if time.time() > deadline - 5:
                    break
                
                temp = set(current)
                temp.discard(remove)
                
                remaining = [x for x in range(1, 129) if x not in temp and x != remove]
                rng.shuffle(remaining)
                
                added = []
                for x in remaining:
                    if len(added) == 2:
                        break
                    if can_add(x, temp | set(added)):
                        added.append(x)
                
                if len(added) == 2:
                    temp.update(added)
                    current = temp
                    found_swap = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    break
            
            if not found_swap:
                break
    
    # Phase 5: 2-for-3 swap local search
    if time.time() < deadline - 3:
        current = set(best_set)
        max_swap_iters = 200
        for it in range(max_swap_iters):
            if time.time() > deadline - 3:
                break
            
            found_swap = False
            cur_list = list(current)
            rng.shuffle(cur_list)
            for i in range(len(cur_list)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(cur_list)):
                    if time.time() > deadline - 3:
                        break
                    
                    remove1, remove2 = cur_list[i], cur_list[j]
                    temp = set(current)
                    temp.discard(remove1)
                    temp.discard(remove2)
                    
                    remaining = [x for x in range(1, 129) if x not in temp]
                    rng.shuffle(remaining)
                    
                    added = []
                    for x in remaining:
                        if len(added) == 3:
                            break
                        if can_add(x, temp | set(added)):
                            added.append(x)
                    
                    if len(added) == 3:
                        temp.update(added)
                        current = temp
                        found_swap = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    # Phase 6: Final polish
    if time.time() < deadline:
        current = set(best_set)
        remaining = [x for x in range(1, 129) if x not in current]
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(x, current):
                current.add(x)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Final validation
    if not is_valid(best_set):
        # Fallback: remove elements until valid
        lst = list(best_set)
        for i in range(len(lst)):
            if time.time() > deadline:
                break
            test_set = set(lst[:i] + lst[i+1:])
            if is_valid(test_set):
                best_set = test_set
                break
    
    return sorted(best_set)
