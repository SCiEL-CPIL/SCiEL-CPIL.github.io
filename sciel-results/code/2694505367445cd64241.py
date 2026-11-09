import time
from math import gcd
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
    
    # Precompute bad_c table: bad_c[a][b] = set of c > b such that (a,b,c) is bad
    # A triple (a,b,c) with a<b<c is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_c = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_c[a][b].add(c)
    
    def can_add(S_set, x):
        """Check if x can be added to S without creating a bad triple."""
        for a in S_set:
            for b in S_set:
                if a == b:
                    continue
                if a < b:
                    if x in bad_c[a][b]:
                        return False
                elif b < a:
                    if x in bad_c[b][a]:
                        return False
        return True
    
    def greedy_build(order):
        """Build a valid set by greedily adding elements in the given order."""
        S_set = set()
        for x in order:
            if can_add(S_set, x):
                S_set.add(x)
        return S_set
    
    def local_search(S_set, max_steps=5000):
        """Improve S_set using local search with add/remove/swap moves."""
        S_list = list(S_set)
        current_size = len(S_set)
        best_size = current_size
        best_S = set(S_set)
        
        step = 0
        no_improve = 0
        while step < max_steps and no_improve < 200:
            if time.time() > deadline - 3.0:
                break
            
            # Choose move type
            r = random.random()
            if r < 0.4:
                move_type = 'add'
            elif r < 0.7:
                move_type = 'remove'
            else:
                move_type = 'swap'
            
            if move_type == 'add':
                # Try to add a random element not in S
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(S_set, x):
                    S_set.add(x)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S_set)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type == 'remove':
                # Try to remove a random element from S
                if not S_set:
                    continue
                x = random.choice(list(S_set))
                S_set.remove(x)
                current_size -= 1
                # This is always valid, but we don't want to shrink
                # Only keep if we can immediately re-add something
                # For now, just revert
                S_set.add(x)
                current_size += 1
                no_improve += 1
                
            elif move_type == 'swap':
                # Remove x from S, add y to S
                if not S_set:
                    continue
                x = random.choice(list(S_set))
                candidates = [y for y in range(1, N + 1) if y not in S_set]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Try the swap: remove x, check if y can be added
                S_set.remove(x)
                if can_add(S_set, y):
                    S_set.add(y)
                    # Size unchanged, but structure changed
                    no_improve += 1
                else:
                    # Revert
                    S_set.add(x)
                    no_improve += 1
            
            step += 1
        
        return best_S
    
    # Multiple restarts with different orderings
    best_S = set()
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    if time.time() < deadline - 5.0:
        S = greedy_build(list(range(1, N + 1)))
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline - 5.0:
        S = greedy_build(list(range(N, 0, -1)))
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Strategy 3: Random orderings
    restart_count = 0
    while time.time() < deadline - 5.0 and restart_count < 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        restart_count += 1
    
    # Strategy 4: Start from best, do more local search
    if time.time() < deadline - 3.0 and best_S:
        S = local_search(set(best_S), max_steps=10000)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Convert to list
    result = sorted(list(best_S)) if best_S else [1]
    
    return result

# EVOLVE-BLOCK-END
