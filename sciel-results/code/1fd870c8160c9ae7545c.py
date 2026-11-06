import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcds = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
    
    # conflicts[x] = list of (a,b) pairs such that {a,b,x} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                if gcds[a][c] == gab and gcds[b][c] == gab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))
    
    conflict_lists = [list(conflicts[i]) for i in range(N + 1)]
    conflict_counts = [len(conflict_lists[i]) for i in range(N + 1)]
    
    def can_add(x, S):
        for a, b in conflict_lists[x]:
            if (S >> a) & 1 and (S >> b) & 1:
                return False
        return True
    
    def greedy_solve(order, S=0):
        for x in order:
            if not (S >> x) & 1 and can_add(x, S):
                S |= (1 << x)
        return S
    
    best_size = 0
    best_mask = 0
    
    # Get good initial solutions
    order_least = sorted(elements, key=lambda x: conflict_counts[x])
    S = greedy_solve(order_least, 0)
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S
    
    order_most = sorted(elements, key=lambda x: conflict_counts[x], reverse=True)
    S = greedy_solve(order_most, 0)
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S
    
    for _ in range(80):
        if time.time() > deadline:
            break
        order = elements[:]
        random.shuffle(order)
        S = greedy_solve(order, 0)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
    
    # Tabu search
    current_mask = best_mask
    current_size = best_size
    
    # Tabu list: elements that were recently added/removed
    tabu_list = {}  # element -> iteration when it becomes non-tabu
    tabu_tenure = 30
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        in_S = [x for x in elements if (current_mask >> x) & 1]
        not_in_S = [x for x in elements if not (current_mask >> x) & 1]
        
        best_move = None
        best_delta = 0
        best_new_mask = None
        
        # Try all 1-swap moves: remove a, add x
        for a in in_S:
            if tabu_list.get(a, 0) > iteration:
                continue
            S_without_a = current_mask ^ (1 << a)
            for x in not_in_S:
                if tabu_list.get(x, 0) > iteration:
                    continue
                if can_add(x, S_without_a):
                    S_new = S_without_a | (1 << x)
                    delta = S_new.bit_count() - current_size
                    # Aspiration: allow tabu move if it improves global best
                    if delta > best_delta or (delta == best_delta and random.random() < 0.1):
                        best_delta = delta
                        best_move = (a, x)
                        best_new_mask = S_new
        
        # Try all single additions
        for x in not_in_S:
            if tabu_list.get(x, 0) > iteration:
                continue
            if can_add(x, current_mask):
                S_new = current_mask | (1 << x)
                delta = S_new.bit_count() - current_size
                if delta > best_delta:
                    best_delta = delta
                    best_move = (None, x)
                    best_new_mask = S_new
        
        # Try 2-swap moves (remove 2, add 2) - more expensive but can escape local optima
        if len(in_S) >= 2 and len(not_in_S) >= 2 and iteration % 3 == 0:
            # Sample some 2-swap moves to keep it fast
            for _ in range(20):
                if time.time() > deadline:
                    break
                a1, a2 = random.sample(in_S, 2)
                if tabu_list.get(a1, 0) > iteration or tabu_list.get(a2, 0) > iteration:
                    continue
                x1, x2 = random.sample(not_in_S, 2)
                if tabu_list.get(x1, 0) > iteration or tabu_list.get(x2, 0) > iteration:
                    continue
                S_without = current_mask ^ (1 << a1) ^ (1 << a2)
                if can_add(x1, S_without) and can_add(x2, S_without | (1 << x1)):
                    S_new = S_without | (1 << x1) | (1 << x2)
                    delta = S_new.bit_count() - current_size
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (a1, a2, x1, x2)
                        best_new_mask = S_new
        
        # Apply best move
        if best_move is not None and best_new_mask is not None:
            # Update tabu list
            if len(best_move) == 2:
                a, x = best_move
                if a is not None:
                    tabu_list[a] = iteration + tabu_tenure
                tabu_list[x] = iteration + tabu_tenure
            else:
                a1, a2, x1, x2 = best_move
                tabu_list[a1] = iteration + tabu_tenure
                tabu_list[a2] = iteration + tabu_tenure
                tabu_list[x1] = iteration + tabu_tenure
                tabu_list[x2] = iteration + tabu_tenure
            
            current_mask = best_new_mask
            current_size = best_new_mask.bit_count()
            
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        else:
            # No improving move found; try a random kick to escape local optimum
            if not_in_S and in_S:
                a = random.choice(in_S)
                x = random.choice(not_in_S)
                S_new = current_mask ^ (1 << a)
                if can_add(x, S_new):
                    current_mask = S_new | (1 << x)
                    current_size = current_mask.bit_count()
                    tabu_list[a] = iteration + tabu_tenure
                    tabu_list[x] = iteration + tabu_tenure
    
    # Final greedy pass
    for x in elements:
        if not (best_mask >> x) & 1 and can_add(x, best_mask):
            best_mask |= (1 << x)
    
    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result
