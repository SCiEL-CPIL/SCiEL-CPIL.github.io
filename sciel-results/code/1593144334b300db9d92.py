import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all 4-term arithmetic progressions
    # aps[x] contains a list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP
    aps = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps[a].append((b, c, e))
            aps[b].append((a, c, e))
            aps[c].append((a, b, e))
            aps[e].append((a, b, c))

    best_set = set()
    best_size = 0

    # Greedy initialization
    def greedy_init(order):
        S = set()
        conflict = [0] * (N + 1)
        for x in order:
            if conflict[x] == 0:
                S.add(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        conflict[c] += 1
                    if a in S and c in S:
                        conflict[b] += 1
                    if b in S and c in S:
                        conflict[a] += 1
        return S, conflict

    # Local search with add/remove/swap
    def local_search(S, conflict, max_steps=50000):
        current_size = len(S)
        steps_since_improvement = 0
        
        while steps_since_improvement < max_steps and time.time() < deadline:
            # Find all valid elements to add
            valids = []
            for x in range(1, N + 1):
                if x not in S and conflict[x] == 0:
                    valids.append(x)
            
            if valids:
                # Add a random valid element
                x = random.choice(valids)
                S.add(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        conflict[c] += 1
                    if a in S and c in S:
                        conflict[b] += 1
                    if b in S and c in S:
                        conflict[a] += 1
                
                if len(S) > current_size:
                    current_size = len(S)
                    steps_since_improvement = 0
            else:
                # Try swaps: remove x, add y
                if S:
                    # Find elements to remove
                    removable = list(S)
                    random.shuffle(removable)
                    found_swap = False
                    
                    for x_remove in removable:
                        if len(removable) > 20:  # Limit swap search
                            break
                            
                        # Temporarily remove x_remove
                        S.discard(x_remove)
                        for a, b, c in aps[x_remove]:
                            if a in S and b in S:
                                conflict[c] -= 1
                            if a in S and c in S:
                                conflict[b] -= 1
                            if b in S and c in S:
                                conflict[a] -= 1
                        
                        # Try to add any valid element
                        for y_add in range(1, N + 1):
                            if y_add not in S and conflict[y_add] == 0:
                                S.add(y_add)
                                for a, b, c in aps[y_add]:
                                    if a in S and b in S:
                                        conflict[c] += 1
                                    if a in S and c in S:
                                        conflict[b] += 1
                                    if b in S and c in S:
                                        conflict[a] += 1
                                found_swap = True
                                break
                        
                        if found_swap:
                            if len(S) > current_size:
                                current_size = len(S)
                                steps_since_improvement = 0
                            break
                        else:
                            # Restore x_remove
                            S.add(x_remove)
                            for a, b, c in aps[x_remove]:
                                if a in S and b in S:
                                    conflict[c] += 1
                                if a in S and c in S:
                                    conflict[b] += 1
                                if b in S and c in S:
                                    conflict[a] += 1
                    
                    if not found_swap:
                        steps_since_improvement += 1
                        # If stuck for too long, remove a random element
                        if steps_since_improvement > 100 and S:
                            x_remove = random.choice(list(S))
                            S.discard(x_remove)
                            for a, b, c in aps[x_remove]:
                                if a in S and b in S:
                                    conflict[c] -= 1
                                if a in S and c in S:
                                    conflict[b] -= 1
                                if b in S and c in S:
                                    conflict[a] -= 1
                            steps_since_improvement = 0
        
        return S, conflict

    # Multiple restarts
    while time.time() < deadline:
        # Different initialization strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy < 0.6:
            # Reverse order (favor larger numbers)
            order = list(range(N, 0, -1))
            random.shuffle(order[:100])  # Slight randomization at start
        else:
            # Sorted order (favor smaller numbers)
            order = list(range(1, N + 1))
            random.shuffle(order[400:])  # Slight randomization at end
        
        S, conflict = greedy_init(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Local search
        S, conflict = local_search(S, conflict, max_steps=20000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # If we have time, do another round of local search
        if time.time() < deadline - 5.0:
            S, conflict = local_search(S, conflict, max_steps=20000)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()

    # Final polish: try to add any remaining elements
    if time.time() < deadline - 1.0:
        conflict = [0] * (N + 1)
        S = set(best_set)
        for x in S:
            for a, b, c in aps[x]:
                if a in S and b in S:
                    conflict[c] += 1
                if a in S and c in S:
                    conflict[b] += 1
                if b in S and c in S:
                    conflict[a] += 1
        
        for x in range(1, N + 1):
            if x not in S and conflict[x] == 0:
                S.add(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        conflict[c] += 1
                    if a in S and c in S:
                        conflict[b] += 1
                    if b in S and c in S:
                        conflict[a] += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return sorted(best_set)
