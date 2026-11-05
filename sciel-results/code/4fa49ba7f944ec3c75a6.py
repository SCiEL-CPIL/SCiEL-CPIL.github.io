import time
import random

N = 256

def get_3aps():
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            if c <= N:
                aps.append((a, b, c))
    return aps

def get_4aps():
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd <= N:
                aps.append((a, b, c, dd))
    return aps

all_3aps = get_3aps()
all_4aps = get_4aps()

three_aps_containing = [[] for _ in range(N + 1)]
four_aps_containing = [[] for _ in range(N + 1)]

for ap in all_3aps:
    for x in ap:
        three_aps_containing[x].append(ap)

for ap in all_4aps:
    for x in ap:
        four_aps_containing[x].append(ap)

def count_3aps(A):
    count = 0
    for ap in all_3aps:
        if all(x in A for x in ap):
            count += 1
    return count

def greedy_construct():
    A = set()
    remaining = list(range(1, N + 1))
    
    while remaining:
        best_x = -1
        best_score = -1
        for x in remaining:
            valid = True
            for ap in four_aps_containing[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            
            if not valid:
                continue
            
            gain = 0
            for ap in three_aps_containing[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    gain += 1
            
            if gain > best_score:
                best_score = gain
                best_x = x
        
        if best_x == -1:
            break
        
        A.add(best_x)
        remaining.remove(best_x)
        
    return A

def local_search(A, time_limit):
    start_time = time.time()
    A = set(A)
    current_score = count_3aps(A)
    
    # 1. Additions
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in A:
                continue
            valid = True
            for ap in four_aps_containing[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            if valid:
                A.add(x)
                current_score = count_3aps(A)
                changed = True
                break

    # 2. Swaps
    improved = True
    while improved:
        improved = False
        for u in list(A):
            for v in range(1, N + 1):
                if v in A:
                    continue
                
                valid = True
                for ap in four_aps_containing[v]:
                    others = [y for y in ap if y != v]
                    if all(y in A for y in others):
                        valid = False
                        break
                            
                if not valid:
                    continue
                
                gain = 0
                for ap in three_aps_containing[v]:
                    others = [y for y in ap if y != v]
                    if all(y in A for y in others) and u not in others:
                        gain += 1
                        
                loss = 0
                for ap in three_aps_containing[u]:
                    others = [y for y in ap if y != u]
                    if all(y in A for y in others):
                        loss += 1
                        
                delta = gain - loss
                if delta > 0:
                    A = (A - {u}) | {v}
                    current_score += delta
                    improved = True
                    break
            if improved:
                break
            if time.time() - start_time > time_limit:
                return A, current_score
                
    return A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_A = set()
    best_score = -1
    
    A1 = greedy_construct()
    s1 = count_3aps(A1)
    if s1 > best_score:
        best_A = A1
        best_score = s1
        
    if time.time() < deadline:
        A_ls, s_ls = local_search(A1, deadline - time.time() - 5)
        if s_ls > best_score:
            best_A = A_ls
            best_score = s_ls
            
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        A_rand = set()
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        for x in remaining:
            valid = True
            for ap in four_aps_containing[x]:
                others = [y for y in ap if y != x]
                if all(y in A_rand for y in others):
                    valid = False
                    break
            if valid:
                A_rand.add(x)
        
        s_rand = count_3aps(A_rand)
        if s_rand > best_score:
            best_A = A_rand
            best_score = s_rand
            
        if time.time() < deadline - 5:
            A_ls2, s_ls2 = local_search(A_rand, deadline - time.time() - 5)
            if s_ls2 > best_score:
                best_A = A_ls2
                best_score = s_ls2
                
    return sorted(best_A)
