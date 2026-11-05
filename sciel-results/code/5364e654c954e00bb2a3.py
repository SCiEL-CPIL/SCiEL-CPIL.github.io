import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 55.0

def compute_structures(S_list):
    n = len(S_list)
    triple_sums = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple_sums.add(S_list[i] + S_list[j] + S_list[k])
    pair_sums_by_elem = {}
    for i, a in enumerate(S_list):
        pairs = set()
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                pairs.add(S_list[j] + S_list[k])
        pair_sums_by_elem[a] = pairs
    return triple_sums, pair_sums_by_elem

def can_add(S_list, x, triple_sums, pair_sums_by_elem):
    for t in triple_sums:
        if t % x == 0:
            return False
    for a in S_list:
        for p in pair_sums_by_elem[a]:
            if (x + p) % a == 0:
                return False
    return True

def construct():
    global START_TIME
    START_TIME = time.time()
    best_size = 0
    best_list = []
    numbers = list(range(1, 193))
    
    while time.time() - START_TIME < TIME_LIMIT:
        current_list = []
        triple_sums = set()
        pair_sums_by_elem = {}
        
        large = [x for x in numbers if x > 96]
        random.shuffle(large)
        for x in large:
            if can_add(current_list, x, triple_sums, pair_sums_by_elem):
                current_list.append(x)
                n = len(current_list) - 1
                for i in range(n):
                    for j in range(i+1, n):
                        triple_sums.add(x + current_list[i] + current_list[j])
                for a in current_list[:-1]:
                    for b in current_list[:-1]:
                        if b != a:
                            pair_sums_by_elem[a].add(x + b)
                pf = set()
                for i in range(n):
                    for j in range(i+1, n):
                        pf.add(current_list[i] + current_list[j])
                pair_sums_by_elem[x] = pf
        
        small = [x for x in numbers if x <= 96 and x not in set(current_list)]
        random.shuffle(small)
        for x in small:
            if can_add(current_list, x, triple_sums, pair_sums_by_elem):
                current_list.append(x)
                n = len(current_list) - 1
                for i in range(n):
                    for j in range(i+1, n):
                        triple_sums.add(x + current_list[i] + current_list[j])
                for a in current_list[:-1]:
                    for b in current_list[:-1]:
                        if b != a:
                            pair_sums_by_elem[a].add(x + b)
                pf = set()
                for i in range(n):
                    for j in range(i+1, n):
                        pf.add(current_list[i] + current_list[j])
                pair_sums_by_elem[x] = pf
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_list = list(current_list)
            if best_size == 48:
                break
        
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            for i in range(len(current_list)):
                if time.time() - START_TIME >= TIME_LIMIT:
                    break
                removed = current_list[i]
                temp_list = [x for x in current_list if x != removed]
                temp_set = set(temp_list)
                temp_triple, temp_pairs = compute_structures(temp_list)
                
                remaining = [x for x in numbers if x not in temp_set and x != removed]
                random.shuffle(remaining)
                
                added = 0
                for x in remaining:
                    if can_add(temp_list, x, temp_triple, temp_pairs):
                        temp_list.append(x)
                        temp_set.add(x)
                        added += 1
                        n = len(temp_list) - 1
                        for a in range(n):
                            for b in range(a+1, n):
                                temp_triple.add(x + temp_list[a] + temp_list[b])
                        for a in temp_list[:-1]:
                            for b in temp_list[:-1]:
                                if b != a:
                                    temp_pairs[a].add(x + b)
                        pf = set()
                        for a in range(n):
                            for b in range(a+1, n):
                                pf.add(temp_list[a] + temp_list[b])
                        temp_pairs[x] = pf
                        if added >= 2:
                            break
                
                if added >= 2:
                    current_list = temp_list
                    triple_sums = temp_triple
                    pair_sums_by_elem = temp_pairs
                    improved = True
                    break
            
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_list = list(current_list)
                if best_size == 48:
                    break
        
        if best_size == 48:
            break
    
    return best_list
