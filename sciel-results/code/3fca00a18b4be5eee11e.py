import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_total = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    pairings_6 = [
        [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
    ]
    
    k222_triples = []  # list of list of 8 triple indices
    triple_to_k222s = [[] for _ in range(num_total)]
    
    for six_pts in itertools.combinations(range(N), 6):
        for pairing in pairings_6:
            pairs = [(six_pts[p1], six_pts[p2]) for p1, p2 in pairing]
            transversals = []
            for mask in range(8):
                tri = tuple(sorted([pairs[i][1] if (mask >> i) & 1 else pairs[i][0] for i in range(3)]))
                transversals.append(triple_to_idx[tri])
            k222_idx = len(k222_triples)
            k222_triples.append(transversals)
            for t_idx in transversals:
                triple_to_k222s[t_idx].append(k222_idx)
    
    num_k222 = len(k222_triples)
    
    present = np.zeros(num_total, dtype=bool)
    k222_counts = np.zeros(num_k222, dtype=np.int32)
    
    def can_add(idx):
        if present[idx]:
            return False
        for k_idx in triple_to_k222s[idx]:
            if k222_counts[k_idx] >= 7:
                return False
        return True
    
    def do_add(idx):
        present[idx] = True
        for k_idx in triple_to_k222s[idx]:
            k222_counts[k_idx] += 1
    
    def do_remove(idx):
        present[idx] = False
        for k_idx in triple_to_k222s[idx]:
            k222_counts[k_idx] -= 1
    
    # Greedy
    random.seed(123)
    order = list(range(num_total))
    random.shuffle(order)
    
    for idx in order:
        if time.time() > deadline - 40:
            break
        if can_add(idx):
            do_add(idx)
    
    best_count = int(np.sum(present))
    
    # Local search
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 3 and no_improve < max_no_improve:
        improved = False
        
        # Try pure additions
        for _ in range(3000):
            idx = random.randint(0, num_total - 1)
            if not present[idx] and can_add(idx):
                do_add(idx)
                improved = True
                break
        
        if improved:
            no_improve = 0
            continue
        
        # Try remove 1, add 2 (net +1)
        found_swap = False
        for _ in range(800):
            present_idx = np.where(present)[0]
            if len(present_idx) == 0:
                break
            rem_idx = present_idx[random.randint(0, len(present_idx) - 1)]
            
            # Find candidates unblocked by removing rem_idx
            rem_k222s = set(triple_to_k222s[rem_idx])
            candidates = set()
            
            for k_idx in triple_to_k222s[rem_idx]:
                for t_idx in k222_triples[k_idx]:
                    if not present[t_idx]:
                        # Check if blocked only by K(2,2,2)s that contain rem_idx
                        all_blocked_by_rem = True
                        for kk in triple_to_k222s[t_idx]:
                            if k222_counts[kk] >= 7 and kk not in rem_k222s:
                                all_blocked_by_rem = False
                                break
                        if all_blocked_by_rem:
                            candidates.add(t_idx)
            
            if len(candidates) >= 2:
                cand_list = list(candidates)
                random.shuffle(cand_list)
                for i in range(min(8, len(cand_list))):
                    for j in range(i + 1, min(8, len(cand_list))):
                        c1, c2 = cand_list[i], cand_list[j]
                        # Try the swap
                        do_remove(rem_idx)
                        ok = True
                        added = []
                        for c in [c1, c2]:
                            if can_add(c):
                                do_add(c)
                                added.append(c)
                            else:
                                ok = False
                                break
                        if ok:
                            found_swap = True
                            improved = True
                            break
                        else:
                            for c in added:
                                do_remove(c)
                            do_add(rem_idx)
                    if found_swap:
                        break
            
            if found_swap:
                break
        
        if found_swap:
            no_improve = 0
            continue
        
        # Try remove 1, add 3 (net +2)
        for _ in range(300):
            present_idx = np.where(present)[0]
            if len(present_idx) == 0:
                break
            rem_idx = present_idx[random.randint(0, len(present_idx) - 1)]
            rem_k222s = set(triple_to_k222s[rem_idx])
            candidates = set()
            
            for k_idx in triple_to_k222s[rem_idx]:
                for t_idx in k222_triples[k_idx]:
                    if not present[t_idx]:
                        all_blocked_by_rem = True
                        for kk in triple_to_k222s[t_idx]:
                            if k222_counts[kk] >= 7 and kk not in rem_k222s:
                                all_blocked_by_rem = False
                                break
                        if all_blocked_by_rem:
                            candidates.add(t_idx)
            
            if len(candidates) >= 3:
                cand_list = list(candidates)
                random.shuffle(cand_list)
                for i in range(min(5, len(cand_list))):
                    for j in range(i + 1, min(5, len(cand_list))):
                        for k in range(j + 1, min(5, len(cand_list))):
                            c1, c2, c3 = cand_list[i], cand_list[j], cand_list[k]
                            do_remove(rem_idx)
                            ok = True
                            added = []
                            for c in [c1, c2, c3]:
                                if can_add(c):
                                    do_add(c)
                                    added.append(c)
                                else:
                                    ok = False
                                    break
                            if ok:
                                improved = True
                                break
                            else:
                                for c in added:
                                    do_remove(c)
                                do_add(rem_idx)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
    
    result = [list(all_triples[i]) for i in range(num_total) if present[i]]
    return result
