import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    max_val = 256
    S = list(range(2, 129))  # Small numbers: 2..128
    L = list(range(129, 257))  # Large numbers: 129..256
    
    # Precompute compatibility and blocked sets
    # B_s: list of l in L such that lcm(s, l) <= 256
    B_s = {}
    for s in S:
        blocked = []
        for l in L:
            if math.lcm(s, l) <= max_val:
                blocked.append(l)
        B_s[s] = blocked
    
    # C_s: list of u in S such that lcm(s, u) <= 256 (conflicts within S)
    C_s = {}
    for s in S:
        conflicts = []
        for u in S:
            if s != u and math.lcm(s, u) <= max_val:
                conflicts.append(u)
        C_s[s] = conflicts
    
    # Precompute reciprocal values
    recip = {x: 1.0 / x for x in S + L}
    
    # Base score from L (if S' is empty)
    base_L_score = sum(recip[l] for l in L)
    
    def get_score(current_set, current_blocked):
        s_score = sum(recip[s] for s in current_set)
        l_score = base_L_score - sum(recip[l] for l in current_blocked)
        return s_score + l_score
    
    def is_compatible(s, current_set):
        for u in current_set:
            if math.lcm(s, u) <= max_val:
                return False
        return True
    
    def run_greedy(order=None):
        if order is None:
            order = S[:]
        
        current_set = []
        current_blocked = set()
        remaining = set(order)
        
        while True:
            best_gain = -1e9
            best_s = None
            
            for s in remaining:
                if not is_compatible(s, current_set):
                    continue
                
                new_blocked = set(B_s[s]) - current_blocked
                loss = sum(recip[l] for l in new_blocked)
                gain = recip[s] - loss
                
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            
            if best_s is None or best_gain < -1e-9:
                break
            
            current_set.append(best_s)
            current_blocked.update(B_s[best_s])
            remaining.remove(best_s)
        
        return current_set, current_blocked
    
    def local_search(initial_set, initial_blocked):
        current_set = initial_set[:]
        current_blocked = initial_blocked.copy()
        current_set_set = set(current_set)
        
        improved = True
        while improved:
            improved = False
            
            # Try adding
            for s in S:
                if s in current_set_set:
                    continue
                if not is_compatible(s, current_set):
                    continue
                
                new_blocked = set(B_s[s]) - current_blocked
                loss = sum(recip[l] for l in new_blocked)
                gain = recip[s] - loss
                
                if gain > 1e-9:
                    current_set.append(s)
                    current_set_set.add(s)
                    current_blocked.update(new_blocked)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try removing
            for i in range(len(current_set)):
                s = current_set[i]
                others_blocked = set()
                for u in current_set:
                    if u != s:
                        others_blocked.update(B_s[u])
                
                only_blocked_by_s = set(B_s[s]) - others_blocked
                gain = -recip[s] + sum(recip[l] for l in only_blocked_by_s)
                
                if gain > 1e-9:
                    current_set.pop(i)
                    current_set_set.remove(s)
                    current_blocked = set()
                    for u in current_set:
                        current_blocked.update(B_s[u])
                    improved = True
                    break
            
            if improved:
                continue

            # Try swapping
            for i in range(len(current_set)):
                s1 = current_set[i]
                for s2 in S:
                    if s2 in current_set_set:
                        continue
                    
                    compatible = True
                    for u in current_set:
                        if u == s1:
                            continue
                        if math.lcm(s2, u) <= max_val:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    others_blocked = set()
                    for u in current_set:
                        if u != s1:
                            others_blocked.update(B_s[u])
                    
                    new_blocked = others_blocked | set(B_s[s2])
                    
                    old_loss = sum(recip[l] for l in current_blocked)
                    new_loss = sum(recip[l] for l in new_blocked)
                    
                    delta = (recip[s2] - recip[s1]) - (new_loss - old_loss)
                    
                    if delta > 1e-9:
                        current_set[i] = s2
                        current_set_set.remove(s1)
                        current_set_set.add(s2)
                        current_blocked = new_blocked
                        improved = True
                        break
                if improved:
                    break
                    
        return current_set, current_blocked

    # Run greedy and then local search
    g_set, g_blocked = run_greedy()
    ls_set, ls_blocked = local_search(g_set, g_blocked)
    
    best_score = get_score(ls_set, ls_blocked)
    best_set = ls_set[:]
    
    # Try random restarts
    random.seed(42)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Random valid set generation
        rand_set = []
        rand_blocked = set()
        shuffled = random.sample(S, len(S))
        for s in shuffled:
            if is_compatible(s, rand_set):
                rand_set.append(s)
                rand_blocked.update(B_s[s])
                
        ls_set2, ls_blocked2 = local_search(rand_set, rand_blocked)
        score2 = get_score(ls_set2, ls_blocked2)
        
        if score2 > best_score:
            best_score = score2
            best_set = ls_set2[:]
    
    # Try 2-opt style swaps more aggressively
    if time.time() < deadline - 5.0:
        # Run local search again with more aggressive swapping
        final_set, final_blocked = local_search(best_set, 
            set().union(*[set(B_s[s]) for s in best_set]) if best_set else set())
        final_score = get_score(final_set, final_blocked)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    # Construct final result
    final_S = best_set
    final_L = [l for l in L if all(math.lcm(s, l) > max_val for s in final_S)]
    
    final_result = final_S + final_L
    
    return final_result
