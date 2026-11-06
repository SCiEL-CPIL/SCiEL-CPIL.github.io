import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    N = 511
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512.0 - a)
        
    # Precompute coprime masks
    mask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in range(1, N + 1):
            if gcd(a, b) == 1:
                m |= (1 << b)
        mask[a] = m
        
    # Greedy initialization
    chosen = []
    chosen_mask = 0
    for a in range(N, 0, -1):
        if (mask[a] & chosen_mask) == chosen_mask:
            chosen.append(a)
            chosen_mask |= (1 << a)
            
    best_score = sum(w[a] for a in chosen)
    best_chosen = list(chosen)
    best_mask = chosen_mask
    
    def get_score(chosen_list):
        return sum(w[a] for a in chosen_list)

    # Precompute for each a, the list of b's coprime to a, sorted by weight desc
    # coprime_to[a] = list of b in 1..N where gcd(a,b)==1, sorted by w[b] desc
    coprime_to = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        lst = [b for b in range(1, N + 1) if (mask[a] & (1 << b)) == (1 << b)]
        lst.sort(key=lambda x: w[x], reverse=True)
        coprime_to[a] = lst

    # For 1-for-1 swap: for each u in chosen, precompute compatible candidates
    # compatible[u] = list of v not in chosen, v coprime to all other chosen, sorted by w[v] desc
    
    # Local Search with multiple swap types
    random.seed(42)
    
    def try_1_for_1(best_chosen, best_mask, best_score):
        """Try 1-for-1 swaps. Returns (new_chosen, new_mask, new_score, improved)"""
        n = len(best_chosen)
        for i in range(n):
            u = best_chosen[i]
            u_bit = 1 << u
            rest_mask = best_mask ^ u_bit
            
            # Find best v not in best_chosen such that v coprime to rest_mask
            # Use precomputed coprime_to but filter by not in best_mask
            best_v = -1
            best_v_w = w[u]  # Need to beat current
            
            # Iterate through candidates sorted by weight
            # We need v coprime to rest_mask AND v not in best_mask
            # coprime_to[u] gives us v coprime to u, but we need v coprime to ALL rest
            # So we need to check (mask[v] & rest_mask) == rest_mask
            
            # Optimization: only consider v with w[v] > w[u]
            for v in coprime_to[u]:
                if w[v] <= best_v_w + 1e-9:
                    break
                if best_mask & (1 << v):
                    continue
                if (mask[v] & rest_mask) == rest_mask:
                    best_v = v
                    best_v_w = w[v]
                    break  # First one in sorted order is best
            
            if best_v != -1 and best_v_w > w[u] + 1e-9:
                new_chosen = list(best_chosen)
                new_chosen[i] = best_v
                new_mask = best_mask ^ u_bit | (1 << best_v)
                new_score = best_score - w[u] + w[best_v]
                if new_score > best_score + 1e-9:
                    return new_chosen, new_mask, new_score, True
        
        return best_chosen, best_mask, best_score, False

    def try_1_for_2(best_chosen, best_mask, best_score):
        """Try 1-for-2 swaps. Returns (new_chosen, new_mask, new_score, improved)"""
        n = len(best_chosen)
        for i in range(n):
            u = best_chosen[i]
            u_bit = 1 << u
            rest_mask = best_mask ^ u_bit
            
            # Find candidates coprime to rest_mask, not in best_mask
            candidates = []
            for x in range(1, N + 1):
                if best_mask & (1 << x):
                    continue
                if (mask[x] & rest_mask) == rest_mask:
                    candidates.append(x)
            
            if len(candidates) < 2:
                continue
            
            # Sort by weight desc
            candidates.sort(key=lambda x: w[x], reverse=True)
            
            # Find pair v, w_node with gcd(v,w_node)==1 and w[v]+w[w_node] > w[u]
            for idx_v in range(len(candidates)):
                v = candidates[idx_v]
                # Early termination: even with best possible partner, can't beat
                if idx_v + 1 < len(candidates):
                    max_possible = w[v] + w[candidates[idx_v + 1]]
                else:
                    max_possible = w[v]
                if max_possible <= w[u] + 1e-9:
                    break
                for idx_w in range(idx_v + 1, len(candidates)):
                    w_node = candidates[idx_w]
                    if w[v] + w[w_node] <= w[u] + 1e-9:
                        break
                    if (mask[v] & (1 << w_node)) == (1 << w_node):
                        new_chosen = list(best_chosen)
                        new_chosen.pop(i)
                        new_chosen.append(v)
                        new_chosen.append(w_node)
                        new_mask = rest_mask | (1 << v) | (1 << w_node)
                        new_score = best_score - w[u] + w[v] + w[w_node]
                        if new_score > best_score + 1e-9:
                            return new_chosen, new_mask, new_score, True
                        break  # Only need to find one improving pair for this u
        
        return best_chosen, best_mask, best_score, False

    def try_2_for_2(best_chosen, best_mask, best_score):
        """Try 2-for-2 swaps. Returns (new_chosen, new_mask, new_score, improved)"""
        n = len(best_chosen)
        if n < 2:
            return best_chosen, best_mask, best_score, False
        
        for i in range(n):
            for j in range(i + 1, n):
                u1 = best_chosen[i]
                u2 = best_chosen[j]
                u1_bit = 1 << u1
                u2_bit = 1 << u2
                rest_mask = best_mask ^ u1_bit ^ u2_bit
                target = w[u1] + w[u2]
                
                # Find candidates coprime to rest_mask, not in best_mask
                candidates = []
                for x in range(1, N + 1):
                    if best_mask & (1 << x):
                        continue
                    if (mask[x] & rest_mask) == rest_mask:
                        candidates.append(x)
                
                if len(candidates) < 2:
                    continue
                
                candidates.sort(key=lambda x: w[x], reverse=True)
                
                # Find pair v, w_node with gcd(v,w_node)==1 and w[v]+w[w_node] > target
                for idx_v in range(len(candidates)):
                    v = candidates[idx_v]
                    if idx_v + 1 < len(candidates):
                        max_possible = w[v] + w[candidates[idx_v + 1]]
                    else:
                        max_possible = w[v]
                    if max_possible <= target + 1e-9:
                        break
                    for idx_w in range(idx_v + 1, len(candidates)):
                        w_node = candidates[idx_w]
                        if w[v] + w[w_node] <= target + 1e-9:
                            break
                        if (mask[v] & (1 << w_node)) == (1 << w_node):
                            new_chosen = list(best_chosen)
                            # Remove u1 and u2
                            # Need to remove carefully since indices may shift
                            if i < j:
                                new_chosen.pop(j)
                                new_chosen.pop(i)
                            else:
                                new_chosen.pop(i)
                                new_chosen.pop(j)
                            new_chosen.append(v)
                            new_chosen.append(w_node)
                            new_mask = rest_mask | (1 << v) | (1 << w_node)
                            new_score = best_score - w[u1] - w[u2] + w[v] + w[w_node]
                            if new_score > best_score + 1e-9:
                                return new_chosen, new_mask, new_score, True
                            break
        
        return best_chosen, best_mask, best_score, False

    # Main search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps
        new_chosen, new_mask, new_score, imp = try_1_for_1(best_chosen, best_mask, best_score)
        if imp:
            best_chosen, best_mask, best_score = new_chosen, new_mask, new_score
            improved = True
            continue
        
        # Try 1-for-2 swaps
        new_chosen, new_mask, new_score, imp = try_1_for_2(best_chosen, best_mask, best_score)
        if imp:
            best_chosen, best_mask, best_score = new_chosen, new_mask, new_score
            improved = True
            continue
        
        # Try 2-for-2 swaps
        new_chosen, new_mask, new_score, imp = try_2_for_2(best_chosen, best_mask, best_score)
        if imp:
            best_chosen, best_mask, best_score = new_chosen, new_mask, new_score
            improved = True
            continue
        
        # Stuck in local optimum - try random perturbation
        if time.time() < deadline - 10.0:
            # Randomly remove 1-3 elements and re-greedy fill
            n_remove = random.randint(1, min(3, len(best_chosen)))
            removed = random.sample(best_chosen, n_remove)
            remaining = [a for a in best_chosen if a not in removed]
            remaining_mask = 0
            for a in remaining:
                remaining_mask |= (1 << a)
            
            # Greedy fill with high-weight candidates
            for a in range(N, 0, -1):
                if remaining_mask & (1 << a):
                    continue
                if (mask[a] & remaining_mask) == remaining_mask:
                    remaining.append(a)
                    remaining_mask |= (1 << a)
            
            new_score = sum(w[a] for a in remaining)
            if new_score > best_score:
                best_chosen = remaining
                best_mask = remaining_mask
                best_score = new_score
                improved = True
                continue
    
    return best_chosen
