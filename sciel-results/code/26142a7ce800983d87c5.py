import random
import time

N = 256

def _is_sidon(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    Sigma = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            Sigma.add(S[i] + S[j])
    
    for z in range(1, N + 1):
        if z in S:
            continue
        # Check if z can be added
        # New sums: 2*z, z+s for s in S
        # If any of these are in Sigma, it's blocked.
        if 2 * z in Sigma:
            continue
        
        blocked = False
        for s in S:
            if z + s in Sigma:
                blocked = True
                break
        
        if not blocked:
            return False
    
    return True

def _greedy_construct(seed):
    random.seed(seed)
    S = []
    Sigma = set()
    available = set(range(1, N + 1))
    
    def is_sidon_with_candidate(x, S, Sigma):
        new_sums = [x + s for s in S] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            return False
        for s in new_sums:
            if s in Sigma:
                return False
        return True

    def count_blocked_by_candidate(x, S, Sigma, available):
        if not is_sidon_with_candidate(x, S, Sigma):
            return -1
        
        new_S = S + [x]
        new_Sigma = Sigma | set([x + s for s in S] + [2 * x])
        
        blocked_count = 0
        for z in available:
            if z == x:
                continue
            
            if 2 * z in new_Sigma:
                blocked_count += 1
                continue
            
            blocked = False
            for s in new_S:
                if z + s in new_Sigma:
                    blocked = True
                    break
            
            if blocked:
                blocked_count += 1
        
        return blocked_count

    def is_blocked(z, S, Sigma):
        if 2 * z in Sigma:
            return True
        for s in S:
            if z + s in Sigma:
                return True
        return False

    while available:
        best_x = None
        best_gain = -1
        
        candidates = list(available)
        random.shuffle(candidates)
        
        for x in candidates:
            gain = count_blocked_by_candidate(x, S, Sigma, available)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            break
        
        S.append(best_x)
        new_sums = [best_x + s for s in S[:-1]] + [2 * best_x]
        Sigma.update(new_sums)
        available.remove(best_x)
        
        new_available = set()
        for z in available:
            if not is_blocked(z, S, Sigma):
                new_available.add(z)
        available = new_available
    
    return S

def construct():
    best_S = None
    best_size = N + 1
    
    # Try multiple seeds to find a smaller maximal Sidon set
    for seed in range(200):
        S = _greedy_construct(seed)
        if _is_sidon(S) and _is_maximal(S, N):
            if len(S) < best_size:
                best_size = len(S)
                best_S = S
    
    if best_S is None:
        # Fallback
        S = _greedy_construct(0)
        return S
    
    return best_S
