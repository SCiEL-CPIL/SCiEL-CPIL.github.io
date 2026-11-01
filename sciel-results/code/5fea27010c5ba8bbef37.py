import time, random, sys
from collections import defaultdict

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _build_greedy(n, seed=0):
    """Greedy longest-overlap superpermutation via integer encoding."""
    rng = random.Random(seed)
    import itertools
    
    all_perms = list(itertools.permutations(range(1, n+1)))
    
    def perm_to_int(p):
        val = 0
        for d in p:
            val = val * n + d
        return val

    int_to_perm = {}
    for p in all_perms:
        int_to_perm[perm_to_int(p)] = p

    word = list(all_perms[0])
    covered = set([perm_to_int(all_perms[0])])
    active = set(perm_to_int(p) for p in all_perms)
    active.discard(perm_to_int(all_perms[0]))
    
    n_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        n_powers[i] = n_powers[i-1] * n

    while active:
        if _remaining() < 1.0:
            break
        
        w_suffix = [0] * n
        v = 0
        for i in range(1, n):
            v = word[-i] + v * n
            w_suffix[i] = v
        
        best_p_int = None
        best_k = -1
        
        for p_int in active:
            for k in range(n-1, 0, -1):
                p_pref = p_int // n_powers[n-k]
                if p_pref == w_suffix[k]:
                    if k > best_k:
                        best_k = k
                        best_p_int = p_int
                    break
        
        if best_p_int is None:
            best_p_int = next(iter(active))
            best_k = 0
            
        best_p_tuple = int_to_perm[best_p_int]
        word.extend(best_p_tuple[best_k:])
        active.discard(best_p_int)
        
        new_len = len(word)
        start_from = max(0, new_len - 2*n + best_k + 1)
        for s in range(start_from, new_len - n + 1):
            w_val = 0
            for i in range(s, s+n):
                w_val = w_val * n + word[i]
            if w_val in active:
                active.discard(w_val)
                
    return word

def construct(n):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 150.0

    if n <= 1:
        return "1"

    # Only do one restart to ensure we finish in time
    word = _build_greedy(n, seed=0)
    
    return "".join(map(str, word))
