import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all possible triples
    triple_mask_cache = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_mask_cache[t] = m
    
    def indep_num(triples):
        tm_list = [triple_mask_cache[tuple(t)] for t in triples]
        best = 0
        for mask in range(1 << n):
            bits = bin(mask).count('1')
            if bits <= best:
                continue
            has_triple = False
            for tm in tm_list:
                if (mask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                best = bits
        return best
    
    def is_linear(triples, new_t):
        nt_set = set(new_t)
        for t in triples:
            if len(nt_set & set(t)) > 1:
                return False
        return True
    
    def greedy_build():
        triples = []
        for _ in range(300):
            random.shuffle(all_triples)
            added = False
            for t in all_triples:
                if is_linear(triples, t):
                    triples.append(list(t))
                    added = True
                    break
            if not added:
                break
        return triples
    
    best_triples = None
    best_indep = n
    
    while time.time() < deadline:
        triples = greedy_build()
        cur_indep = indep_num(triples)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(triples)
            for i in range(len(triples)):
                old = triples[i]
                rest = triples[:i] + triples[i+1:]
                for _ in range(20):
                    nt = list(random.sample(points, 3))
                    if is_linear(rest, nt):
                        triples[i] = nt
                        ni = indep_num(triples)
                        if ni < cur_indep:
                            cur_indep = ni
                            improved = True
                            break
                        else:
                            triples[i] = old
                if improved:
                    break
        
        if cur_indep < best_indep:
            best_indep = cur_indep
            best_triples = triples[:]
    
    return best_triples
