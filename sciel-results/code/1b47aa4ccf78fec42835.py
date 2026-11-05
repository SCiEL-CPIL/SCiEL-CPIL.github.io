import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 16
    k = 4
    
    all_sets = []
    for combo in itertools.combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    pairs = list(itertools.combinations(range(n), 2))
    triples = list(itertools.combinations(range(n), 3))
    
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    set_pairs = []
    set_triples = []
    for s in all_sets:
        elems = [i for i in range(n) if (s >> i) & 1]
        ps = [pair_to_idx[p] for p in itertools.combinations(elems, 2)]
        ts = [triple_to_idx[t] for t in itertools.combinations(elems, 3)]
        set_pairs.append(ps)
        set_triples.append(ts)
        
    pair_to_mask = [0] * len(pairs)
    for i, p in enumerate(pairs):
        m = 0
        for x in p:
            m |= (1 << x)
        pair_to_mask[i] = m

    best_family = []
    best_size = 0
    
    random.seed(42)
    
    def _run_greedy(max_time):
        nonlocal best_family, best_size
        
        count3 = [0] * len(triples)
        L_pair = [[] for _ in range(len(pairs))]
        family_indices = []
        
        candidates = list(range(len(all_sets)))
        random.shuffle(candidates)
        
        start = time.time()
        while candidates and (time.time() - start) < max_time:
            s_idx = candidates.pop()
            s = all_sets[s_idx]
            
            # Check 3-sets
            ok = True
            for t_idx in set_triples[s_idx]:
                if count3[t_idx] >= 2:
                    ok = False
                    break
            if not ok:
                continue
                
            # Check 2-sets
            for p_idx in set_pairs[s_idx]:
                u_mask = pair_to_mask[p_idx]
                p_s = s ^ u_mask
                
                # Get sets in family containing this pair
                cand_list = L_pair[p_idx]
                
                # Find compatible sets (disjoint petals with S)
                compat = []
                for a_idx in cand_list:
                    a = all_sets[a_idx]
                    if (a & p_s) == 0:
                        compat.append(a_idx)
                        
                if len(compat) >= 2:
                    # Check if any two in compat have disjoint petals
                    # i.e., A & B == u_mask
                    for i in range(len(compat)):
                        a = all_sets[compat[i]]
                        for j in range(i + 1, len(compat)):
                            b = all_sets[compat[j]]
                            if (a & b) == u_mask:
                                ok = False
                                break
                        if not ok:
                            break
                if not ok:
                    break
                    
            if ok:
                family_indices.append(s_idx)
                for t_idx in set_triples[s_idx]:
                    count3[t_idx] += 1
                for p_idx in set_pairs[s_idx]:
                    L_pair[p_idx].append(s_idx)
                
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family = list(family_indices)
                    
        return len(family_indices)

    while time.time() - start_time < deadline:
        _run_greedy(max_time=5.0)
        
    result = []
    for idx in best_family:
        s = all_sets[idx]
        elems = [i+1 for i in range(n) if (s >> i) & 1]
        result.append(elems)
        
    return result
