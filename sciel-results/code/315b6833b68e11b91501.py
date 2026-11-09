"""Rewritten program using minimum hitting set approach for K(2,2,2)-free triple systems."""
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all K(2,2,2) configurations (3 disjoint pairs -> 8 transversal triples)
    configs = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        for o in range(m+1, n):
                            p1, p2 = i, j
                            p3, p4 = k, l
                            p5, p6 = m, o
                            trans = [
                                tuple(sorted((p1, p3, p5))),
                                tuple(sorted((p1, p3, p6))),
                                tuple(sorted((p1, p4, p5))),
                                tuple(sorted((p1, p4, p6))),
                                tuple(sorted((p2, p3, p5))),
                                tuple(sorted((p2, p3, p6))),
                                tuple(sorted((p2, p4, p5))),
                                tuple(sorted((p2, p4, p6))),
                            ]
                            trans_idx = [triple_to_idx[t] for t in trans]
                            configs.append(trans_idx)
    
    num_configs = len(configs)
    
    # For each triple, which configs does it hit?
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
    
    # Greedy hitting set
    unsatisfied = [True] * num_configs
    in_hitting = [False] * num_triples
    hitting_set = set()
    
    # Compute initial coverage
    coverage = [0] * num_triples
    for t_idx in range(num_triples):
        for c_idx in triple_to_configs[t_idx]:
            coverage[t_idx] += 1
    
    while True:
        best_t = -1
        best_cov = -1
        for t_idx in range(num_triples):
            if in_hitting[t_idx]:
                continue
            cov = sum(1 for c_idx in triple_to_configs[t_idx] if unsatisfied[c_idx])
            if cov > best_cov:
                best_cov = cov
                best_t = t_idx
        if best_t == -1 or best_cov == 0:
            break
        in_hitting[best_t] = True
        hitting_set.add(best_t)
        for c_idx in triple_to_configs[best_t]:
            unsatisfied[c_idx] = False
    
    # Local improvement: try to remove each hitting set member
    improved = True
    while improved:
        improved = False
        for t_idx in list(hitting_set):
            hitting_set.discard(t_idx)
            ok = True
            for config in configs:
                if not any(t in hitting_set for t in config):
                    ok = False
                    break
            if ok:
                improved = True
            else:
                hitting_set.add(t_idx)
    
    result = [list(t) for i, t in enumerate(all_triples) if i not in hitting_set]
    return result
