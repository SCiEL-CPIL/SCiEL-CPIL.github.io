import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    
    # Generate STS(15) using PG(3,2) construction
    sts_triples = []
    for i in range(1, 16):
        for j in range(i+1, 16):
            k = i ^ j
            if k > j:
                sts_triples.append((i-1, j-1, k-1))
    
    def is_linear(triples):
        pair_set = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(t[i], t[j]), max(t[i], t[j]))
                    if p in pair_set:
                        return False
                    pair_set.add(p)
        return True
    
    def covered_pairs_count(triples):
        pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    pairs.add((min(t[i], t[j]), max(t[i], t[j])))
        return len(pairs)
    
    def independence_number(triples):
        triple_set = set(triples)
        for size in range(8, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s = set(subset)
                if not any(t[0] in s and t[1] in s and t[2] in s for t in triple_set):
                    return size
        return 1
    
    def build_system(M):
        M_set = set(M)
        final_triples = []
        for t in sts_triples:
            pairs = [(min(t[0],t[1]),max(t[0],t[1])), (min(t[0],t[2]),max(t[0],t[2])), (min(t[1],t[2]),max(t[1],t[2]))]
            if not any(p in M_set for p in pairs):
                final_triples.append(list(t))
        for p in M:
            final_triples.append([15, p[0], p[1]])
        return final_triples
    
    def local_search_matching(M, max_iter=5000):
        current_M = M[:]
        current_cp = covered_pairs_count(build_system(current_M))
        best_M = current_M[:]
        best_cp = current_cp
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            idx = random.randint(0, len(current_M) - 1)
            old_pair = current_M[idx]
            
            candidates = []
            for a in range(15):
                for b in range(a+1, 15):
                    if (a, b) == old_pair:
                        continue
                    disjoint = True
                    for j, p in enumerate(current_M):
                        if j == idx:
                            continue
                        if a in p or b in p:
                            disjoint = False
                            break
                    if disjoint:
                        candidates.append((a, b))
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            for new_pair in candidates[:3]:
                new_M = current_M[:]
                new_M[idx] = new_pair
                new_cp = covered_pairs_count(build_system(new_M))
                
                if new_cp > current_cp:
                    current_M = new_M
                    current_cp = new_cp
                    if current_cp > best_cp:
                        best_cp = current_cp
                        best_M = current_M[:]
                    break
                elif new_cp == current_cp and random.random() < 0.1:
                    current_M = new_M
                    break
        
        return best_M
    
    best_result = None
    best_alpha = float('inf')
    best_cp = -1
    candidate_systems = []
    
    for restart in range(100):
        if time.time() > deadline - 20:
            break
        
        pts = list(range(15))
        random.shuffle(pts)
        M = [tuple(sorted((pts[i], pts[i+1]))) for i in range(0, 14, 2)]
        
        searched_M = local_search_matching(M, max_iter=3000)
        system = build_system(searched_M)
        
        cp = covered_pairs_count(system)
        if cp > best_cp:
            best_cp = cp
            candidate_systems = [system]
        elif cp == best_cp:
            candidate_systems.append(system)
    
    # Evaluate top candidates
    candidate_systems.sort(key=lambda s: -covered_pairs_count(s))
    for system in candidate_systems[:10]:
        if time.time() > deadline - 5:
            break
        alpha = independence_number(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_result = system
    
    if best_result is None:
        return []
    
    return best_result

construct()
